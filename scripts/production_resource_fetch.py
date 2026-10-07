"""Manual full-audio acquisition only: frozen latest lessons, no model or publication."""
from contextlib import redirect_stdout, redirect_stderr
import base64
import io
import json
import os
from pathlib import Path
import sys
import time

from scripts import production_qwen as pipeline
from scripts import sharded_qwen_pilot as shards
from scripts.parallel_courses import configured_courses
from scripts.production_db import load_remote
from scripts.production_result_export import encrypt
from src.api.auth_recovery import authenticated_session
from src.api.icourse import ICourseClient
from src.api.webvpn import WebVPNSession
from src.data.database import Database
from src.runtime.audio_preparation import PREPARE_STREAM_TIMEOUT, PREPARE_IDLE_TIMEOUT, validate_prepared_audio
from src.runtime.scheduler import AudioDownloader


def check_request():
    if os.environ.get('GITHUB_RUN_ATTEMPT', '1') != '1':
        raise ValueError('Resource rerun requires fresh explicit authorization')
    recipient = base64.b64decode(os.environ['RECIPIENT_PUBLIC_KEY'], validate=True)
    if len(recipient) != 32:
        raise ValueError('Invalid recipient key')
    return recipient


def save_result(row, slot):
    recipient = check_request()
    pipeline.out('resource-result.enc').write_bytes(encrypt(
        shards.encoded(row), recipient, os.environ['GITHUB_RUN_ID'], slot))
    # Only the fixed safe audit is public; lesson names/IDs remain encrypted.
    pipeline.out('resource-audit.json').write_bytes(shards.encoded(row['audit']))


def select_latest(client, db, courses):
    """Resolve each subscribed course independently; unavailable is not a fake success."""
    rows = []
    for slot, course in enumerate(courses):
        row = {'task_slot': slot, 'course_id': course}
        try:
            task, teacher = pipeline.latest_validation_task(client, db, course)
            row.update(status='selected', task=task, teacher=teacher)
        except Exception as error:
            row.update(status='selection_failed', error_type=type(error).__name__)
        rows.append(row)
    return rows


def selection_audit(rows):
    return {'resource_only': True, 'course_count': len(rows), 'courses': [
        dict(task_slot=r['task_slot'], course_id=r['course_id'], status=r['status'],
             **(r['task'][2]['_validation'] if r['status'] == 'selected' else
                {'error_type': r['error_type']})) for r in rows]}


def plan():
    check_request()
    courses = configured_courses(os.environ['COURSE_IDS'])
    pipeline.out('selection-audit.json').write_bytes(shards.encoded(
        {'resource_only': True, 'course_count': len(courses), 'status': 'selecting'}))
    if len(courses) != 4:
        raise ValueError('Expected exactly four configured subscriptions')
    vpn = db = None
    sessions = []
    def factory():
        value = WebVPNSession()
        sessions.append(value)
        return value
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            load_remote(pipeline.root()/'resource-history.db')
            db = Database(str(pipeline.root()/'resource-history.db'))
            vpn = authenticated_session(factory=factory)
            rows = select_latest(ICourseClient(vpn), db, courses)
        shards.seal({'selections.json': shards.encoded(rows)}, 'resource-selection',
                    pipeline.out('selection.enc'), slot=0)
        pipeline.out('selection-export.enc').write_bytes(encrypt(
            shards.encoded({'resource_only': True, 'selections': rows}), check_request(),
            os.environ['GITHUB_RUN_ID'], 255))
        pipeline.out('selection-audit.json').write_bytes(shards.encoded(selection_audit(rows)))
        pipeline.write_outputs(tasks={'include': [{'task_slot': i} for i in range(4)]})
    finally:
        pipeline.out('authentication-audit.json').write_bytes(shards.encoded(
            {'authenticated': vpn is not None,
             'attempts': [s.auth_diagnostics for s in sessions]}))
        if db is not None: db.conn.close()
        if vpn is not None: vpn.session.close()


def wait_audio(handle, *, clock=time.monotonic, sleep=time.sleep):
    began = changed = clock()
    last_size = 0
    while handle.process.poll() is None:
        size = Path(handle.path).stat().st_size if Path(handle.path).exists() else 0
        now = clock()
        if size > last_size: last_size, changed = size, now
        if now-began > PREPARE_STREAM_TIMEOUT:
            raise TimeoutError('Audio preparation deadline exceeded')
        if now-changed > PREPARE_IDLE_TIMEOUT:
            raise TimeoutError('Audio preparation stalled')
        sleep(.5)


def fetch():
    check_request()
    slot = int(os.environ['COURSE_SLOT'])
    if not 0 <= slot < 4: raise ValueError('Invalid resource slot')
    pipeline.artifact('icourse-resource-selection', pipeline.root()/'selection', required=True)
    rows = shards.unseal(pipeline.root()/'selection'/'selection.enc', 'resource-selection', slot=0)
    selected = json.loads(rows['selections.json'])[slot]
    if selected['task_slot'] != slot: raise ValueError('Frozen resource slot differs')
    audit = dict(selection_audit([selected])['courses'][0], resource_only=True,
                 model_calls=0, publication=False, emailed=False)
    row = {'selection': selected, 'audit': audit}
    if selected['status'] != 'selected':
        save_result(row, slot)
        return False
    course, title, lecture = selected['task']
    spec = {'mode': 'resources_only', 'course_id': course, 'sub_id': lecture['sub_id']}
    files = {}; downloader = handle = vpn = None
    began = time.monotonic()
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            vpn = authenticated_session()
            downloader = AudioDownloader(str(pipeline.root()/'audio'), max_concurrent=1)
            downloader.schedule(ICourseClient(vpn), course, lecture['sub_id'], preserve_timestamps=True)
            handle = downloader.get(lecture['sub_id'], timeout=180)
            if handle is None: raise ValueError('No playable production audio')
            wait_audio(handle)
            # Preserve actual samples/hash/diagnostics before testing completeness.
            pipeline.retain_prepared_audio(handle, spec, files)
            validate_prepared_audio(spec)
        audit['status'] = 'complete'
    except Exception as error:
        audit.update(status='failed', error_type=type(error).__name__, error_code=pipeline.failure_code(error))
        if handle is not None:
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    pipeline.retain_prepared_audio(handle, spec, files)
            except Exception as retention_error:
                audit['retention_error_type'] = type(retention_error).__name__
    finally:
        if downloader is not None: downloader.shutdown()
        if vpn is not None: vpn.session.close()
        audit.update(seconds=time.monotonic()-began, audio_retained='lecture.flac' in files)
        row['specification'] = spec
        audit['audio_diagnostics'] = spec.get('audio_diagnostics', {})
        # Audio is protected by the existing run/task/stage-bound multipart codec.
        if files:
            files['specification.json'] = shards.encoded(spec)
            try:
                shards.seal(files, 'prepared', pipeline.out('resource-audio.enc'), slot=slot)
            except Exception as checkpoint_error:
                audit.update(status='failed', checkpoint_error_type=type(checkpoint_error).__name__)
        save_result(row, slot)
    return audit['status'] == 'complete'


def main():
    mode = sys.argv[1]
    if mode == 'plan': plan()
    elif mode == 'fetch':
        if not fetch(): raise SystemExit(1)
    else: raise ValueError('Invalid resource mode')


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print('Resource-only verification failed ('+type(error).__name__+'); private details withheld', flush=True)
        raise SystemExit(1)
