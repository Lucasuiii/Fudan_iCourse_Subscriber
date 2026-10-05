"""Formal pilot queue -> bounded ASR jobs -> LectureRunner -> scoped publication.

Private payloads are authenticated to run/task/stage. No baseline or test slots.
The CLI emits counts and exception types only, never classroom content.
"""
from __future__ import annotations
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from scripts import sharded_qwen_pilot as shards
from scripts.qwen_sharding import build_audio_plan, validate_plan, validate_result, fingerprint, MAX_TASKS
from scripts.parallel_courses import configured_courses
from scripts.production_db import load_remote, snapshot, lecture_snapshot, merge_lecture, publish
from src.data.database import Database


def root():
    return shards.root()


def out(name):
    directory = root()/'out'
    directory.mkdir(mode=0o700, exist_ok=True)
    return directory/name


def decode(path, role):
    return shards.unseal(Path(path), role, slot=0 if role == 'queue' else None)


def encode(files, role, path):
    shards.seal(files, role, Path(path), slot=0 if role == 'queue' else None)


def read_json(blob):
    return json.loads(blob)


def failure_code(error):
    # Whitelisted internal reasons only; provider/media exceptions may contain
    # private URLs or credentials and must never be printed verbatim.
    reasons = {
        'A later finalization lost its quota checkpoint; automatic refund forbidden': 'quota_checkpoint_lost',
        'Prior finalization has no quota checkpoint; fresh review forbidden': 'quota_checkpoint_missing',
        'Prior finalization quota is unknown; fresh review forbidden': 'quota_checkpoint_missing',
        'Recovery artifact is expired or ambiguous': 'recovery_expired',
        'Required recovery artifact is absent': 'recovery_missing',
        'Production audio is incomplete': 'incomplete_audio',
        'All planned ASR shards must finish before finalization': 'incomplete_shards',
        'Shard incomplete; summary forbidden': 'incomplete_shards',
        'Publication conflict retry budget exhausted; encrypted delta retained': 'publication_conflicts',
    }
    return reasons.get(str(error), 'missing_stage_input' if isinstance(error, FileNotFoundError) else 'stage_failure')


def artifact(name, target, *, run=None, required=False):
    """API failure or expiry is not equivalent to a confirmed absent checkpoint."""
    run = run or os.environ['GITHUB_RUN_ID']
    repo = os.environ['GITHUB_REPOSITORY']
    pages = json.loads(subprocess.check_output(['gh', 'api', '--paginate', '--slurp',
        f'repos/{repo}/actions/runs/{run}/artifacts?per_page=100'], stderr=subprocess.PIPE, timeout=120))
    found = [a for page in pages for a in page['artifacts'] if a['name'] == name]
    if not found:
        if required: raise ValueError('Required recovery artifact is absent')
        return False
    if len(found) != 1 or found[0]['expired']:
        raise ValueError('Recovery artifact is expired or ambiguous')
    shards.command(['gh', 'run', 'download', run, '--repo', repo, '--name', name, '--dir', str(target)])
    return True


def write_outputs(**values):
    shards.outputs(**values)


def last_finalization_attempt(run, slot, *, prior_only=False):
    """A lost runner checkpoint must not silently refund a later cloud attempt."""
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        return 0
    repo = os.environ['GITHUB_REPOSITORY']
    if run == os.environ['GITHUB_RUN_ID']:
        attempts = int(os.environ.get('GITHUB_RUN_ATTEMPT', '1')) - int(prior_only)
    else:
        info = json.loads(subprocess.check_output(['gh','api',f'repos/{repo}/actions/runs/{run}'],
                                                  stderr=subprocess.PIPE, timeout=60))
        attempts = info['run_attempt']
    if attempts > 20:
        raise ValueError('Finalization history exceeds bounded recovery audit')
    latest = 0
    for attempt in range(1, attempts+1):
        pages = json.loads(subprocess.check_output(['gh','api','--paginate','--slurp',
            f'repos/{repo}/actions/runs/{run}/attempts/{attempt}/jobs?per_page=100'],
            stderr=subprocess.PIPE, timeout=120))
        for page in pages:
            for job in page['jobs']:
                if not job['name'].endswith(f'finalize-{slot}'):
                    continue
                steps = job.get('steps')
                if steps is None:
                    # No visibility is not evidence that quota was unused.
                    latest = max(latest, attempt)
                elif any(step['name'] == 'Finalize through LectureRunner with saved quota'
                         and step.get('conclusion') != 'skipped' and step.get('started_at')
                         for step in steps):
                    latest = max(latest, attempt)
    return latest


def validate_checkpoint_age(saved, run, slot, *, prior_only=False):
    stamp = int(read_json(saved.get('attempt.json', b'0')))
    if last_finalization_attempt(run, slot, prior_only=prior_only) > stamp:
        raise ValueError('A later finalization lost its quota checkpoint; automatic refund forbidden')


def validation_course():
    course = os.environ.get('VALIDATION_COURSE_ID', '').strip()
    rank = validation_rank()
    if not course and rank != 1:
        raise ValueError('Recording rank is only allowed in isolated course validation')
    if course:
        if not course.isascii() or not course.isdigit():
            raise ValueError('Invalid validation course')
        if os.environ.get('PUBLISH_RESULTS') != 'false' or os.environ.get('SEND_EMAIL') != 'false':
            raise ValueError('Classroom validation requires publication and email disabled')
        if course not in configured_courses(os.environ['COURSE_IDS']):
            raise ValueError('Validation course is not subscribed')
    return course


def validation_rank():
    raw = os.environ.get('VALIDATION_LECTURE_RANK', '1').strip()
    if not raw.isascii() or not raw.isdigit() or not 1 <= int(raw) <= 10:
        raise ValueError('Invalid reverse recording rank')
    return int(raw)


def latest_validation_task(client, db, course, *, today=None, rank=1):
    """Probe actual playback, including entries with stale playback_status.

    No benchmark acquisition limit or cached summary is used. Deleted lectures
    remain excluded; an empty holiday schedule is a normal unavailable entry.
    """
    if type(rank) is not int or not 1 <= rank <= 10:
        raise ValueError('Invalid reverse recording rank')
    today = today or datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    detail = client.get_course_detail(course)
    candidates = []
    for lecture in detail.get('lectures', []):
        date = str(lecture.get('date', ''))
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date): continue
        try: datetime.strptime(date, '%Y-%m-%d')
        except ValueError: continue
        sub_id = str(lecture.get('sub_id', ''))
        if date > today or not sub_id.isascii() or not sub_id.isdigit(): continue
        if (db.get_lecture(sub_id) or {}).get('deleted_at'): continue
        period = re.search(r'第\s*(\d+)', str(lecture.get('sub_title', '')))
        candidates.append((date, int(period.group(1)) if period else -1, int(sub_id), lecture))
    seen, skipped, playable = set(), 0, 0
    for _, _, _, lecture in sorted(candidates, key=lambda item: item[:3], reverse=True):
        sub_id = str(lecture['sub_id'])
        if sub_id in seen: continue
        seen.add(sub_id)
        # Use the same API resolution/fallbacks as AudioDownloader. Do not
        # freeze an expiring private URL in the queue or print it in Actions.
        if client.get_video_url(course, sub_id):
            playable += 1
            if playable != rank:
                continue
            selected = dict(lecture, sub_id=sub_id,
                            _validation={'date': lecture['date'], 'skipped_unavailable': skipped,
                                         'playable_rank': rank})
            return (course, detail['title'], selected), detail.get('teacher', '')
        skipped += 1
    raise ValueError('Requested playable non-future lecture rank unavailable')


def plan():
    course = validation_course()
    # On a workflow rerun keep opaque slot identities and exact selections fixed.
    if artifact('qwen-production-queue', root()/'previous'):
        files = decode(root()/'previous'/'queue.enc', 'queue')
    else:
        if int(os.environ.get('GITHUB_RUN_ATTEMPT', '1')) > 1:
            raise ValueError('Rerun has lost its queue checkpoint; refusing a new selection')
        from main import login_with_retry, _enumerate_lectures, _crawl_semester_catalog
        from src.api.icourse import ICourseClient
        from src.runtime.reporter import Reporter
        db = Database(str(root()/'queue.db'))
        try:
            db.conn.close(); load_remote(root()/'queue.db'); db = Database(str(root()/'queue.db'))
            reporter = Reporter()
            client = ICourseClient(login_with_retry())
            if course:
                task, teacher = latest_validation_task(client, db, course, rank=validation_rank())
                history = snapshot(db, root()/'history.db')
                # A separate scratch database forces this authorized lecture
                # through ASR even when production already has a summary.
                # The original encrypted history remains in the queue bundle.
                db.conn.close(); db = Database(str(root()/'validation.db'))
                db.upsert_course(course, task[1], teacher)
                lecture = task[2]
                db.insert_lecture(lecture['sub_id'], course, lecture.get('sub_title', ''), lecture['date'])
                tasks = [task]
            else:
                _crawl_semester_catalog(client, db, reporter)
                tasks = _enumerate_lectures(client, db, reporter)
                tasks = [t for t in tasks if (db.get_lecture(str(t[2]['sub_id'])).get('error_count') or 0) < 3]
            if len(tasks) > MAX_TASKS:
                raise ValueError('Queue exceeds 256 tasks; narrow the subscribed course scope')
            files = {'queue.json': shards.encoded(tasks), 'database.db': snapshot(db, root()/'snapshot.db')}
            if course: files['history.db'] = history
        finally:
            db.conn.close()
    tasks = read_json(files['queue.json'])
    identities = [(str(t[0]), str(t[2]['sub_id'])) for t in tasks]
    if len(tasks) > MAX_TASKS or len(set(identities)) != len(identities):
        raise ValueError('Invalid or duplicate lecture queue')
    if course:
        if (len(tasks) != 1 or str(tasks[0][0]) != course or not tasks[0][2].get('_validation')
                or tasks[0][2]['_validation'].get('playable_rank', 1) != validation_rank()):
            raise ValueError('Validation queue does not match the requested course')
        out('validation-selection.json').write_bytes(shards.encoded(tasks[0][2]['_validation']))
    encode(files, 'queue', out('queue.enc'))
    write_outputs(tasks={'include': [{'task_slot': i} for i in range(len(tasks))]}, count=len(tasks))
    print(f'Planned {len(tasks)} lectures; at most 5 active pipelines and 15 runners', flush=True)


def task_files():
    files = decode(root()/'inbox'/'queue.enc', 'queue')
    tasks = read_json(files['queue.json'])
    slot = int(os.environ['COURSE_SLOT'])
    if not 0 <= slot < len(tasks): raise ValueError('Invalid queue task slot')
    course, title, lecture = tasks[slot]
    (root()/'course.db').write_bytes(files['database.db'])
    return Database(str(root()/'course.db')), str(course), title, lecture


def read_preparation():
    return decode(root()/'inbox'/'prepared.enc', 'prepared')


def shared_local_checkpoints(plan):
    """Read every prior worker before any replacement worker may claim blocks."""
    saved = {}
    for worker_id in range(len(plan['shards'])):
        destination = root()/'shared-recovery'/str(worker_id)
        if artifact(f'qwen-production-shared-{plan["course_slot"]}-{worker_id}',
                    destination, run=str(plan['run_id'])):
            with shards.environment({'GITHUB_RUN_ID': str(plan['run_id']),
                                     'COURSE_SLOT': str(plan['course_slot'])}):
                local = read_json(decode(destination/'shared-local.enc', f'shared-local-{worker_id}')['local.json'])
            if local['plan_hash'] != fingerprint(plan) or local['worker_id'] != worker_id:
                raise ValueError('Dynamic local checkpoint belongs to another input')
            saved[worker_id] = local['chunks']
    return saved


def shared_results(plan):
    """Read-only complete original blocks for gather and private export."""
    from scripts.shared_asr_worker import store_for
    from src.pipeline.asr_queue import SharedQueue
    store = store_for(plan)
    try:
        return SharedQueue(plan, store).results()
    finally:
        store.close()


def freeze_course_terms(db, course, title, sub_id):
    """See earlier published lessons even when the task queue was planned before them."""
    from src.ai.automatic_glossary import AutomaticGlossary
    if os.environ.get('PUBLISH_RESULTS') != 'true':
        return AutomaticGlossary(db, course).freeze(title, sub_id)
    history_path = root()/'glossary-history.db'
    revision = load_remote(history_path)
    if revision is None:
        return AutomaticGlossary(db, course).freeze(title, sub_id)
    history = Database(str(history_path))
    try:
        # The selected lesson may be new and absent from published history.
        # Its immutable queue date controls which older records are eligible.
        date = (db.get_lecture(sub_id) or {}).get('date')
        frozen = AutomaticGlossary(history, course).freeze(title, sub_id, lecture_date=date)
        frozen['history_revision'] = revision
        return frozen
    finally:
        history.conn.close()


def recover_preparation(db, course, sub_id):
    """Carry immutable audio and completed blocks into the next normal run."""
    raw = db.read_meta('qwen_pipeline:'+sub_id)
    metadata = json.loads(raw) if raw else {}
    recovery = metadata.get('recovery')
    if not recovery:
        return None
    old_run, old_slot = str(recovery['run_id']), int(recovery['task_slot'])
    target = root()/'recovery'
    artifact(f'qwen-production-prepare-{old_slot}', target, run=old_run, required=True)
    with shards.environment({'GITHUB_RUN_ID': old_run, 'COURSE_SLOT': str(old_slot)}):
        files = decode(target/'prepared.enc', 'prepared')
    spec = read_json(files['specification.json']); original = spec['plan']
    validate_plan(original)
    if (fingerprint(original) != recovery['plan_hash']
            or original['selection'] != {'course_id': course, 'sub_id': sub_id}):
        raise ValueError('Recovery belongs to another lecture or audio plan')
    plan = dict(original, run_id=os.environ['GITHUB_RUN_ID'], course_slot=int(os.environ['COURSE_SLOT']))
    spec.update(plan=plan, review=metadata.get('review', {}))
    saved_dir = target/'finalization'
    if artifact(f'qwen-production-state-{old_slot}', saved_dir, run=old_run):
        with shards.environment({'GITHUB_RUN_ID': old_run, 'COURSE_SLOT': str(old_slot)}):
            saved = decode(saved_dir/'state.enc', 'state')
        if saved['specification.json'] != files['specification.json']:
            raise ValueError('Recovered review state belongs to a different input')
        validate_checkpoint_age(saved, old_run, old_slot)
        spec['review'] = read_json(saved['review.json'])
    elif last_finalization_attempt(old_run, old_slot):
        raise ValueError('Prior finalization has no quota checkpoint; fresh review forbidden')
    for shard in original['shards']:
        shard_id = shard['shard_id']; destination = target/str(shard_id)
        result = None
        if artifact(f'qwen-production-asr-{old_slot}-{shard_id}', destination, run=old_run):
            with shards.environment({'GITHUB_RUN_ID': old_run, 'COURSE_SLOT': str(old_slot)}):
                result = read_json(decode(destination/'worker-result.enc', f'result-{shard_id}')['result.json'])
        elif f'completed-{shard_id}.json' in files:
            result = read_json(files[f'completed-{shard_id}.json'])
        if result is not None:
            validate_result(original, result, shard_id)
            result['plan_hash'] = fingerprint(plan)
            validate_result(plan, result, shard_id)
            files[f'completed-{shard_id}.json'] = shards.encoded(result)
    if original.get('execution') == 'shared_queue':
        from scripts.shared_asr_worker import store_for
        from src.pipeline.asr_queue import SharedQueue
        info = json.loads(subprocess.check_output(['gh', 'api',
            f'repos/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{old_run}'],
            stderr=subprocess.PIPE, timeout=60))
        if info['status'] != 'completed':
            raise ValueError('Cannot recover a shared queue while its source run is active')
        store = store_for(original)
        try:
            queue = SharedQueue(original, store)
            previous = shared_local_checkpoints(original)
            queue.restore_rows([row for rows in previous.values() for row in rows], int(info['run_attempt'])+1)
            results = queue.results(require_complete=False)
        finally:
            store.close()
        for result in results:
            result['plan_hash'] = fingerprint(plan)
            files[f'completed-{result["shard_id"]}.json'] = shards.encoded(result)
    files['database.db'] = lecture_snapshot(db, root()/'snapshot.db', course, sub_id)
    files['specification.json'] = shards.encoded(spec)
    return files


def prepare():
    slot = int(os.environ['COURSE_SLOT'])
    if artifact(f'qwen-production-prepare-{slot}', root()/'previous'):
        files = decode(root()/'previous'/'prepared.enc', 'prepared')
        specification = read_json(files['specification.json'])
        if specification['mode'] != 'failed':
            if specification.get('plan', {}).get('execution') == 'shared_queue':
                from scripts.shared_asr_worker import initialize
                initialize(specification['plan'], files)
            encode(files, 'prepared', out('prepared.enc'))
            write_outputs(workers={'shard_id': list(range(len(specification.get('plan', {}).get('shards', [])))) or [-1]})
            return
    db, course, title, lecture = task_files()
    sub_id = str(lecture['sub_id'])
    scheduler = None
    specification = {'course_id': course, 'course_title': title, 'lecture': lecture}
    files = {}
    try:
        from src.pipeline.prepared_lecture import cached_material
        from src.ai.qwen_transcriber import QwenTranscriber
        from src.ai.course_glossary import course_terms
        from main import login_with_retry
        from src.api.icourse import ICourseClient
        from src.runtime.scheduler import Scheduler
        from src.runtime.reporter import Reporter
        from src.pipeline.ppt_pipeline import PPTPipeline
        existing = db.get_lecture(sub_id)
        if existing.get('deleted_at') or existing.get('summary'):
            specification['mode'] = 'finished'
        else:
            if not existing.get('transcript'):
                recovered = recover_preparation(db, course, sub_id)
                if recovered:
                    recovered_plan = read_json(recovered['specification.json'])['plan']
                    if recovered_plan.get('execution') == 'shared_queue':
                        from scripts.shared_asr_worker import initialize
                        initialize(recovered_plan, recovered)
                    encode(recovered, 'prepared', out('prepared.enc'))
                    plan = read_json(recovered['specification.json'])['plan']
                    write_outputs(workers={'shard_id': list(range(len(plan['shards']))) or [-1]})
                    return
            # Audio retrieval uses the production downloader and its unchanged
            # authenticated playback fallback chain; no benchmark acquisition cap.
            reporter = Reporter(); client = ICourseClient(login_with_retry())
            scheduler = Scheduler(reporter)
            ppt = PPTPipeline(db, scheduler, reporter).submit(client, course, sub_id, defer_ocr=True)
            from src.runtime import config
            if config.USE_OFFICIAL_TRANSCRIPT:
                try:
                    support = client.get_transcript_segments(sub_id) or []
                    # Auxiliary evidence is bounded and never replaces Qwen text.
                    specification['official_support'] = support if len(json.dumps(support, ensure_ascii=False)) <= 20000 else []
                except Exception:
                    specification['official_support'] = []
            cached = cached_material(db, existing)
            if existing.get('transcript'):
                specification['mode'] = 'cached'
                if cached:
                    material, review = cached
                    # Audio has already been released after the saved transcript.
                    # Keep used quota, completed variants and unresolved calls.
                    review.setdefault('material', {})
                    if not review.get('complete'):
                        review['material']['remaining_review_unavailable_without_audio'] = True
                        review['complete'] = True
                    specification.update(material=material, review=review)
            else:
                terms = course_terms(title)
                if os.environ.get('AUTO_COURSE_TERMS') == 'true':
                    glossary_snapshot = freeze_course_terms(db, course, title, sub_id)
                    terms = glossary_snapshot['terms']
                    specification['glossary_snapshot'] = glossary_snapshot
                scheduler.audio_downloader.schedule(client, course, sub_id)
                handle = scheduler.audio_downloader.get(sub_id, timeout=180)
                if handle is None: raise ValueError('No playable production audio')
                began = time.monotonic()
                while not Path(handle.path).exists():
                    if handle.process.poll() is not None or time.monotonic()-began > 60:
                        raise ValueError('Decoded audio did not become available')
                    time.sleep(.1)
                transcriber = QwenTranscriber()
                with open(handle.path, 'rb') as audio:
                    windows = transcriber.prepare_pcm_stream(audio.read, lambda: handle.process.poll() is not None,
                        lambda: b''.join(handle.stderr_chunks), lambda: handle.process.returncode,
                        audio_path=handle.path, timeout=2400)
                duration = transcriber.last_audio_duration
                media = transcriber.last_media_duration or 0
                if media and duration < media-max(120, media*.05):
                    raise ValueError('Production audio is incomplete')
                flac = root()/'lecture.flac'
                shards.command(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'f32le', '-ar', '16000', '-ac', '1',
                                '-i', handle.path, '-c:a', 'flac', '-y', str(flac)], timeout=300)
                flac_bytes = flac.read_bytes()
                digest = hashlib.sha256(flac_bytes).hexdigest()
                plan = build_audio_plan({'selection': {'course_id': course, 'sub_id': sub_id},
                    'audio_seconds': duration, 'full_chunks': [{'start': a, 'end': b} for a,b in windows],
                    'vad_windows': transcriber.last_vad_windows, 'recognition_terms': terms},
                    reference={'pipeline': 'production'}, course_slot=slot, run_id=os.environ['GITHUB_RUN_ID'],
                    audio_sha256=digest, mode=os.environ.get('SHARD_MODE', '2'), production=True)
                files['lecture.flac'] = flac_bytes
                for block in plan['blocks']:
                    chunk = root()/f"chunk-{block['chunk_id']}.flac"
                    # atrim operates on sample indices so FLAC preserves exact
                    # original PCM boundaries, including overlapping VAD blocks.
                    shards.command(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(flac), '-af',
                        f"atrim=start_sample={round(block['start']*16000)}:end_sample={round(block['end']*16000)}",
                        '-c:a', 'flac', '-y', str(chunk)])
                    files[chunk.name] = chunk.read_bytes()
                    block['flac_sha256'] = hashlib.sha256(files[chunk.name]).hexdigest()
                validate_plan(plan)
                specification.update(mode='sharded', plan=plan, media_seconds=media)
            ppt.drain()
        files.update({'specification.json': shards.encoded(specification),
                      'database.db': lecture_snapshot(db, root()/'snapshot.db', course, sub_id)})
        if specification['mode'] == 'sharded' and os.environ.get('PUBLISH_RESULTS') == 'true':
            db.write_meta('qwen_pipeline:'+sub_id, json.dumps({'review': {}, 'recovery': {
                'run_id': os.environ['GITHUB_RUN_ID'], 'task_slot': slot,
                'plan_hash': fingerprint(specification['plan'])},
                'updated_at': datetime.now(timezone.utc).isoformat()}, ensure_ascii=False))
            delta = root()/'initial-state.db'
            files['database.db'] = lecture_snapshot(db, delta, course, sub_id)
            # Persist the recovery identity before any worker or cloud review
            # starts. A lost job cannot look like a brand-new lecture next run.
            publish(delta, course, sub_id)
        if specification.get('plan', {}).get('execution') == 'shared_queue':
            from scripts.shared_asr_worker import initialize
            initialize(specification['plan'], files)
        encode(files, 'prepared', out('prepared.enc'))
        write_outputs(workers={'shard_id': list(range(len(specification.get('plan', {}).get('shards', [])))) or [-1]})
    except Exception as error:
        db.update_error(sub_id, 'prepare', type(error).__name__)
        # A coordination/API failure after slicing must retain the acquired
        # input. Retry initializes the queue from this same immutable plan.
        if specification.get('plan') and 'lecture.flac' in files:
            specification['prepare_error_type'] = type(error).__name__
        else:
            specification.update(mode='failed', error_type=type(error).__name__)
            files = {}
        files.update({'specification.json': shards.encoded(specification),
                      'database.db': lecture_snapshot(db, root()/'snapshot.db', course, sub_id)})
        encode(files, 'prepared', out('prepared.enc'))
        raise
    finally:
        if scheduler: scheduler.shutdown()
        db.conn.close()


def worker():
    files = read_preparation(); spec = read_json(files['specification.json'])
    if int(os.environ['SHARD_ID']) == -1:
        if spec['mode'] == 'failed': raise ValueError('Preparation failed')
        return
    plan = spec['plan']; validate_plan(plan)
    if plan.get('execution') == 'shared_queue':
        from scripts.shared_asr_worker import run_worker, store_for
        from src.pipeline.asr_queue import SharedQueue
        worker_id = int(os.environ['SHARD_ID'])
        attempt = int(os.environ.get('GITHUB_RUN_ATTEMPT', '1'))
        previous = shared_local_checkpoints(plan) if attempt > 1 else {}
        store = store_for(plan)
        try:
            SharedQueue(plan, store).restore_rows([row for rows in previous.values() for row in rows], attempt)
            report = run_worker(plan, files, store, worker_id, attempt,
                                previous_rows=previous.get(worker_id, []))
        finally:
            store.close()
        out('worker-audit.json').write_bytes(shards.encoded(report))
        return
    shard_id = int(os.environ['SHARD_ID'])
    assigned = plan['shards'][shard_id]['chunk_ids']
    audio = {'manifest.json': shards.encoded(plan)}
    audio.update({f'chunk-{i}.flac': files[f'chunk-{i}.flac'] for i in assigned})
    if f'completed-{shard_id}.json' in files:
        audio['completed.json'] = files[f'completed-{shard_id}.json']
    encode(audio, f'input-{shard_id}', root()/'inbox'/f'shard-{shard_id}.enc')
    if artifact(f'qwen-production-asr-{os.environ["COURSE_SLOT"]}-{shard_id}', root()/'previous'):
        shutil.copyfile(root()/'previous'/'worker-result.enc', out('worker-result.enc'))
    shards.worker()


class CheckpointDatabase(Database):
    """Re-encrypt committed SQLite state after each pipeline state transition."""
    checkpoint = None

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if name in ('update_transcript', 'clear_transcript', 'update_summary', 'mark_processed',
                    'clear_error', 'update_error'):
            def mutation(*args, **kwargs):
                result = value(*args, **kwargs)
                if self.checkpoint: self.checkpoint()
                return result
            return mutation
        return value


def gather():
    slot = int(os.environ['COURSE_SLOT'])
    files = read_preparation(); spec = read_json(files['specification.json'])
    prior = root()/'previous'
    restored_current = artifact(f'qwen-production-state-{slot}', prior)
    if restored_current:
        saved = decode(prior/'state.enc', 'state')
        if saved['specification.json'] != files['specification.json']:
            raise ValueError('Finalization checkpoint belongs to a different prepared lecture')
        validate_checkpoint_age(saved, os.environ['GITHUB_RUN_ID'], slot, prior_only=True)
        files['database.db'] = saved['database.db']
        review = read_json(saved['review.json'])
    else:
        if last_finalization_attempt(os.environ['GITHUB_RUN_ID'], slot, prior_only=True):
            raise ValueError('Prior finalization quota is unknown; fresh review forbidden')
        review = spec.get('review', {})
    from src.ai.qwen_review_ledger import validate_ledger
    validate_ledger(review)
    (root()/'course.db').write_bytes(files['database.db'])
    db = CheckpointDatabase(str(root()/'course.db'))
    course, lecture = spec['course_id'], spec['lecture']; sub_id = str(lecture['sub_id'])
    material = spec.get('material')
    initial_errors = db.get_lecture(sub_id).get('error_count') or 0

    def checkpoint():
        row = db.get_lecture(sub_id)
        if spec['mode'] == 'sharded' or (material is not None and db.get_lecture(sub_id).get('transcript')):
            metadata = {'review': review, 'updated_at': datetime.now(timezone.utc).isoformat()}
            if spec['mode'] == 'sharded':
                metadata['recovery'] = {'run_id': os.environ['GITHUB_RUN_ID'], 'task_slot': slot,
                                        'plan_hash': fingerprint(spec['plan'])}
            if material is not None and db.get_lecture(sub_id).get('transcript'):
                durable = {k:v for k,v in material.items() if k != 'audio_path'}
                transcript = db.get_lecture(sub_id)['transcript']
                metadata.update(material=durable, transcript_sha256=hashlib.sha256(transcript.encode()).hexdigest())
            if row.get('processed_at'):
                # Completed histories need only audit identity and quota totals,
                # not another copy of the full classroom in the metadata shard.
                metadata.pop('material', None); metadata.pop('recovery', None)
                metadata['complete'] = True
                if material:
                    metadata.update(audio_sha256=material['audio_sha256'], plan_hash=material['plan_hash'],
                                    audio_seconds=material['audio_seconds'])
            db.write_meta('qwen_pipeline:'+sub_id, json.dumps(metadata, ensure_ascii=False))
        payload = {'specification.json': files['specification.json'], 'review.json': shards.encoded(review),
                   'attempt.json': shards.encoded(int(os.environ.get('GITHUB_RUN_ATTEMPT', '1'))),
                   'database.db': lecture_snapshot(db, root()/'snapshot.db', course, sub_id)}
        encode(payload, 'state', out('state.enc'))
        if lecture.get('_validation'):
            plan = spec.get('plan', {})
            from src.ai.automatic_glossary import AutomaticGlossary
            stages = AutomaticGlossary(db, course).stages()
            audit = dict(lecture['_validation'], mode=spec['mode'],
                asr_execution=plan.get('execution', 'fixed_shards'),
                automatic_terms=os.environ.get('AUTO_COURSE_TERMS') == 'true',
                frozen_terms_count=len(plan.get('recognition_terms', [])),
                frozen_terms_sha256=fingerprint(plan.get('recognition_terms', [])),
                glossary_saved=bool(db.read_meta('auto_glossary:'+str(course)+':'+sub_id)),
                glossary_candidates=sum(g['stage'] == 'candidate' for g in stages),
                glossary_confirmed=sum(g['stage'] == 'confirmed' for g in stages),
                planned_shards=len(plan.get('shards', [])), planned_blocks=len(plan.get('blocks', [])),
                audio_seconds=plan.get('audio_seconds'), media_seconds=spec.get('media_seconds'),
                transcript_chars=len(row.get('transcript') or ''), summary_chars=len(row.get('summary') or ''),
                processed=bool(row.get('processed_at')), emailed=bool(row.get('emailed_at')),
                error_stage=row.get('error_stage'), error_count=row.get('error_count') or 0,
                review_seconds=review.get('seconds', 0), review_clips=len(review.get('attempts', [])),
                review_complete=bool(review.get('complete')), review_failed=bool(review.get('failed')),
                review_error_type=review.get('error_type'),
                homework_candidates=len(review.get('homework', {}).get('candidates', [])),
                homework_deferred=review.get('homework', {}).get('deferred_count', 0),
                homework_clips=sum(a['interval'].get('kind') == 'homework' for a in review.get('attempts', [])),
                homework_visual_status=review.get('homework', {}).get('visual', {}).get('status'),
                homework_visual_capture_status=review.get('homework', {}).get('visual', {}).get('capture_status'),
                homework_visual_reference_status=review.get('homework', {}).get('visual', {}).get('reference_status'),
                homework_vision_calls=len(review.get('homework', {}).get('vision_calls', [])),
                homework_vision_call_statuses=[c['status'] for c in review.get('homework', {}).get('vision_calls', [])],
                homework_vision_frame_count=sum(len(c.get('images', [])) for c in review.get('homework', {}).get('vision_calls', [])),
                homework_vision_image_count=sum(c.get('image_count', 0) for c in review.get('homework', {}).get('vision_calls', [])),
                asr_complete=bool(material and material.get('complete')))
            out('validation-result.json').write_bytes(shards.encoded(audit))
    db.checkpoint = checkpoint
    try:
        checkpoint()
        if spec['mode'] == 'failed': raise ValueError('Preparation failed')
        if spec['mode'] == 'sharded':
            from src.pipeline.prepared_lecture import assemble_material
            plan = spec['plan']
            if plan.get('execution') == 'shared_queue':
                results = shared_results(plan)  # All original blocks must be complete.
            else:
                results = []
                for shard in plan['shards']:
                    path = root()/'results'/f'qwen-production-asr-{slot}-{shard["shard_id"]}'/'worker-result.enc'
                    result = read_json(decode(path, f'result-{shard["shard_id"]}')['result.json'])
                    validate_result(plan, result, shard['shard_id'], require_complete=True)
                    results.append(result)
            if hashlib.sha256(files['lecture.flac']).hexdigest() != plan['audio_sha256']:
                raise ValueError('Prepared audio hash changed')
            flac = root()/'lecture.flac'; flac.write_bytes(files['lecture.flac'])
            raw = root()/'audio.raw'
            shards.command(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(flac), '-f', 'f32le',
                            '-ac', '1', '-ar', '16000', '-y', str(raw)], timeout=300)
            if abs(raw.stat().st_size/64000-plan['audio_seconds']) > .1:
                raise ValueError('Decoded finalization audio differs from the immutable plan')
            material = assemble_material(plan, results, audio_path=str(raw), media_seconds=spec['media_seconds'])
            material['official_support'] = spec.get('official_support', [])
        row = db.get_lecture(sub_id)
        if spec['mode'] != 'finished' or (row.get('summary') and not row.get('deleted_at')):
            from src.pipeline.lecture_runner import LectureRunner
            from src.ai.qwen_transcriber import QwenTranscriber
            from src.ai.summarizer import Summarizer
            from src.runtime.reporter import Reporter
            from types import SimpleNamespace
            scheduler = SimpleNamespace(audio_downloader=SimpleNamespace(release=lambda *_: None))
            summarizer = None if row.get('summary') else Summarizer()
            runner = LectureRunner(None, db, scheduler, QwenTranscriber(), summarizer, Reporter())
            runner.run(course, spec['course_title'], lecture, prepared_asr=material,
                       prepared_ppt=True, review_state=review if material else None, checkpoint=checkpoint)
            row = db.get_lecture(sub_id)
            if not row.get('processed_at'):
                raise RuntimeError('LectureRunner retained a retryable processing failure')
        checkpoint()
    except Exception as error:
        row = db.get_lecture(sub_id)
        # Phase-specific errors already committed by LectureRunner/prepare are
        # counted once. A missing shard or hard failure gets an explicit stage.
        already_counted = (restored_current and row.get('error_stage')) or spec['mode'] == 'failed'
        if not already_counted and (row.get('error_count') or 0) == initial_errors:
            db.update_error(sub_id, 'sharded_finalize', failure_code(error))
        checkpoint()
        raise
    finally:
        db.conn.close()


def publish_result():
    files = decode(root()/'inbox'/'state.enc', 'state')
    spec = read_json(files['specification.json'])
    course, sub_id = spec['course_id'], str(spec['lecture']['sub_id'])
    configured = configured_courses(os.environ['COURSE_IDS'])
    if course not in configured: raise ValueError('Publication course is no longer subscribed')
    delta = root()/'delta.db'; delta.write_bytes(files['database.db'])
    if os.environ.get('PUBLISH_RESULTS') == 'true':
        publish(delta, course, sub_id)
        print('Encrypted lecture state published', flush=True)
    else:
        target = root()/'isolated.db'
        load_remote(target)
        merge_lecture(delta, target, course, sub_id)
        db = Database(str(target))
        try: payload = lecture_snapshot(db, root()/'snapshot.db', course, sub_id)
        finally: db.conn.close()
        encode({'database.db': payload, 'specification.json': files['specification.json'],
                'review.json': files['review.json']}, 'published', out('published.enc'))
        print('Isolated publication merge validated; remote database unchanged', flush=True)


def deliver():
    """One batch mail outbox after durable publication, with saved send receipts."""
    if os.environ.get('PUBLISH_RESULTS') != 'true' or os.environ.get('SEND_EMAIL') != 'true':
        raise ValueError('Email requires explicit publish and send switches')
    from scripts.parallel_courses import deliver as send
    target = Path('data/icourse.db'); target.parent.mkdir(exist_ok=True)
    load_remote(target)
    queue = decode(root()/'inbox'/'queue.enc', 'queue')
    tasks = read_json(queue['queue.json'])
    configured = configured_courses(os.environ['COURSE_IDS'])
    baseline = Database(str(target))
    receipts_before = {r['sub_id']: (r['emailed_at'], r['failure_notified_at'])
                       for r in baseline.conn.execute('SELECT * FROM lectures').fetchall()}
    baseline.conn.close()
    if artifact('qwen-production-mail-receipts', root()/'previous'):
        saved = decode(root()/'previous'/'receipts.enc', 'mail')
        previous = root()/'receipts.db'; previous.write_bytes(saved['database.db'])
        prior = Database(str(previous))
        current = Database(str(target))
        try:
            for row in prior.conn.execute('SELECT * FROM lectures').fetchall():
                if row['course_id'] in configured and (row['emailed_at'] or row['failure_notified_at']):
                    current.conn.execute('''UPDATE lectures SET emailed_at=COALESCE(emailed_at,?),
                        failure_notified_at=COALESCE(failure_notified_at,?) WHERE sub_id=?''',
                        (row['emailed_at'], row['failure_notified_at'], row['sub_id']))
            current.conn.commit()
        finally:
            prior.conn.close(); current.conn.close()
    def mail_checkpoint(db):
        encode({'database.db': snapshot(db, root()/'mail-snapshot.db')}, 'mail', out('receipts.enc'))
    original = {name: getattr(Database, name) for name in ('mark_emailed_batch', 'mark_failure_notified_batch')}
    def wrapped(name):
        def save(db, *args, **kwargs):
            result = original[name](db, *args, **kwargs)
            mail_checkpoint(db)
            return result
        return save
    for name in original: setattr(Database, name, wrapped(name))
    try:
        send()
    finally:
        for name, method in original.items(): setattr(Database, name, method)
        db = Database(str(target))
        try:
            mail_checkpoint(db)
            for row in db.conn.execute('SELECT * FROM lectures').fetchall():
                course, sub_id = row['course_id'], row['sub_id']
                if course in configured and (row['emailed_at'], row['failure_notified_at']) != receipts_before.get(sub_id):
                    delta = root()/'mail-delta.db'
                    lecture_snapshot(db, delta, course, sub_id)
                    publish(delta, course, sub_id)
        finally:
            db.conn.close()


def finalize():
    """Retain the ordinary catalog/subscription sync, including empty lecture queues."""
    queue = decode(root()/'inbox'/'queue.enc', 'queue')
    delta = root()/'catalog.db'; delta.write_bytes(queue['database.db'])
    db = Database(str(delta))
    try:
        with db.conn:
            for table in ('courses', 'lectures', 'ppt_pages'):
                db.conn.execute('DELETE FROM '+table)
            db.conn.execute('DELETE FROM meta')
        snapshot(db, root()/'catalog-snapshot.db')
    finally:
        db.conn.close()
    if os.environ.get('PUBLISH_RESULTS') == 'true':
        publish(root()/'catalog-snapshot.db')
        if os.environ.get('SEND_EMAIL') == 'true':
            deliver()
    else:
        encode({'database.db': (root()/'catalog-snapshot.db').read_bytes()}, 'catalog', out('catalog.enc'))


def main():
    os.environ['QWEN_PRODUCTION_TASK'] = 'true'
    mode = sys.argv[1]
    if mode == 'clean':
        shutil.rmtree(root(), ignore_errors=True); return
    # Existing components log private course names. Keep their output inside
    # the process; only the enclosing workflow gets sanitized completion info.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        {'plan': plan, 'prepare': prepare, 'worker': worker, 'gather': gather,
         'publish': publish_result, 'deliver': deliver, 'finalize': finalize}[mode]()
    print(f'Production pilot {mode} completed', flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print(f'Production pilot failed ({type(error).__name__}, {failure_code(error)}); private details withheld', flush=True)
        raise SystemExit(1)
