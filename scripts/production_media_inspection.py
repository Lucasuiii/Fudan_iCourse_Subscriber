"""Read source container/stream headers for one saved lecture, without decoding.

No media file, packets, frames, ASR, model credentials or publisher are used.
The current source's metadata is diagnostic evidence, not proof that a previous
download of that lecture had the same bytes or was complete.
"""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import math
import os
import re
import subprocess
import sys
import time


ENTRIES = ('format=start_time,duration,size:'
           'stream=index,codec_type,codec_name,start_time,duration,sample_rate,'
           'channels,time_base,duration_ts,nb_frames')


def safe_metadata(raw):
    """Whitelist numeric fields; never pass tags, private URLs or error text."""
    result = {'format': {}, 'streams': []}
    def number(value):
        try: n = float(value)
        except (TypeError, ValueError): return None
        return n if math.isfinite(n) else None
    for key in ('start_time', 'duration', 'size'):
        n = number(raw.get('format', {}).get(key))
        if n is not None: result['format'][key] = n
    for stream in raw.get('streams', [])[:32]:
        if not isinstance(stream, dict): continue
        row = {}
        for key in ('index', 'start_time', 'duration', 'sample_rate', 'channels', 'duration_ts', 'nb_frames'):
            n = number(stream.get(key))
            if n is not None: row[key] = n
        for key in ('codec_type', 'codec_name'):
            v = stream.get(key)
            if isinstance(v, str) and re.fullmatch(r'[a-zA-Z0-9_]{1,40}', v): row[key] = v
        value = stream.get('time_base')
        if isinstance(value, str) and re.fullmatch(r'\d{1,12}/[1-9]\d{0,11}', value): row['time_base'] = value
        if 'start_time' in row and 'duration' in row:
            row['end_time'] = row['start_time']+row['duration']
        result['streams'].append(row)
    return result


def probe_headers(url, headers):
    # -nofind_stream_info avoids the normal packet decoding used to infer
    # missing properties. Only existing demuxer/header metadata is requested.
    began = time.monotonic()
    process = subprocess.run(['ffprobe', '-v', 'error', '-nofind_stream_info',
        '-rw_timeout', '30000000', '-headers', headers, '-show_entries', ENTRIES,
        '-of', 'json', url], capture_output=True, timeout=90)
    result = {'probe_return_code': process.returncode,
              'probe_seconds': time.monotonic()-began, 'header_only': True,
              'stderr_present': bool(process.stderr.strip())}
    if process.returncode:
        result['status'] = 'failed'
        return result
    if len(process.stdout) > 256*1024: raise ValueError('Oversized source metadata')
    result.update(safe_metadata(json.loads(process.stdout)), status='complete')
    return result


def inspect():
    import base64
    from scripts import production_qwen as pipeline, sharded_qwen_pilot as shards
    from scripts.production_result_export import encrypt, identity
    from src.api.webvpn import WebVPNSession
    from src.api.icourse import ICourseClient
    run, slot = os.environ['SOURCE_RUN_ID'], int(os.environ['SOURCE_SLOT'])
    identity(run, slot)
    recipient = base64.b64decode(os.environ['RECIPIENT_PUBLIC_KEY'], validate=True)
    if len(recipient) != 32: raise ValueError('Invalid recipient public key')
    info = json.loads(subprocess.check_output(['gh', 'api',
        f'repos/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{run}'],
        stderr=subprocess.PIPE, timeout=60))
    if info['status'] != 'completed' or info['path'].split('@')[0] != '.github/workflows/parallel_pilot.yml':
        raise ValueError('Source must be a completed formal pilot')
    target = pipeline.root()/'media-inspection-source'
    pipeline.artifact(f'qwen-production-prepare-{slot}', target, run=run, required=True)
    with shards.environment({'GITHUB_RUN_ID':run, 'COURSE_SLOT':str(slot)}):
        files = pipeline.decode(target/'prepared.enc', 'prepared')
    spec = json.loads(files['specification.json']); del files
    lecture = spec['lecture']; course = str(spec['course_id'])
    vpn = WebVPNSession()
    try:
        vpn.login(); vpn.authenticate_icourse()
        client = ICourseClient(vpn)
        # Resolve exactly the stored lecture using the existing fallback chain.
        # No scanning, alternate-track switch, audio extraction or model call.
        url = client.get_video_url(course, str(lecture['sub_id']))
        if not url: raise ValueError('Saved lesson has no current playable source')
        vpn_url, headers = client.get_stream_params(url)
        metadata = probe_headers(vpn_url, headers)
    finally:
        vpn.session.close()
    payload = {'source_run_id':run, 'source_slot':slot, 'source_commit':info['head_sha'],
        'course_id':course, 'sub_id':str(lecture['sub_id']), 'date':lecture.get('date'),
        'sub_title':lecture.get('sub_title'), 'retained_audio':spec.get('audio_diagnostics', {}),
        'current_source_metadata':metadata, 'same_source_bytes_verified':False}
    pipeline.out('media-inspection.enc').write_bytes(encrypt(shards.encoded(payload), recipient, run, slot))
    if metadata['status'] != 'complete': raise ValueError('Source metadata probe failed')


if __name__ == '__main__':
    os.environ['QWEN_PRODUCTION_TASK'] = 'true'
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()): inspect()
        print('Source stream headers inspected encrypted; no audio decoding or ASR')
    except Exception as error:
        print(f'Source metadata inspection failed ({type(error).__name__}); private details withheld')
        sys.exit(1)
