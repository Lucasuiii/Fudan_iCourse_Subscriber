"""Read source headers and bounded late packet timestamps without decoding.

No media file, packet payload export, frames, ASR or publisher are used.
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


def safe_probe_errors(stderr):
    """Diagnostic reason codes only, never echo a signed URL or header."""
    from src.runtime.scheduler import record_decode_errors
    counts = {}
    record_decode_errors(stderr, counts)
    lowered = stderr.lower()
    for code, markers in {
        'seek_failed': (b'could not seek', b'failed to seek', b'error seeking'),
        'range_unsupported': (b'cannot seek', b'not seekable'),
        'invalid_argument': (b'invalid argument',),
        'invalid_media': (b'invalid data found', b'moov atom not found'),
    }.items():
        if any(marker in lowered for marker in markers): counts[code] = 1
    statuses = sorted({int(code) for code in re.findall(rb'http error ([45]\d\d)', lowered)})
    return {'error_counts':counts, 'http_error_statuses':statuses}


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
    result.update(safe_probe_errors(process.stderr))
    if process.returncode:
        result['status'] = 'failed'
        return result
    if len(process.stdout) > 256*1024: raise ValueError('Oversized source metadata')
    result.update(safe_metadata(json.loads(process.stdout)), status='complete')
    return result


def probe_late_packets(url, headers, retained_seconds, stream):
    """At most 128 packet headers, no packet payload or frame decoding.

    This checks whether current source audio beyond the saved input is readable;
    it does not assert that the previously acquired source was identical.
    """
    endpoint = stream.get('end_time')
    if not isinstance(endpoint, (float, int)) or endpoint <= retained_seconds+120:
        return {'status':'not_needed', 'packets':[]}
    starts = [max(0, retained_seconds+10), max(0, endpoint-10)]
    intervals = ','.join(f'{start:.3f}%+#64' for start in starts)
    began = time.monotonic()
    process = subprocess.run(['ffprobe', '-v', 'error', '-nofind_stream_info',
        '-rw_timeout', '30000000', '-headers', headers, '-select_streams', 'a:0',
        '-read_intervals', intervals, '-show_packets', '-show_entries',
        'packet=stream_index,pts_time,dts_time,duration_time,size,pos', '-of', 'json', url],
        capture_output=True, timeout=90)
    result = {'probe_return_code':process.returncode, 'probe_seconds':time.monotonic()-began,
              'requested_starts':starts, 'maximum_packets':128, 'decoding':False,
              'payload_exported':False, 'stderr_present':bool(process.stderr.strip()), 'packets':[]}
    result.update(safe_probe_errors(process.stderr))
    if process.returncode:
        result['status']='failed'
        return result
    if len(process.stdout) > 256*1024: raise ValueError('Oversized packet metadata')
    packets = json.loads(process.stdout).get('packets', [])
    if len(packets) > 128: raise ValueError('Packet diagnostic bound exceeded')
    for packet in packets:
        row={}
        for key in ('stream_index','pts_time','dts_time','duration_time','size','pos'):
            try: value=float(packet.get(key))
            except (TypeError,ValueError): continue
            if math.isfinite(value):row[key]=value
        result['packets'].append(row)
    result['status']='complete'
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
    payload = {'source_run_id':run, 'source_slot':slot, 'same_source_bytes_verified':False}
    stage = 'source_identity'
    vpn = None
    try:
        info = json.loads(subprocess.check_output(['gh', 'api',
            f'repos/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{run}'],
            stderr=subprocess.PIPE, timeout=60))
        if info['status'] != 'completed' or info['path'].split('@')[0] != '.github/workflows/parallel_pilot.yml':
            raise ValueError('Source must be a completed formal pilot')
        payload['source_commit'] = info['head_sha']
        stage = 'retained_artifact'
        target = pipeline.root()/'media-inspection-source'
        pipeline.artifact(f'qwen-production-prepare-{slot}', target, run=run, required=True)
        stage = 'retained_input'
        with shards.environment({'GITHUB_RUN_ID':run, 'COURSE_SLOT':str(slot)}):
            files = pipeline.decode(target/'prepared.enc', 'prepared')
        spec = json.loads(files['specification.json']); del files
        lecture = spec['lecture']; course = str(spec['course_id'])
        payload.update(course_id=course, sub_id=str(lecture['sub_id']), date=lecture.get('date'),
                       sub_title=lecture.get('sub_title'), retained_audio=spec.get('audio_diagnostics', {}))
        import requests
        payload['authentication_attempts'] = []
        for attempt in range(2):
            vpn = WebVPNSession()
            stage = 'webvpn_login'
            try:
                vpn.login()
                stage = 'icourse_authentication'
                vpn.authenticate_icourse(strict=True)
            except Exception as error:
                from src.api.webvpn import AuthenticationError
                row={'attempt':attempt+1,'stage':stage,'failure_type':type(error).__name__,
                     'diagnostics':vpn.auth_diagnostics if type(vpn.auth_diagnostics) is list else []}
                if isinstance(error,AuthenticationError):row['reason']=error.reason
                payload['authentication_attempts'].append(row)
                transient=(isinstance(error,(requests.exceptions.Timeout,requests.exceptions.ConnectionError))
                           or isinstance(error,AuthenticationError) and error.reason=='cold_session')
                if attempt == 1 or not transient:raise
                vpn.session.close();time.sleep(2)
                continue
            payload['authentication_attempts'].append({'attempt':attempt+1,'verified':True,
                'diagnostics':vpn.auth_diagnostics if type(vpn.auth_diagnostics) is list else []})
            break
        stage = 'playback_selection'
        client = ICourseClient(vpn)
        # Resolve exactly the stored lecture using the existing fallback chain.
        # No scanning, alternate-track switch, audio extraction or model call.
        url = client.get_video_url(course, str(lecture['sub_id']))
        if not url: raise ValueError('Saved lesson has no current playable source')
        vpn_url, headers = client.get_stream_params(url)
        stage = 'source_headers'
        metadata = probe_headers(vpn_url, headers)
        payload['current_source_metadata'] = metadata
        if metadata['status'] != 'complete': raise ValueError('Source metadata probe failed')
        audio = [s for s in metadata.get('streams', []) if s.get('codec_type') == 'audio']
        retained = spec.get('audio_diagnostics', {}).get('audio_seconds')
        stage = 'late_packets'
        late = (probe_late_packets(vpn_url, headers, retained, audio[0])
                if len(audio) == 1 and isinstance(retained, (int,float))
                else {'status':'ambiguous_or_unavailable','packets':[]})
        payload['current_source_late_packets'] = late
        if late['status'] == 'failed': raise ValueError('Source late packet probe failed')
        payload['inspection_status'] = 'complete'
    except Exception as error:
        # Do not serialize provider messages: they can contain signed URLs,
        # login response bodies or cookies. Fixed stage names locate failures.
        payload.update(inspection_status='failed', failure_stage=stage,
                       failure_type=type(error).__name__)
        from src.api.webvpn import AuthenticationError
        if isinstance(error,AuthenticationError): payload['authentication_reason'] = error.reason
        raise
    finally:
        if vpn is not None and type(getattr(vpn,'auth_diagnostics',None)) is list:
            payload['authentication_diagnostics'] = vpn.auth_diagnostics
        try:
            if vpn is not None: vpn.session.close()
        finally:
            pipeline.out('media-inspection.enc').write_bytes(encrypt(shards.encoded(payload), recipient, run, slot))


if __name__ == '__main__':
    os.environ['QWEN_PRODUCTION_TASK'] = 'true'
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()): inspect()
        print('Source metadata inspected encrypted; no audio decoding or ASR')
    except Exception as error:
        print(f'Source metadata inspection failed ({type(error).__name__}); private details withheld')
        sys.exit(1)
