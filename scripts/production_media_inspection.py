"""Read source headers, five small byte ranges and late packet timestamps.

No media file, body/packet payload export, decoding, ASR or publisher are used.
The current source's metadata is diagnostic evidence, not proof that a previous
download of that lecture had the same bytes or was complete.
"""
from contextlib import redirect_stdout, redirect_stderr
import io
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from urllib.parse import parse_qsl,urlsplit,urlunsplit,urlencode


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
    statuses = sorted({int(code) for code in re.findall(
        rb'(?:http error|server returned) ([45]\d\d)', lowered)})
    result = {'error_counts':counts, 'http_error_statuses':statuses}
    ranges = re.findall(rb'\bRange: bytes=(\d{1,16})-(\d{0,16})',stderr)
    if ranges:
        result['http_range_requests'] = [{'start':int(a),'end':int(z) if z else None}
                                         for a,z in ranges[:16]]
    response_statuses = re.findall(rb'HTTP/1\.[01] ([1-5]\d\d)',stderr)
    if response_statuses: result['http_response_statuses'] = [int(s) for s in response_statuses[:16]]
    if lowered.strip() and not counts and not statuses:
        # Preserve a fixed diagnostic vocabulary, never arbitrary provider text
        # or numeric tokens that could be identifiers, timestamps or signatures.
        vocabulary = set(b'failed error invalid unknown unsupported option value set for '
            b'could not seek read position packet packets stream streams codec decoder '
            b'decoding decoded frames find found interval intervals specification '
            b'rw_timeout headers read_intervals select_streams show_packets show_entries '
            b'nofind_stream_info opening input file no operation permitted arguments '
            b'argument avformat demuxing match section entries print format '
            b'server returned forbidden access denied'.split())
        terms = [word.decode('ascii') for word in re.findall(rb'[a-z][a-z_-]*',lowered[:8192])
                 if word in vocabulary][:96]
        if terms: result['unclassified_terms'] = terms
    return result


def probe_byte_ranges(session, url, headers, total_bytes, *, starts=None,open_ended=False):
    """Five 4-KiB samples only; close ignored ranges without downloading media.

    No response body, URL, cookies or redirect destinations are exported.
    Matching Content-Range is required; HTTP 200 alone is not a successful seek.
    """
    if (not isinstance(total_bytes, (int, float)) or not math.isfinite(total_bytes)
            or total_bytes != int(total_bytes) or not 4096 <= total_bytes <= 2**53):
        return {'status':'size_unavailable', 'requests':[]}
    total_bytes = int(total_bytes)
    request_headers = dict(line.split(':', 1) for line in headers.split('\r\n') if ':' in line)
    request_headers = {k.strip():v.strip() for k,v in request_headers.items()}
    request_headers['Accept-Encoding'] = 'identity'
    width = 4096
    starts = sorted({0, min(1024**2,total_bytes-width),
                     min(1024**3-width,total_bytes-width),
                     min(1024**3,total_bytes-width),total_bytes-width}) if starts is None else starts
    if (not isinstance(starts,list) or not 1<=len(starts)<=5
            or any(type(n) is not int or not 0<=n<=total_bytes-width for n in starts)):
        raise ValueError('Invalid diagnostic byte ranges')
    rows = []
    for start in starts:
        end = start+width-1
        row = {'start':start, 'end':None if open_ended else end,
               'read_limit':width+1, 'range_valid':False,'bounded_sample_only':True}
        response = None
        began = time.monotonic()
        try:
            response = session.get(url, headers={**request_headers,'Range':f'bytes={start}-'+('' if open_ended else str(end))},
                stream=True, allow_redirects=False, timeout=(10,15))
            row['http_status'] = response.status_code
            row['redirect_present'] = 'Location' in response.headers
            row['accept_ranges_bytes'] = response.headers.get('Accept-Ranges','').lower() == 'bytes'
            content_type = response.headers.get('Content-Type','').lower()
            row['content_class'] = ('html' if 'text/html' in content_type else
                                    'media' if content_type.startswith(('video/','audio/','application/octet-stream')) else 'other')
            length = response.headers.get('Content-Length','')
            if re.fullmatch(r'\d{1,16}',length): row['content_length'] = int(length)
            match = re.fullmatch(r'bytes (\d{1,16})-(\d{1,16})/(\d{1,16})',
                                 response.headers.get('Content-Range',''))
            if match:
                row['content_range'] = dict(zip(('start','end','total'),map(int,match.groups())))
            body = response.raw.read(width+1, decode_content=False)
            row['bytes_read'] = len(body)
            row['range_valid'] = (response.status_code == 206
                and row.get('content_range') == {'start':start,'end':total_bytes-1 if open_ended else end,'total':total_bytes}
                and len(body) == (min(width+1,total_bytes-start) if open_ended else width)
                and row['content_class'] != 'html')
            if row['range_valid']: row['sample_sha256'] = hashlib.sha256(body).hexdigest()
        except Exception as error:
            row['failure_type'] = type(error).__name__
        finally:
            if response is not None: response.close()
            row['seconds'] = time.monotonic()-began
            rows.append(row)
    return {'status':'complete' if all(r['range_valid'] for r in rows) else 'failed',
            'maximum_requests':5, 'maximum_body_bytes':5*(width+1), 'payload_exported':False,
            'same_source_bytes_verified':False, 'requests':rows}


def probe_initial_range_reuse(session,url,headers,total_bytes):
    """Four bounded samples distinguish range shape from initial UUID reuse.

    Only clientUUID changes, preserving the path, t signature and timestamp.
    This is a diagnostic contrast, never a downloader authentication fallback.
    """
    parts=urlsplit(url);query=parse_qsl(parts.query,keep_blank_values=True)
    if sum(k=='clientUUID' for k,v in query)!=1:
        return {'status':'uuid_unavailable','requests':[]}
    def fresh():
        items=[(k,str(uuid.uuid4()) if k=='clientUUID' else v) for k,v in query]
        return urlunsplit((parts.scheme,parts.netloc,parts.path,urlencode(items),parts.fragment))
    first=fresh();rows=[]
    variants=[('fresh_uuid_closed_first',first,False),('same_uuid_open_first',first,True),
              ('same_uuid_open_repeat',first,True),('another_fresh_uuid_open_first',fresh(),True)]
    for label,target,open_ended in variants:
        result=probe_byte_ranges(session,target,headers,total_bytes,starts=[0],open_ended=open_ended)
        rows.append({'variant':label,**result})
    return {'status':'complete','maximum_requests':4,'maximum_body_bytes':4*4097,
            'changed_query_keys':['clientUUID'],'signature_and_timestamp_preserved':True,
            'payload_exported':False,'requests':rows}


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
    process = subprocess.run(['ffprobe', '-v', 'trace', '-nofind_stream_info',
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
    process = subprocess.run(['ffprobe', '-v', 'trace', '-nofind_stream_info',
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


def probe_fresh_late_packets(client, signed_url, retained_seconds, stream):
    """One fresh complete signature, same media path, at most 128 packets.

    Refreshing only clientUUID is insufficient to test a stale signature.
    Do not reuse the URL already opened by the header probe. No URL or token
    is included in the returned audit, and no audio is decoded.
    """
    parts = urlsplit(signed_url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    if sum(k == 't' for k, _ in query) != 1 or sum(k == 'clientUUID' for k, _ in query) != 1:
        return {'status':'signature_unavailable', 'packets':[]}
    base = urlunsplit(parts._replace(query=urlencode([(k,v) for k,v in query
                                                    if k not in ('t','clientUUID')])))
    # Use the actual current clock, never an invented future ticket time.
    refreshed = client.sign_video_url(base, now=int(time.time()))
    target, headers = client.get_stream_params(refreshed)
    result = probe_late_packets(target, headers, retained_seconds, stream)
    refreshed_query = dict(parse_qsl(urlsplit(refreshed).query))
    result.update(same_media_path=urlsplit(refreshed).path == parts.path,
                  full_signature_refreshed=refreshed_query.get('t') != dict(query)['t'],
                  url_opened_before_packet_probe=False)
    return result


def probe_relay_late_packets(client, signed_url, retained_seconds, stream):
    """Exercise production range transport within a 16-MiB read budget."""
    from src.runtime.media_transport import SignedRangeRelay, MediaTransportError
    relay = SignedRangeRelay(client,signed_url,chunk_bytes=1024*1024,
                             max_upstream_bytes=16*1024*1024)
    try:
        relay.start()
        result = probe_late_packets(relay.url,'',retained_seconds,stream)
    except MediaTransportError as error:
        result = {'status':'failed','failure_code':error.code,'packets':[]}
    except subprocess.TimeoutExpired:
        result = {'status':'failed','failure_type':'TimeoutExpired','packets':[]}
    finally:
        relay.close()
    result.update(source_transport=relay.audit(), maximum_upstream_bytes=16*1024*1024,
                  decoding=False, payload_exported=False)
    return result


def inspect():
    import base64
    from scripts import production_qwen as pipeline, sharded_qwen_pilot as shards
    from scripts.production_result_export import encrypt, identity, validate_inspection_source
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
        validate_inspection_source(info, run, slot)
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
        stage = 'byte_ranges'
        payload['current_source_byte_ranges'] = probe_byte_ranges(
            vpn.session, vpn_url, headers, metadata.get('format',{}).get('size'))
        stage = 'initial_range_reuse'
        payload['initial_range_reuse'] = probe_initial_range_reuse(
            vpn.session,vpn_url,headers,metadata.get('format',{}).get('size'))
        audio = [s for s in metadata.get('streams', []) if s.get('codec_type') == 'audio']
        retained = spec.get('audio_diagnostics', {}).get('audio_seconds')
        stage = 'late_packets'
        late = (probe_late_packets(vpn_url, headers, retained, audio[0])
                if len(audio) == 1 and isinstance(retained, (int,float))
                else {'status':'ambiguous_or_unavailable','packets':[]})
        payload['current_source_late_packets'] = late
        stage = 'fresh_late_packets'
        payload['fresh_source_late_packets'] = (
            probe_fresh_late_packets(client, url, retained, audio[0])
            if len(audio) == 1 and isinstance(retained, (int,float))
            else {'status':'ambiguous_or_unavailable','packets':[]})
        stage = 'relay_late_packets'
        payload['transport_source_late_packets'] = (
            probe_relay_late_packets(client,url,retained,audio[0])
            if len(audio) == 1 and isinstance(retained,(int,float))
            else {'status':'ambiguous_or_unavailable','packets':[]})
        if payload['transport_source_late_packets']['status'] == 'failed':
            raise ValueError('Source range transport probe failed')
        stage = 'late_packets'
        if late['status'] == 'failed': raise ValueError('Source late packet probe failed')
        if payload['current_source_byte_ranges']['status'] == 'failed':
            stage = 'byte_ranges'
            raise ValueError('Source byte range probe failed')
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
