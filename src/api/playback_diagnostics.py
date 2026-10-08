"""Fixed lookup error vocabulary; no provider messages, response bodies or URLs."""
import requests
FAILURES = {'tls_error', 'timeout', 'connection_error', 'http_error', 'invalid_json',
            'api_error', 'invalid_payload', 'other_error'}


def lookup_error(error):
    classes = ((requests.exceptions.SSLError, 'tls_error'),
               (requests.exceptions.Timeout, 'timeout'),
               (requests.exceptions.ConnectionError, 'connection_error'),
               (requests.exceptions.HTTPError, 'http_error'),
               (requests.exceptions.JSONDecodeError, 'invalid_json'),
               (RuntimeError, 'api_error'), (ValueError, 'invalid_payload'))
    result = {'result': 'failed', 'failure': next((code for cls, code in classes
                                                  if isinstance(error, cls)), 'other_error')}
    status = getattr(getattr(error, 'response', None), 'status_code', None)
    if type(status) is int and 100 <= status <= 599: result['http_status'] = status
    code = getattr(error, 'playback_api_code', None)
    if type(code) is int and -1_000_000 <= code <= 1_000_000: result['api_code'] = code
    return result


def attach_lookup(audit, client, course, sub):
    """A best-effort diagnostic cannot replace a downloader failure."""
    try:
        getter = getattr(client, 'video_lookup_diagnostics', None)
        value = getter(course, sub) if callable(getter) else None
        if (type(value) is dict and type(value.get('url_found')) is bool
                and type(value.get('sources')) is list and len(value['sources']) <= 3):
            rows = []
            for row in value['sources']:
                if type(row) is not dict or row.get('source') not in ('sub_info', 'sub_detail', 'signing'): continue
                if row.get('result') not in ('payload', 'failed'): continue
                clean = {'source': row['source'], 'result': row['result']}
                if row['result'] == 'failed':
                    clean['failure'] = row.get('failure') if row.get('failure') in FAILURES else 'other_error'
                    for key, low, high in (('http_status', 100, 599), ('api_code', -1_000_000, 1_000_000)):
                        if type(row.get(key)) is int and low <= row[key] <= high: clean[key] = row[key]
                rows.append(clean)
            audit['playback_lookup'] = {'sources': rows, 'url_found': value['url_found']}
    except Exception:
        pass
    return audit
