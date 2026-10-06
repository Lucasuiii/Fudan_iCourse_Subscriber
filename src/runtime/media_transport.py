"""Loopback byte-range transport for an immutable, freshly signed media source.

Only bounded upstream ranges are buffered. A failed range resumes at its next
unread byte; FFmpeg never sees duplicate bytes or an upstream signed URL.
No login, audio decoding, model, database or publisher is invoked here.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import re
import secrets
import threading
import time
from urllib.parse import parse_qsl, urlsplit

import requests


class MediaTransportError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)  # Fixed codes only; never provider messages.


class SignedRangeRelay:
    def __init__(self, client, signed_url, *, chunk_bytes=8*1024*1024,
                 prefix_bytes=64*1024, attempts=3, max_upstream_bytes=None,
                 session_factory=requests.Session, timeout=(10, 15)):
        self.client, self.signed_url = client, signed_url
        self.chunk_bytes, self.prefix_bytes = chunk_bytes, prefix_bytes
        self.attempts, self.max_bytes = attempts, max_upstream_bytes
        self.session_factory, self.timeout = session_factory, timeout
        self.total = None
        self._etag = self._modified = None
        self._prefix = b''
        self._stop = threading.Event()
        self._fetch_lock = threading.Lock()
        self._audit_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._response_lock = threading.Lock()
        self._responses = set()
        self._last_initial_time = None
        self._server = self._thread = None
        self._closed = False
        self._path = '/'+secrets.token_urlsafe(32)
        self.url = None
        self._audit = dict(mode='signed_range_relay', range_requests=0,
                           range_verified=0, retries=0, signature_renewals=0,
                           upstream_bytes=0, source_total_bytes=None,
                           validator_kind=None, terminal_error_code=None)
        if not 4096 <= chunk_bytes <= 16*1024*1024 or not 1 <= attempts <= 4:
            raise ValueError('Invalid bounded media transport limits')
        if not 1 <= prefix_bytes <= chunk_bytes:
            raise ValueError('Invalid media prefix limit')

    def audit(self):
        with self._audit_lock:
            return dict(self._audit)

    def _count(self, key, amount=1):
        with self._audit_lock:
            self._audit[key] += amount

    def _fail(self, code):
        if not self._stop.is_set():
            with self._audit_lock:
                self._audit['terminal_error_code'] = code
        raise MediaTransportError(code)

    def _signed_request(self, start, retry):
        now = int(time.time())
        # Initial-byte reuse may be rejected even with a new UUID. On retry,
        # wait for the actual clock to advance, never fabricate a future time.
        original = dict(parse_qsl(urlsplit(self.signed_url).query)).get('t','')
        try: original_time = int(original.rsplit('-',2)[-2])
        except (ValueError, IndexError): original_time = None
        if (start == 0 and now in (original_time, self._last_initial_time)) or retry:
            previous = now
            while now == previous:
                if self._stop.wait(.05): raise MediaTransportError('stopped')
                now = int(time.time())
        fresh = self.client.renew_video_url(self.signed_url, now=now)
        # Renewing authentication must never change the selected media path.
        before, after = urlsplit(self.signed_url), urlsplit(fresh)
        def selection_query(parts):
            return sorted((k,v) for k,v in parse_qsl(parts.query,keep_blank_values=True)
                          if k not in ('t','clientUUID'))
        if ((before.scheme, before.netloc, before.path) != (after.scheme, after.netloc, after.path)
                or selection_query(before) != selection_query(after)):
            self._fail('source_changed')
        target, raw_headers = self.client.get_stream_params(fresh)
        headers = dict(line.split(':',1) for line in raw_headers.split('\r\n') if ':' in line)
        headers = {key.strip(): value.strip() for key,value in headers.items()}
        headers['Accept-Encoding'] = 'identity'
        if self._etag and not self._etag.startswith('W/'):
            headers['If-Match'] = self._etag
        elif self._modified:
            headers['If-Unmodified-Since'] = self._modified
        self._count('signature_renewals')
        if start == 0: self._last_initial_time = now
        return target, headers

    def _verify_response(self, response, start, end, *, open_ended=False):
        if response.status_code == 412: self._fail('source_changed')
        if response.status_code in (401,403,408,429,500,502,503,504):
            raise MediaTransportError('upstream_retryable_http')
        if response.status_code != 206: self._fail('range_not_honored')
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
        if not match: self._fail('invalid_content_range')
        first,last,total = map(int,match.groups())
        expected_last = total-1 if open_ended else min(end,total-1)
        if total <= 0 or first != start or last != expected_last or last < first:
            self._fail('invalid_content_range')
        etag, modified = response.headers.get('ETag'), response.headers.get('Last-Modified')
        if self.total is None:
            # Size alone cannot prevent splicing two same-size recordings.
            if not (etag and not etag.startswith('W/') or modified):
                self._fail('source_validator_missing')
            self.total, self._etag, self._modified = total,etag,modified
            with self._audit_lock:
                self._audit.update(source_total_bytes=total,
                                   validator_kind='etag' if etag and not etag.startswith('W/') else 'last_modified')
        elif (total != self.total or etag != self._etag or modified != self._modified):
            self._fail('source_changed')
        if response.headers.get('Content-Encoding','identity').lower() not in ('','identity'):
            self._fail('encoded_range')
        length = response.headers.get('Content-Length')
        if length is not None and (not length.isdigit() or int(length) != last-first+1):
            self._fail('invalid_content_length')
        self._count('range_verified')
        return min(last,end)-first+1

    def _read_range(self, start, end, *, initial_probe=False):
        data = bytearray()
        with self._fetch_lock:
            if self.audit()['terminal_error_code']: raise MediaTransportError('transport_failed')
            for attempt in range(self.attempts):
                if self._stop.is_set(): raise MediaTransportError('stopped')
                response = None
                session = self.session_factory()
                try:
                    offset = start+len(data)
                    target, headers = self._signed_request(offset, attempt > 0)
                    self._count('range_requests')
                    open_ended = initial_probe and offset == 0
                    range_value = f'bytes={offset}-'+('' if open_ended else str(end))
                    response = session.get(target, headers={**headers,'Range':range_value},
                                           stream=True, timeout=self.timeout, allow_redirects=False)
                    with self._response_lock: self._responses.add(response)
                    remaining = self._verify_response(response, offset, end,open_ended=open_ended)
                    while remaining:
                        if self._stop.is_set(): raise MediaTransportError('stopped')
                        size = min(64*1024, remaining)
                        if self.max_bytes is not None:
                            budget = self.max_bytes-self.audit()['upstream_bytes']
                            if budget <= 0: self._fail('diagnostic_byte_limit')
                            size = min(size,budget)
                        block = response.raw.read(size, decode_content=False)
                        if not block: raise MediaTransportError('upstream_premature_eof')
                        if len(block) > size: self._fail('invalid_read_length')
                        data.extend(block); remaining -= len(block)
                        self._count('upstream_bytes',len(block))
                    return bytes(data)
                except MediaTransportError as error:
                    if error.code not in ('upstream_retryable_http','upstream_premature_eof'):
                        raise
                except (requests.RequestException, OSError):
                    pass
                except Exception as error:
                    # urllib3 may raise ProtocolError/IncompleteRead from raw.
                    from urllib3.exceptions import HTTPError
                    if not isinstance(error,HTTPError): self._fail('transport_internal_error')
                finally:
                    if response is not None:
                        with self._response_lock: self._responses.discard(response)
                        response.close()
                    session.close()
                if self._stop.is_set(): raise MediaTransportError('stopped')
                if attempt+1 < self.attempts: self._count('retries')
            self._fail('upstream_retries_exhausted')

    def start(self):
        try:
            # Match the known-working initial bytes=0- form, but read only a
            # bounded prefix and close. Subsequent reads use closed ranges.
            self._prefix = self._read_range(0,self.prefix_bytes-1,initial_probe=True)
            owner = self
            class Handler(BaseHTTPRequestHandler):
                protocol_version = 'HTTP/1.1'
                def log_message(self,*args): pass
                def do_HEAD(self): self.serve(False)
                def do_GET(self): self.serve(True)
                def serve(self,body):
                    self.close_connection = True
                    if self.path != owner._path:
                        self.send_error(404); return
                    value = self.headers.get('Range')
                    start,end = 0,owner.total-1
                    if value:
                        match = re.fullmatch(r'bytes=(\d+)-(\d*)',value)
                        if not match:
                            self.send_error(416);return
                        start = int(match[1]); end = min(int(match[2]) if match[2] else end,end)
                        if start > end:
                            self.send_error(416);return
                    self.send_response(206 if value else 200)
                    self.send_header('Accept-Ranges','bytes')
                    self.send_header('Content-Type','application/octet-stream')
                    self.send_header('Content-Length',str(end-start+1))
                    self.send_header('Connection','close')
                    if value: self.send_header('Content-Range',f'bytes {start}-{end}/{owner.total}')
                    self.end_headers()
                    if not body:return
                    try:
                        position = start
                        while position <= end and not owner._stop.is_set():
                            if position < len(owner._prefix):
                                data = owner._prefix[position:min(len(owner._prefix),end+1)]
                            else:
                                finish = min(position+owner.chunk_bytes-1,end)
                                data = owner._read_range(position,finish)
                            self.wfile.write(data); self.wfile.flush()
                            position += len(data)
                    except (OSError,MediaTransportError):
                        # Closing a short response lets the existing FFmpeg EOF
                        # and duration gates reject it. No fabricated tail.
                        return
            class Server(ThreadingHTTPServer):
                daemon_threads = True
                def handle_error(self,*args): pass  # No private traceback.
            self._server = Server(('127.0.0.1',0),Handler)
            self.url = f'http://127.0.0.1:{self._server.server_port}{self._path}'
            self._thread = threading.Thread(target=self._server.serve_forever,daemon=True)
            self._thread.start()
            return self
        except Exception:
            self.close()
            raise

    def close(self):
        with self._close_lock:
            if self._closed:return
            self._closed = True; self._stop.set()
            with self._response_lock: responses = list(self._responses)
            for response in responses: response.close()
            if self._server is not None:
                if self._thread is not None:self._server.shutdown()
                self._server.server_close()
            if self._thread is not None:self._thread.join(timeout=2)

    def __enter__(self): return self.start()
    def __exit__(self,*args): self.close()
