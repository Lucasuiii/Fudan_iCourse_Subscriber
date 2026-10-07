"""Pure source checks, retry decisions and a bounded verified-byte buffer.

No authentication, sockets, files or decoding live here. The transport owns
those resources; this module decides which bytes and recoveries are safe.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from http.client import IncompleteRead
import re
import threading
from urllib.parse import parse_qsl, urljoin, urlsplit

import requests
from urllib3.exceptions import ReadTimeoutError


class MediaTransportError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)  # Fixed codes only; never provider messages.


def connection_failure_code(error):
    """Inspect bounded exception types, never parse or retain provider messages."""
    pending = [error]
    for _ in range(8):
        if not pending: break
        current = pending.pop()
        if isinstance(current, IncompleteRead): return 'upstream_premature_eof'
        if isinstance(current, (TimeoutError, requests.exceptions.Timeout, ReadTimeoutError)):
            return 'upstream_timeout'
        pending.extend(arg for arg in current.args if isinstance(arg, BaseException))
        if current.__cause__ is not None: pending.append(current.__cause__)
    return 'upstream_connection_error'


def media_identity(url):
    parts = urlsplit(url)
    query = tuple(sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                         if k not in ('t', 'clientUUID')))
    return parts.scheme, parts.netloc, parts.path, query


class MediaSource:
    """Freeze both selected media and resolved target, then response validators."""
    def __init__(self, signed_url):
        self.selected = media_identity(signed_url)
        self.target = None
        self.total = None
        self.etag = self.modified = None

    def bind_request(self, fresh_url, target):
        if media_identity(fresh_url) != self.selected:
            raise MediaTransportError('source_changed')
        identity = media_identity(target)
        if self.target is not None and identity != self.target:
            raise MediaTransportError('source_changed')
        self.target = identity

    @property
    def validator_kind(self):
        return 'etag' if self.etag and not self.etag.startswith('W/') else 'last_modified'

    def conditional_headers(self):
        if self.etag and not self.etag.startswith('W/'):
            return {'If-Match': self.etag}
        return {'If-Unmodified-Since': self.modified} if self.modified else {}

    def verify_range(self, response, start, end, *, open_ended=False):
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
        if not match: raise MediaTransportError('invalid_content_range')
        first, last, total = map(int, match.groups())
        expected_last = total-1 if open_ended else min(end, total-1)
        if total <= 0 or first != start or last != expected_last or last < first:
            raise MediaTransportError('invalid_content_range')
        etag, modified = response.headers.get('ETag'), response.headers.get('Last-Modified')
        if self.total is not None and (total != self.total or etag != self.etag or modified != self.modified):
            raise MediaTransportError('source_changed')
        if self.total is None and not (etag and not etag.startswith('W/') or modified):
            raise MediaTransportError('source_validator_missing')
        if response.headers.get('Content-Encoding', 'identity').lower() not in ('', 'identity'):
            raise MediaTransportError('encoded_range')
        length = response.headers.get('Content-Length')
        if length is not None and (not length.isdigit() or int(length) != last-first+1):
            raise MediaTransportError('invalid_content_length')
        # Bind only after all header checks pass. No response body is consumed.
        if self.total is None:
            self.total, self.etag, self.modified = total, etag, modified
        return min(last, end)-first+1


LOGIN_PATHS = ('/login', '/wengine-vpn/login', '/cas/login', '/authserver/login',
               '/authserver/authenticate', '/idp/authCenter/authenticate')


def redirect_kind(response, *, login_urls=()):
    """Classify safely; never follow or export a replacement signed URL."""
    location = response.headers.get('Location')
    if not location: return 'missing'
    try:
        source = urlsplit(response.url)
        target = urlsplit(urljoin(response.url, location))
        if (target.scheme not in ('http', 'https') or target.username or target.password
                or source.scheme == 'https' and target.scheme != 'https'):
            return 'other'
        same_origin = (source.scheme, source.netloc) == (target.scheme, target.netloc)
        login_path = target.path.rstrip('/') in LOGIN_PATHS
        if login_path and (same_origin or (target.scheme, target.netloc) == ('https', 'id.fudan.edu.cn')):
            return 'login'
        # The API adapter supplies exact, configured WebVPN encodings of SSO
        # entrypoints. A login-looking suffix inside an arbitrary proxy route
        # is insufficient; do not decode or follow an untrusted destination.
        if isinstance(login_urls, (tuple, list)):
            for url in login_urls[:16]:
                known = urlsplit(url)
                if (not known.username and not known.password
                        and (target.scheme, target.netloc, target.path.rstrip('/'))
                        == (known.scheme, known.netloc, known.path.rstrip('/'))):
                    return 'login'
        if same_origin and source.path == target.path: return 'same_media'
    except ValueError:
        pass
    return 'other'


def redirect_observation(response):
    """Enum/boolean evidence only: never expose Location, paths or query values."""
    row = {'location_present': bool(response.headers.get('Location'))}
    try:
        source = urlsplit(response.url)
        target = urlsplit(urljoin(response.url, response.headers.get('Location', '')))
        same_origin = (source.scheme, source.netloc) == (target.scheme, target.netloc)
        route = ('login_path' if target.path.rstrip('/') in LOGIN_PATHS else
                 'root' if target.path in ('', '/') else
                 'vpn_wrapped' if target.path.startswith(('/https/', '/http/')) else
                 'vpn_control' if target.path.startswith('/wengine-vpn/') else
                 'same_media' if same_origin and target.path == source.path else 'other')
        row.update(authority=('same_origin' if same_origin else
                   'fudan_sso' if (target.scheme, target.netloc) == ('https', 'id.fudan.edu.cn') else 'other'),
                   route=route, https=target.scheme == 'https',
                   downgrade=source.scheme == 'https' and target.scheme != 'https',
                   credential_authority=bool(target.username or target.password),
                   query_present=bool(target.query))
    except (ValueError, TypeError):
        row['malformed'] = True
    return row


class RecoveryAction(Enum):
    RETRY = 'retry'
    REFRESH = 'refresh'
    STOP = 'stop'


@dataclass(frozen=True)
class RangeRecoveryPolicy:
    attempts: int = 3

    def __post_init__(self):
        if type(self.attempts) is not int or not 1 <= self.attempts <= 4:
            raise ValueError('Invalid bounded retry limit')

    def decide(self, code, attempt, *, can_refresh):
        if code == 'media_session_refresh_needed':
            return RecoveryAction.REFRESH if can_refresh and attempt+1 < self.attempts else RecoveryAction.STOP
        retryable = {'upstream_retryable_http', 'upstream_premature_eof',
                     'range_not_honored', 'upstream_connection_error', 'upstream_timeout'}
        return RecoveryAction.RETRY if code in retryable and attempt+1 < self.attempts else RecoveryAction.STOP

    @staticmethod
    def terminal_code(code):
        if code == 'media_session_refresh_needed': return 'media_session_unavailable'
        if code == 'upstream_http_rejected': return 'range_not_honored'
        if code in ('upstream_retryable_http', 'upstream_premature_eof', 'upstream_connection_error', 'upstream_timeout'):
            return 'upstream_retries_exhausted'
        return code

    @staticmethod
    def delay(code, attempt):
        return 2*(attempt+1) if code == 'range_not_honored' else 0


class VerifiedRangeBuffer:
    """LRU of complete, checked ranges within one relay's frozen source.

    Never persists credentials or partial reads. This is a per-process buffer,
    not a durable download checkpoint. Both bytes and entry count are bounded.
    """
    def __init__(self, capacity=0, max_entries=128):
        if type(capacity) is not int or not 0 <= capacity <= 64*1024*1024:
            raise ValueError('Invalid media buffer capacity')
        if type(max_entries) is not int or not 1 <= max_entries <= 128:
            raise ValueError('Invalid media buffer entry limit')
        self.capacity, self.max_entries = capacity, max_entries
        self.bytes = 0
        self._ranges = OrderedDict()
        self._lock = threading.Lock()
        self._closed = False

    def get(self, start, end):
        with self._lock:
            for first, data in reversed(self._ranges.items()):
                if first <= start <= end < first+len(data):
                    self._ranges.move_to_end(first)
                    return data[start-first:end-first+1]
        return None

    def put(self, start, data):
        if not data or len(data) > self.capacity: return
        data = bytes(data)
        with self._lock:
            if self._closed: return
            old = self._ranges.pop(start, b'')
            self.bytes -= len(old)
            self._ranges[start] = data
            self.bytes += len(data)
            while self.bytes > self.capacity or len(self._ranges) > self.max_entries:
                _, discarded = self._ranges.popitem(last=False)
                self.bytes -= len(discarded)

    def close(self):
        with self._lock:
            self._closed = True
            self._ranges.clear()
            self.bytes = 0
