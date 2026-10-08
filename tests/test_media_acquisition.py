"""Acquisition recovery, verified buffering and decoder gates without live media."""
import json
from http.client import IncompleteRead
import threading
import time
from types import SimpleNamespace
import unittest

import requests
from src.runtime.media_protocol import (MediaSource, MediaTransportError, connection_failure_code,
    RangeRecoveryPolicy, RecoveryAction, VerifiedRangeBuffer, redirect_kind)
from src.runtime.media_transport import SignedRangeRelay
from src.runtime.audio_preparation import DecodeErrorScanner, validate_prepared_audio
from test_media_transport import Origin


class AcquisitionProtocolTests(unittest.TestCase):
    def test_nested_network_errors_keep_safe_specific_cause(self):
        from urllib3.exceptions import ProtocolError
        self.assertEqual(connection_failure_code(ProtocolError('private URL', IncompleteRead(b''))),
                         'upstream_premature_eof')
        self.assertEqual(connection_failure_code(requests.exceptions.ReadTimeout('private URL')), 'upstream_timeout')
        self.assertEqual(connection_failure_code(ConnectionResetError('private URL')), 'upstream_connection_error')

    def test_bounded_policy_preserves_permanent_failures_and_session_limit(self):
        policy = RangeRecoveryPolicy(3)
        for code in ('upstream_retryable_http', 'upstream_premature_eof',
                     'upstream_connection_error', 'range_not_honored'):
            self.assertEqual(policy.decide(code, 0, can_refresh=False), RecoveryAction.RETRY)
            self.assertEqual(policy.decide(code, 2, can_refresh=True), RecoveryAction.STOP)
        for code in ('source_changed', 'invalid_content_range', 'invalid_content_length',
                     'encoded_range', 'media_redirect_untrusted', 'diagnostic_byte_limit'):
            self.assertEqual(policy.decide(code, 0, can_refresh=True), RecoveryAction.STOP)
            self.assertEqual(policy.terminal_code(code), code)
        self.assertEqual(policy.decide('media_session_refresh_needed', 0, can_refresh=True), RecoveryAction.REFRESH)
        for attempt, allowed in [(0, False), (2, True)]:
            self.assertEqual(policy.decide('media_session_refresh_needed', attempt, can_refresh=allowed), RecoveryAction.STOP)
        for limit in (0, 5, True, 2.5):
            with self.assertRaises(ValueError): RangeRecoveryPolicy(limit)

    def test_bad_headers_cannot_bind_source_or_become_cached(self):
        source = MediaSource('https://media.example/one?track=1&t=old')
        response = SimpleNamespace(headers={'Content-Range': 'bytes 0-9/10', 'ETag': '"v1"',
                                             'Content-Length': '9'})
        with self.assertRaises(MediaTransportError): source.verify_range(response, 0, 9)
        self.assertIsNone(source.total)
        response.headers['Content-Length'] = '10'
        self.assertEqual(source.verify_range(response, 0, 9), 10)
        self.assertEqual(source.conditional_headers(), {'If-Match': '"v1"'})
        for changes in ({'ETag': '"v2"'}, {'Content-Range': 'bytes 1-9/10'}, {'Content-Encoding': 'gzip'}):
            bad = SimpleNamespace(headers=dict(response.headers, **changes))
            with self.assertRaises(MediaTransportError): source.verify_range(bad, 0, 9)

    def test_buffer_bounds_bytes_entries_and_reuses_only_contained_ranges(self):
        cache = VerifiedRangeBuffer(8, max_entries=2)
        cache.put(0, b'abcd'); cache.put(4, b'efgh')
        self.assertEqual(cache.get(1, 2), b'bc')
        self.assertIsNone(cache.get(2, 5))  # No speculative stitching.
        cache.put(8, b'ijkl')
        self.assertIsNone(cache.get(4, 7)); self.assertEqual(cache.bytes, 8)
        cache.put(20, b'too large for this cache')
        self.assertEqual(cache.bytes, 8)
        cache.close(); cache.put(0, b'abcd')
        self.assertIsNone(cache.get(0, 3)); self.assertEqual(cache.bytes, 0)
        tiny = VerifiedRangeBuffer(1024, max_entries=2)
        for i in range(20): tiny.put(i, b'x')
        self.assertEqual(tiny.bytes, 2)

    def test_login_redirect_requires_exact_trusted_authority(self):
        for target in ('https://id.fudan.edu.cn:444/login', 'https://id.fudan.edu.cn.evil.test/login'):
            response = SimpleNamespace(url='https://vpn.example/media', headers={'Location': target})
            self.assertEqual(redirect_kind(response), 'other')

    def test_wrapped_login_matches_exact_configured_route_not_just_suffix(self):
        from src.api.icourse import ICourseClient
        from src.api.webvpn import get_vpn_url
        client = ICourseClient(None)
        routes = client.trusted_media_login_urls()
        source = get_vpn_url('https://icourse.fudan.edu.cn/media')
        good = get_vpn_url('https://id.fudan.edu.cn/idp/authCenter/authenticate')
        for target, expected in [(good+'?ticket=private','login'),
                                 (good+'/', 'login'),
                                 (good+'/unexpected','other'),
                                 (get_vpn_url('https://foreign.invalid/idp/authCenter/authenticate'),'other'),
                                 (good.replace('webvpn.fudan.edu.cn','webvpn.fudan.edu.cn.evil.test'),'other'),
                                 (good.replace('https://webvpn','http://webvpn'),'other')]:
            response = SimpleNamespace(url=source,headers={'Location':target})
            self.assertEqual(redirect_kind(response,login_urls=routes),expected)

    def test_redirect_evidence_never_exports_authority_path_or_ticket(self):
        from src.runtime.media_protocol import redirect_observation
        response = SimpleNamespace(url='https://vpn.example/private-media',headers={
            'Location':'https://private-user:private-pass@unknown.example/private-path?ticket=private-ticket'})
        row = redirect_observation(response)
        self.assertTrue(row['credential_authority'])
        self.assertTrue(row['query_present'])
        self.assertEqual(row['authority'],'other')
        self.assertEqual(row['route'],'other')
        for secret in ('unknown.example','private-path','private-user','private-pass','private-ticket'):
            self.assertNotIn(secret,json.dumps(row))


class AcquisitionHTTPTests(unittest.TestCase):
    DATA = bytes(range(256))*4096

    def test_permanent_http_rejection_does_not_gain_transient_retries(self):
        for status in (400, 404, 422):
            with Origin(self.DATA, rejected_status=status) as origin:
                relay = SignedRangeRelay(origin.client, origin.signed, attempts=3)
                with self.assertRaises(MediaTransportError): relay.start()
                self.assertEqual(len(origin.requests), 1)
                audit = relay.audit()
                self.assertEqual(audit['upstream_bytes'], 0)
                self.assertEqual(audit['last_error_code'], 'upstream_http_rejected')
                self.assertEqual(audit['last_failure_stage'], 'validating')
                self.assertEqual(audit['retries'], 0)

    def test_real_reopen_uses_complete_verified_cache_and_close_releases_it(self):
        with Origin(self.DATA) as origin:
            relay = SignedRangeRelay(origin.client, origin.signed, prefix_bytes=4096,
                                     chunk_bytes=131072, cache_bytes=262144).start()
            try:
                first = requests.get(relay.url, headers={'Range': 'bytes=4096-135167'}, timeout=10)
                self.assertEqual(first.content, self.DATA[4096:135168])
                count = len(origin.requests)
                second = requests.get(relay.url, headers={'Range': 'bytes=5000-5999'}, timeout=10)
                self.assertEqual(second.content, self.DATA[5000:6000])
                self.assertEqual(len(origin.requests), count)
                audit = relay.audit()
                self.assertEqual(audit['cache_hits'], 1)
                self.assertEqual(audit['cached_bytes_served'], 1000)
                self.assertLessEqual(audit['cache_bytes'], 262144)
                self.assertEqual(audit['state'], 'ready')
                for secret in ('http://', 'ticket', 'fake-private', 'immutable'):
                    self.assertNotIn(secret, json.dumps(audit))
            finally: relay.close()
            self.assertEqual(relay.audit()['cache_bytes'], 0)
            self.assertEqual(relay.audit()['state'], 'closed')

    def test_partial_failed_range_never_enters_cache_or_serves_old_prefix(self):
        with Origin(self.DATA, drop_start=4096) as origin, SignedRangeRelay(
                origin.client, origin.signed, prefix_bytes=4096, attempts=1, cache_bytes=262144) as relay:
            with self.assertRaises(MediaTransportError): relay._read_range(4096, 135167)
            audit = relay.audit()
            self.assertEqual(audit['cache_bytes'], 4096)  # Only completed probe.
            self.assertEqual(audit['last_failure_offset'], 69632)
            self.assertEqual(audit['last_error_code'], 'upstream_premature_eof')
            self.assertEqual(audit['last_failure_stage'], 'reading')
            self.assertEqual(audit['state'], 'failed')
            response = requests.get(relay.url, headers={'Range': 'bytes=0-99'}, timeout=10)
            self.assertEqual(response.status_code, 503)
            self.assertNotEqual(response.content, self.DATA[:100])

    def test_source_mutation_blocks_reuse_of_previously_verified_cache(self):
        with Origin(self.DATA) as origin, SignedRangeRelay(origin.client, origin.signed,
                prefix_bytes=4096, cache_bytes=262144) as relay:
            self.assertEqual(relay._read_range(4096, 8191), self.DATA[4096:8192])
            origin.change = True
            with self.assertRaises(MediaTransportError) as error: relay._read_range(8192, 12287)
            self.assertEqual(error.exception.code, 'source_changed')
            self.assertEqual(relay.audit()['last_failure_stage'], 'validating')
            with self.assertRaises(MediaTransportError): relay._read_range(4096, 8191)
            self.assertEqual(relay.audit()['cache_hits'], 0)

    def test_shutdown_does_not_wait_for_signing_or_restart_network_after_it(self):
        with Origin(self.DATA) as origin:
            relay = SignedRangeRelay(origin.client, origin.signed, prefix_bytes=4096, cache_bytes=8192).start()
            entered, release = threading.Event(), threading.Event()
            original = origin.client.renew_video_url
            def delayed(*args, **kwargs):
                entered.set(); release.wait(5)
                return original(*args, **kwargs)
            origin.client.renew_video_url = delayed
            errors = []
            def fetch():
                try: relay._read_range(4096, 8191)
                except MediaTransportError as error: errors.append(error.code)
            thread = threading.Thread(target=fetch); thread.start()
            try:
                self.assertTrue(entered.wait(2))
                before = time.monotonic(); relay.close()
                self.assertLess(time.monotonic()-before, 2)
            finally:
                release.set(); thread.join(3); relay.close()
            self.assertFalse(thread.is_alive()); self.assertEqual(errors, ['stopped'])
            self.assertEqual(len(origin.requests), 1)
            self.assertEqual(relay.audit()['state'], 'closed')


class DecoderGateTests(unittest.TestCase):
    def test_split_error_is_detected_once_and_survives_tail_rotation(self):
        counts = {}; scanner = DecodeErrorScanner(counts)
        scanner.feed(b'Error during demu'); scanner.feed(b'xing: private-url\n')
        scanner.feed(b'normal progress\n')
        self.assertEqual(counts, {'input_read_error': 1})
        scanner.feed(b'x'*10000)
        scanner.feed(b'Error while deco'); scanner.feed(b'ding stream\n')
        self.assertEqual(counts, {'input_read_error': 1, 'decode_error': 1})
        self.assertNotIn('private', json.dumps(counts))
        self.assertLess(len(scanner._tail), 64)

    def test_nonfinite_empty_or_unaligned_audio_is_never_eligible(self):
        for duration in (0, -1, float('nan'), float('inf'), True):
            spec = {'audio_seconds': duration, 'media_seconds': 100,
                    'audio_diagnostics': {'decode_return_code': 0}}
            with self.assertRaisesRegex(ValueError, 'invalid sample'): validate_prepared_audio(spec)
        spec = {'audio_seconds': 100, 'media_seconds': 100, 'audio_diagnostics': {
            'decode_return_code': 0, 'pcm_sample_aligned': False}}
        with self.assertRaisesRegex(ValueError, 'invalid sample'): validate_prepared_audio(spec)
        spec['audio_diagnostics']['pcm_sample_aligned'] = True
        validate_prepared_audio(spec)
