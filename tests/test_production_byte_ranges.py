"""Bounded HTTP samples diagnose seek behavior without exporting media."""
import io
import json
import unittest
from unittest.mock import MagicMock
from scripts.production_media_inspection import probe_byte_ranges,safe_probe_errors,probe_initial_range_reuse
from urllib.parse import parse_qs,urlsplit


class SourceByteRangeTests(unittest.TestCase):
    def test_uuid_contrast_preserves_signature_timestamp_and_auth_headers(self):
        session,responses=self.session('valid')
        result=probe_initial_range_reuse(session,'https://private/source?t=secret&clientUUID=old&now=123',
                                        'Cookie: private-cookie\r\n',1808388390)
        urls=[call.args[0] for call in session.get.call_args_list]
        parsed=[parse_qs(urlsplit(url).query) for url in urls]
        self.assertEqual(urls[0],urls[1]);self.assertEqual(urls[1],urls[2]);self.assertNotEqual(urls[2],urls[3])
        for row in parsed:
            self.assertEqual(row['t'],['secret']);self.assertEqual(row['now'],['123'])
        for index,call in enumerate(session.get.call_args_list):
            self.assertEqual(call.kwargs['headers']['Cookie'],'private-cookie')
            self.assertEqual(call.kwargs['headers']['Range'],'bytes=0-4095' if index==0 else 'bytes=0-')
        self.assertEqual(result['maximum_requests'],4)
        for secret in ('private','secret','123','old'):
            self.assertNotIn(secret,json.dumps(result))

    def test_http_trace_exports_offsets_and_statuses_but_not_private_headers(self):
        text=(b'GET /signed/private?key=secret HTTP/1.1\nCookie: private-cookie\n'
              b'Range: bytes=1138114319-\nHTTP/1.1 403 Forbidden\n'
              b'Server returned 403 Forbidden (access denied)\n')
        result=safe_probe_errors(text)
        self.assertEqual(result['http_error_statuses'],[403])
        self.assertEqual(result['http_range_requests'],[{'start':1138114319,'end':None}])
        self.assertEqual(result['http_response_statuses'],[403])
        for secret in ('signed','private','secret','Cookie'):
            self.assertNotIn(secret,json.dumps(result))

    def test_unclassified_probe_failure_only_retains_fixed_words_not_values(self):
        result=safe_probe_errors(b"Failed to set value 'private-cookie-123' for option 'read_intervals': Operation not permitted https://private/signed?key=secret\n")
        self.assertIn('read_intervals',result['unclassified_terms'])
        self.assertIn('permitted',result['unclassified_terms'])
        for secret in ('private','cookie','123','secret','signed'):
            self.assertNotIn(secret,json.dumps(result))

    def session(self, mode):
        session = MagicMock()
        responses = []
        def get(url, **kwargs):
            start,end = kwargs['headers']['Range'][6:].split('-')
            start=int(start);open_ended=not end;end=int(end) if end else 1808388389
            response = MagicMock()
            response.status_code = 200 if mode == 'ignored' else 206
            response.headers = {'Content-Type':'video/mp4','Content-Length':'4096',
                'Content-Range':f'bytes {start}-{end}/1808388390','Accept-Ranges':'bytes'}
            if mode == 'wrong': response.headers['Content-Range'] = 'bytes 0-4095/1808388390'
            if mode == 'ignored': response.headers.pop('Content-Range')
            body=io.BytesIO(b'x'*(8192 if mode=='ignored' or open_ended else 4096))
            response.raw = MagicMock()
            response.raw.read.side_effect=lambda size,**unused:body.read(size)
            responses.append(response)
            return response
        session.get.side_effect=get
        return session,responses

    def test_offsets_around_one_gib_and_file_tail_validate_response_ranges(self):
        session,responses=self.session('valid')
        result=probe_byte_ranges(session,'private-signed-url','Cookie: private-cookie\r\n',1808388390)
        self.assertEqual(result['status'],'complete')
        self.assertEqual([r['start'] for r in result['requests']],
                         [0,1048576,1073737728,1073741824,1808384294])
        for call,response in zip(session.get.call_args_list,responses):
            self.assertFalse(call.kwargs['allow_redirects'])
            self.assertEqual(call.kwargs['headers']['Accept-Encoding'],'identity')
            response.raw.read.assert_called_once_with(4097,decode_content=False)
            response.close.assert_called_once()
        self.assertNotIn('private',json.dumps(result))
        self.assertFalse(result['payload_exported'])
        self.assertFalse(result['same_source_bytes_verified'])

    def test_ignored_or_wrong_ranges_never_count_as_seek_success(self):
        for mode in ('ignored','wrong'):
            session,responses=self.session(mode)
            result=probe_byte_ranges(session,'private','Cookie: private\r\n',1808388390)
            self.assertEqual(result['status'],'failed')
            self.assertLessEqual(sum(r['bytes_read'] for r in result['requests']),20485)
            self.assertTrue(any(not r['range_valid'] for r in result['requests']))
            for response in responses:response.close.assert_called_once()

    def test_read_failure_closes_response_and_exports_only_exception_type(self):
        session,responses=self.session('valid')
        original=session.get.side_effect
        def get(*args,**kwargs):
            response=original(*args,**kwargs)
            response.raw.read.side_effect=OSError('private URL and token')
            return response
        session.get.side_effect=get
        result=probe_byte_ranges(session,'private','Cookie: secret\r\n',1808388390)
        self.assertEqual(result['status'],'failed')
        self.assertEqual([r['failure_type'] for r in result['requests']],['OSError']*5)
        self.assertNotIn('private',json.dumps(result))
        for response in responses:response.close.assert_called_once()

    def test_missing_size_does_not_fetch_and_small_files_deduplicate_ranges(self):
        session=MagicMock()
        self.assertEqual(probe_byte_ranges(session,'url','',None)['status'],'size_unavailable')
        session.get.assert_not_called()
        session,responses=self.session('valid')
        result=probe_byte_ranges(session,'url','',4096)
        self.assertEqual(len(result['requests']),1)


if __name__=='__main__': unittest.main()
