"""Bounded HTTP samples diagnose seek behavior without exporting media."""
import io
import json
import unittest
from unittest.mock import MagicMock
from scripts.production_media_inspection import probe_byte_ranges


class SourceByteRangeTests(unittest.TestCase):
    def session(self, mode):
        session = MagicMock()
        responses = []
        def get(url, **kwargs):
            start,end = map(int,kwargs['headers']['Range'][6:].split('-'))
            response = MagicMock()
            response.status_code = 200 if mode == 'ignored' else 206
            response.headers = {'Content-Type':'video/mp4','Content-Length':'4096',
                'Content-Range':f'bytes {start}-{end}/1808388390','Accept-Ranges':'bytes'}
            if mode == 'wrong': response.headers['Content-Range'] = 'bytes 0-4095/1808388390'
            if mode == 'ignored': response.headers.pop('Content-Range')
            body=io.BytesIO(b'x'*(8192 if mode=='ignored' else 4096))
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
