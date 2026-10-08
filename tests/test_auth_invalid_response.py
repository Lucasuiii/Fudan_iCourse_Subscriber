"""Malformed auth replies through the real flow, including a loopback HTTP server."""
import contextlib
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest
from unittest.mock import MagicMock
import requests
from src.api.auth_recovery import authenticated_session
from src.api.webvpn import authentication_failure
import test_webvpn_auth_diagnostics as fixtures
response = fixtures.response


def malformed(text='', status=200, *, redirected=False):
    value = requests.Response()
    value.status_code = status
    value._content = text.encode()
    value.url = 'https://private.invalid/?ticket=private'
    value.history = [response()] if redirected else []
    return value


def flow(reply, context):
    vpn = fixtures.AuthDiagnosticTests().session(cas=response(url='https://private.invalid/?lck='+context))
    posts = list(vpn.session.post.side_effect)
    posts[1] = reply
    vpn.session.post.side_effect = posts
    vpn.probe_login_service = MagicMock()
    vpn.login = MagicMock()
    return vpn


class InvalidAuthResponseTests(unittest.TestCase):
    def run_login(self, sessions, **kwargs):
        self.factory = MagicMock(side_effect=sessions)
        self.sleep = MagicMock()
        with contextlib.redirect_stdout(io.StringIO()):
            return authenticated_session(factory=self.factory, sleep=self.sleep,
                student_id='synthetic-student', password='synthetic-password', **kwargs)

    def test_empty_reply_restarts_with_a_new_context_and_strict_verification(self):
        first, second = flow(malformed(), 'old-context'), flow(response({'code':200,'loginToken':'new-token'}), 'new-context')
        self.assertIs(self.run_login([first, second]), second)
        self.assertEqual(self.factory.call_count, 2)
        first.session.close.assert_called_once()
        second.session.close.assert_not_called()
        self.sleep.assert_called_once_with(5)
        for vpn, context in ((first,'old-context'),(second,'new-context')):
            call = vpn.session.post.call_args_list[1]
            self.assertIn('/authExecute', call.args[0])
            self.assertEqual(call.kwargs['json']['lck'], context)
        self.assertEqual(first.session.post.call_count, 2)  # No ticket after failed reply.
        self.assertEqual(second.session.post.call_count, 3)
        self.assertTrue(second.auth_diagnostics[-1]['verified'])
        self.assertEqual(second.session.get.call_count, 5)

    def test_three_bad_replies_exhaust_budget_and_keep_safe_response_history(self):
        sessions = [flow(malformed(), str(i)) for i in range(3)]
        with self.assertRaises(requests.exceptions.JSONDecodeError) as raised:
            self.run_login(sessions)
        audit = raised.exception.auth_failure_diagnostics
        self.assertEqual(audit['auth_attempts'], 3)
        self.assertEqual(len(audit['attempt_failures']), 3)
        for vpn in sessions:
            vpn.session.close.assert_called_once()
            self.assertEqual(vpn.session.post.call_count, 2)
            self.assertEqual(vpn.session.get.call_count, 3)
        self.assertEqual(audit['response'], {'http_status':200,'body_kind':'empty',
            'challenge_hint':False,'redirected':False})
        self.assertNotIn('private', json.dumps(audit))
        self.assertNotIn('context', json.dumps(audit))
        self.assertEqual(self.sleep.call_count, 2)

    def test_challenges_unknown_html_rejections_and_redirects_do_not_retry(self):
        replies = [malformed('<html>captcha private</html>', 503),
                   malformed('<html>短信验证码 private</html>', 200),
                   malformed('{"needOtp":true,"private":', 200),
                   malformed('<html>two-factor authentication</html>', 503),
                   malformed('<html>private login page</html>', 200),
                   malformed('private-password-rejected', 200), malformed('',401),
                   malformed('',200,redirected=True),
                   response({'code':401,'needVerifyCode':True,'msg':'private-password'})]
        for reply in replies:
            with self.subTest(reply=type(reply).__name__):
                vpn = flow(reply, 'context')
                with self.assertRaises(Exception) as raised: self.run_login([vpn])
                self.assertEqual(self.factory.call_count, 1)
                self.sleep.assert_not_called()
                vpn.session.close.assert_called_once()
                self.assertNotIn('private',json.dumps(raised.exception.auth_failure_diagnostics))

    def test_malformed_json_and_gateway_html_can_recover(self):
        for reply in (malformed('{"code":'), malformed('<html>Bad Gateway</html>',502)):
            with self.subTest(reply=reply.status_code):
                first, second = flow(reply,'first'), flow(response({'code':200,'loginToken':'token'}),'second')
                self.assertIs(self.run_login([first,second]),second)
                self.assertEqual(self.factory.call_count,2)

    def test_single_attempt_media_auth_keeps_its_existing_budget(self):
        vpn = flow(malformed(),'context')
        with self.assertRaises(requests.exceptions.JSONDecodeError): self.run_login([vpn],max_attempts=1)
        self.assertEqual(self.factory.call_count,1)
        self.sleep.assert_not_called()

    def test_original_json_error_and_whitelisted_diagnostics_survive(self):
        vpn = flow(malformed('private response'),'context')
        with self.assertRaises(requests.exceptions.JSONDecodeError) as raised:
            self.run_login([vpn])
        error = raised.exception
        error.auth_failure_diagnostics['response'].update(secret='private',http_status=True)
        error.auth_failure_diagnostics['attempt_failures'][0]['response']['body_kind']='private'
        audit = authentication_failure(error)
        self.assertEqual(audit['failure_phase'],'icourse_auth_execute')
        self.assertEqual(audit['failure'],'auth_invalid_response')
        self.assertNotIn('private',json.dumps(audit))
        self.assertNotIn('http_status',audit['response'])

    def test_each_json_auth_step_preserves_response_shape(self):
        for method, indices in (('get',(2,)),('post',(0,1))):
            for index in indices:
                candidate = fixtures.AuthDiagnosticTests().session()
                candidate.probe_login_service = MagicMock();candidate.login = MagicMock()
                sequence = list(getattr(candidate.session,method).side_effect)
                sequence[index] = malformed()
                getattr(candidate.session,method).side_effect = sequence
                with self.subTest(method=method,index=index), self.assertRaises(requests.exceptions.JSONDecodeError) as raised:
                    self.run_login([candidate],max_attempts=1)
                self.assertEqual(raised.exception.auth_failure_diagnostics['response']['body_kind'],'empty')
        for method, args, phase in (
                ('_query_auth_methods', ('synthetic', 'synthetic'), 'webvpn_auth_methods'),
                ('_get_public_key', (), 'webvpn_public_key'),
                ('_auth_execute', ('synthetic',)*6, 'webvpn_auth_execute')):
            vpn = fixtures.WebVPNSession();vpn.session.close();vpn.session = MagicMock()
            vpn.session.get.return_value = malformed();vpn.session.post.return_value = malformed()
            with self.subTest(phase=phase), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(requests.exceptions.JSONDecodeError) as raised:
                    getattr(vpn, method)(*args)
            audit = authentication_failure(raised.exception)
            self.assertEqual(audit['failure_phase'], phase)
            self.assertEqual(audit['response']['body_kind'], 'empty')

    def test_real_http_gateway_reply_uses_fresh_post_context_and_recovers(self):
        contexts=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                contexts.append(payload['lck'])
                body = b'<html>Bad Gateway</html>' if len(contexts)==1 else b'{"code":200,"loginToken":"synthetic-token"}'
                self.send_response(503 if len(contexts)==1 else 200)
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            sessions=[]
            for context in ('first-context','second-context'):
                vpn=flow(None,context)
                replies=[response({'data':[{'moduleCode':'userAndPwd','authChainCode':'chain'}]}),
                    response(text='locationValue="https://private/cas?ticket=synthetic-ticket"')]
                def post(url, _replies=replies, **kwargs):
                    if '/authExecute' in url:
                        return requests.post(f'http://127.0.0.1:{server.server_port}/authExecute',json=kwargs['json'],timeout=5)
                    return _replies.pop(0)
                vpn.session.post.side_effect=post;sessions.append(vpn)
            self.assertIs(self.run_login(sessions),sessions[1])
            self.assertEqual(contexts,['first-context','second-context'])
            self.assertTrue(sessions[1].auth_diagnostics[-1]['verified'])
        finally:
            server.shutdown();server.server_close();thread.join(2)


if __name__=='__main__': unittest.main()
