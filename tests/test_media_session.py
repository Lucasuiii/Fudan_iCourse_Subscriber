"""Real loopback HTTP cookie rotation and bounded SSO refresh; no live login."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit

import requests
from src.api.icourse import ICourseClient
from src.api.webvpn import get_vpn_url
from src.runtime.media_transport import SignedRangeRelay, MediaTransportError


class SessionOrigin:
    DATA = bytes(range(256))*256

    def __init__(self, mode='rotate'):
        self.mode=mode; self.token='first'; self.calls=[]; self.cookies=[]

    def __enter__(self):
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def reply(self,status,body=b'',headers=()):
                self.send_response(status)
                self.send_header('Content-Length',str(len(body)))
                for key,value in headers: self.send_header(key,value)
                self.end_headers()
                try: self.wfile.write(body)
                except OSError: pass
            def do_GET(self):
                path=urlsplit(self.path).path
                owner.calls.append(path)
                if path == '/':
                    if owner.mode == 'cold':
                        self.reply(302,headers=[('Location','/login?private-ticket')]);return
                    owner.token='refreshed'
                    self.reply(200,headers=[('Set-Cookie','media_token=refreshed; Path=/')]);return
                if path == '/api':
                    self.reply(200,b'{"code":0,"params":{"id":"cached-private-user","tenant_id":"fake","phone":"fake"}}');return
                if path != '/media':
                    self.reply(500);return
                cookie=self.headers.get('Cookie','')
                owner.cookies.append(cookie)
                match=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers.get('Range',''))
                if not match: self.reply(400);return
                start=int(match[1]);end=min(int(match[2]) if match[2] else len(owner.DATA)-1,len(owner.DATA)-1)
                if start and owner.mode in ('foreign','same_media','wrapped_foreign'):
                    if owner.mode=='foreign':location='https://untrusted.invalid/login?private-ticket'
                    elif owner.mode=='wrapped_foreign':location=get_vpn_url('https://untrusted.invalid/cas/login')+'?private-ticket'
                    else:location='/media?t=untrusted-ticket'
                    self.reply(302,b'private-login-html',[('Location',location)]);return
                if start and owner.mode in ('login','cold','persistent','changed','wrapped_login'):
                    if owner.token!='refreshed' or owner.mode=='persistent':
                        location = (get_vpn_url('https://id.fudan.edu.cn/idp/authCenter/authenticate')
                                    if owner.mode=='wrapped_login' else '/login')
                        self.reply(302,b'private-login-html',[('Location',location+'?private-ticket')]);return
                expected='' if owner.token is None else 'media_token='+owner.token
                if cookie!=expected:
                    self.reply(401,b'private-login-html');return
                headers=[('Content-Range',f'bytes {start}-{end}/{len(owner.DATA)}'),
                         ('ETag','"changed"' if start and owner.mode=='changed' else '"immutable"')]
                if start==0 and owner.mode=='rotate':
                    owner.token='second';headers.append(('Set-Cookie','media_token=second; Path=/'))
                elif start==0 and owner.mode=='delete':
                    owner.token=None;headers.append(('Set-Cookie','media_token=; Max-Age=0; Path=/'))
                self.reply(206,owner.DATA[start:end+1],headers)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.base=f'http://127.0.0.1:{self.server.server_port}'
        self.signed=self.base+'/media?t=old&clientUUID=old'
        owner=self
        class VPN:
            def __init__(self):
                self.session=requests.Session()
                self.session.cookies.set('media_token','first',domain='127.0.0.1',path='/')
                self.session.cookies.set('foreign','private-cookie',domain='untrusted.invalid',path='/')
                self.session.cookies.set('other_path','private-cookie',domain='127.0.0.1',path='/not-media')
                self.session.cookies.set('secure','private-cookie',domain='127.0.0.1',path='/',secure=True)
            def get(self,url,**kwargs): return self.session.get(owner.base+'/api',**kwargs)
        class Client(ICourseClient):
            counter=0
            def renew_video_url(self,url,now=None):
                self.counter+=1
                return owner.base+f'/media?t=fresh{self.counter}&clientUUID={self.counter}'
            def get_stream_params(self,url):
                # The old flattened header must be ignored when a real login
                # jar exists; path/domain/secure rules must select the cookies.
                value = self.vpn.session.cookies.get('media_token')
                return url,'Cookie: '+('media_token='+value if value else '')+'\r\n'
        self.client=Client(VPN());self.client._userinfo={'id':'cached-private-user'}
        self.config=patch('src.api.icourse.config.WEBVPN_BASE',self.base)
        self.config.start()
        return self

    def __exit__(self,*args):
        self.config.stop();self.client.vpn.session.close()
        self.server.shutdown();self.server.server_close();self.thread.join(2)


class MediaSessionTests(unittest.TestCase):
    def relay(self,origin,**kwargs):
        return SignedRangeRelay(origin.client,origin.signed,prefix_bytes=4096,
                                chunk_bytes=16384,**kwargs)

    def test_rotated_cookie_survives_ranges_and_reaches_the_login_session(self):
        with SessionOrigin() as origin, self.relay(origin) as relay:
            session=relay._session
            self.assertEqual(requests.get(relay.url,timeout=15).content,origin.DATA)
            self.assertIs(relay._session,session)
            self.assertEqual(origin.cookies[0],'media_token=first')
            self.assertTrue(all(c=='media_token=second' for c in origin.cookies[1:]))
            self.assertEqual(origin.client.vpn.session.cookies.get('media_token'),'second')
            audit=relay.audit();self.assertEqual(audit['cookie_updates'],1)
            self.assertEqual(audit['session_refresh_attempts'],0)
            self.assertNotIn('private',json.dumps(audit))

    def test_cookie_deletion_is_not_resurrected_by_the_old_header(self):
        with SessionOrigin('delete') as origin, self.relay(origin) as relay:
            self.assertEqual(requests.get(relay.url,timeout=15).content,origin.DATA)
            self.assertTrue(all(c=='' for c in origin.cookies[1:]))
            self.assertIsNone(origin.client.vpn.session.cookies.get('media_token'))
            self.assertEqual(relay.audit()['cookie_updates'],1)

    def test_known_login_redirect_refreshes_existing_cookies_once_and_resumes(self):
        with SessionOrigin('login') as origin, self.relay(origin,allow_session_refresh=True) as relay:
            self.assertEqual(requests.get(relay.url,timeout=15).content,origin.DATA)
            self.assertEqual(origin.calls.count('/'),1);self.assertEqual(origin.calls.count('/api'),1)
            self.assertNotIn('/login',origin.calls)
            self.assertEqual(origin.client._userinfo['id'],'cached-private-user')
            audit=relay.audit()
            self.assertEqual(audit['session_refresh_attempts'],1)
            self.assertEqual(audit['session_refresh_successes'],1)
            self.assertEqual(audit['redirect_counts'],{'login':1})
            self.assertEqual(audit['last_failure_offset'],4096)
            self.assertEqual(audit['upstream_bytes'],len(origin.DATA))
            self.assertNotIn('private',json.dumps(audit))
            audit['redirect_counts']['login']=999
            self.assertEqual(relay.audit()['redirect_counts'],{'login':1})

    def test_default_readonly_transport_never_refreshes_authentication(self):
        with SessionOrigin('login') as origin, self.relay(origin) as relay:
            with self.assertRaises(requests.RequestException): requests.get(relay.url,timeout=15)
            self.assertNotIn('/',origin.calls);self.assertNotIn('/api',origin.calls)
            self.assertEqual(relay.audit()['terminal_error_code'],'media_session_unavailable')

    def test_exact_wrapped_sso_route_refreshes_once_without_following_ticket(self):
        with SessionOrigin('wrapped_login') as origin, self.relay(origin,allow_session_refresh=True) as relay:
            self.assertEqual(requests.get(relay.url,timeout=15).content,origin.DATA)
            self.assertEqual(origin.calls.count('/'),1)
            self.assertEqual(origin.calls.count('/api'),1)
            audit = relay.audit()
            self.assertEqual(audit['session_refresh_attempts'],1)
            self.assertEqual(audit['last_redirect']['classification'],'login')
            self.assertEqual(audit['last_redirect']['route'],'vpn_wrapped')
            self.assertNotIn('private',json.dumps(audit))
            audit['last_redirect']['route']='modified'
            self.assertEqual(relay.audit()['last_redirect']['route'],'vpn_wrapped')

    def test_wrapped_foreign_login_suffix_cannot_trigger_refresh(self):
        with SessionOrigin('wrapped_foreign') as origin, self.relay(origin,allow_session_refresh=True) as relay:
            with self.assertRaises(requests.RequestException): requests.get(relay.url,timeout=15)
            audit = relay.audit()
            self.assertEqual(audit['terminal_error_code'],'media_redirect_untrusted')
            self.assertEqual(audit['last_redirect']['classification'],'other')
            self.assertEqual(audit['last_redirect']['route'],'vpn_wrapped')
            self.assertEqual(audit['session_refresh_attempts'],0)
            self.assertNotIn('/',origin.calls)
            self.assertNotIn('untrusted.invalid',json.dumps(audit))

    def test_cold_or_repeated_login_redirect_does_not_loop_or_forward_html(self):
        for mode in ('cold','persistent'):
            with self.subTest(mode=mode), SessionOrigin(mode) as origin, self.relay(origin,allow_session_refresh=True) as relay:
                with self.assertRaises(requests.RequestException): requests.get(relay.url,timeout=15)
                self.assertEqual(origin.calls.count('/'),1)
                self.assertNotIn('/login',origin.calls)
                self.assertEqual(relay.audit()['upstream_bytes'],4096)
                self.assertEqual(relay.audit()['terminal_error_code'],'media_session_unavailable')
                self.assertEqual(relay.audit()['session_refresh_attempts'],1)

    def test_unknown_redirect_stops_without_following_or_refreshing(self):
        with SessionOrigin('foreign') as origin, self.relay(origin,allow_session_refresh=True) as relay:
            with self.assertRaises(requests.RequestException): requests.get(relay.url,timeout=15)
            audit=relay.audit()
            self.assertEqual(audit['terminal_error_code'],'media_redirect_untrusted')
            self.assertEqual(audit['redirect_counts'],{'other':1})
            self.assertEqual(audit['upstream_bytes'],4096)
            self.assertEqual(audit['session_refresh_attempts'],0)
            self.assertNotIn('private',json.dumps(audit));self.assertNotIn('untrusted.invalid',json.dumps(audit))

    def test_redirect_classification_does_not_accept_downgrade_or_credentials(self):
        relay=SignedRangeRelay(MagicMock(),'https://vpn.example/media?t=old&clientUUID=old')
        for location,expected in [('/login?ticket=private','login'),
                ('/wengine-vpn/login?ticket=private','login'),
                ('/media?t=private','same_media'),
                ('http://vpn.example/login','other'),
                ('https://private-user:private-pass@vpn.example/login','other'),
                ('https://untrusted.invalid/login','other'),
                ('https://id.fudan.edu.cn/cas/login?ticket=private','login'),
                ('https://id.fudan.edu.cn/other','other'),('', 'missing')]:
            response=SimpleNamespace(url='https://vpn.example/media',headers={'Location':location})
            self.assertEqual(relay._redirect_kind(response),expected)

    def test_same_media_redirect_remains_bounded_without_following_new_signature(self):
        with SessionOrigin('same_media') as origin, self.relay(origin,attempts=2) as relay:
            with self.assertRaises(requests.RequestException): requests.get(relay.url,timeout=15)
            self.assertEqual(relay.audit()['redirect_counts'],{'same_media':2})
            self.assertEqual(relay.audit()['terminal_error_code'],'range_not_honored')
            self.assertEqual(relay.audit()['upstream_bytes'],4096)

    def test_changed_source_after_session_refresh_is_not_spliced(self):
        with SessionOrigin('changed') as origin, self.relay(origin,allow_session_refresh=True) as relay:
            with self.assertRaises(requests.RequestException): requests.get(relay.url,timeout=15)
            self.assertEqual(relay.audit()['terminal_error_code'],'source_changed')
            self.assertEqual(relay.audit()['upstream_bytes'],4096)

    def test_refresh_requires_both_portal_and_icourse_verification(self):
        valid={'code':0,'params':{'id':'private-user','tenant_id':'fake','phone':'fake'}}
        for portal,api,expected in [(302,valid,False),(200,{'code':7001},False),
                (200,{'code':0},False),(200,{'code':0,'params':{'id':'another-user'}},False),
                (200,valid,True)]:
            vpn=MagicMock();client=ICourseClient(vpn);client._userinfo={'id':'private-user'}
            vpn.session.get.return_value=SimpleNamespace(status_code=portal)
            vpn.get.return_value=SimpleNamespace(status_code=200,json=lambda:api)
            success=client.refresh_media_session()
            self.assertEqual(success,expected)
            self.assertFalse(vpn.session.get.call_args.kwargs['allow_redirects'])
            if portal==200:self.assertFalse(vpn.get.call_args.kwargs['allow_redirects'])
            else:vpn.get.assert_not_called()
            vpn.login.assert_not_called();vpn.authenticate_icourse.assert_not_called()
            self.assertEqual(client._userinfo,valid['params'] if success else {'id':'private-user'})


if __name__=='__main__': unittest.main()
