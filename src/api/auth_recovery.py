"""Bounded fresh-session authentication; never replay tickets or retain credentials."""
import time

import requests

from src.api.webvpn import AuthenticationError, WebVPNSession


RETRYABLE_AUTH_REASONS = frozenset(('service_unavailable', 'cold_session',
                                  'cas_context_missing', 'api_verification_failed'))


def authenticated_session(*, max_attempts=3, student_id=None, password=None,
                          factory=WebVPNSession, sleep=time.sleep):
    if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
        raise ValueError('Invalid bounded authentication attempts')
    for attempt in range(max_attempts):
        vpn = factory()
        try:
            vpn.probe_login_service()
            if student_id is None and password is None:
                if getattr(vpn, 'requires_webvpn_login', True): vpn.login()
                verified = vpn.authenticate_icourse(strict=True)
            else:
                if getattr(vpn, 'requires_webvpn_login', True): vpn.login(student_id, password)
                verified = vpn.authenticate_icourse(student_id, password, strict=True)
            if verified is not True:
                raise AuthenticationError('api_verification_failed')
            return vpn
        except Exception as error:
            vpn.session.close()
            retryable = (isinstance(error, AuthenticationError)
                         and error.reason in RETRYABLE_AUTH_REASONS
                         or isinstance(error, (requests.exceptions.Timeout,
                                               requests.exceptions.ConnectionError))
                         and not isinstance(error, requests.exceptions.SSLError))
            if not retryable or attempt == max_attempts-1:
                raise
            # The next attempt gets a new Session and a new one-use ticket.
            # Outage probes run before credentials are submitted.
            sleep(min(5*(attempt+1), 10))
