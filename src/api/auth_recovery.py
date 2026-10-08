"""Bounded fresh-session authentication; never replay tickets or retain credentials."""
import time

import requests

from src.api.webvpn import AuthenticationError, WebVPNSession, authentication_failure


RETRYABLE_AUTH_REASONS = frozenset(('service_unavailable', 'cold_session',
                                  'cas_context_missing', 'api_verification_failed'))


def authenticated_session(*, max_attempts=3, student_id=None, password=None,
                          factory=WebVPNSession, sleep=time.sleep, probe_attempts=1):
    if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
        raise ValueError('Invalid bounded authentication attempts')
    if type(probe_attempts) is not int or not 1 <= probe_attempts <= 3:
        raise ValueError('Invalid bounded preflight attempts')
    history = []
    for attempt in range(max_attempts):
        vpn = factory()
        try:
            # This retry is only a credential-free portal probe. It never
            # repeats password submission or follows a consumed CAS ticket.
            for probe in range(probe_attempts):
                vpn.auth_probe_attempts = probe+1
                vpn.auth_probe_transient_failures = probe
                try:
                    vpn.probe_login_service()
                    break
                except Exception as error:
                    temporary = (isinstance(error, AuthenticationError)
                                 and error.reason == 'service_unavailable'
                                 or isinstance(error, (requests.exceptions.Timeout,
                                                      requests.exceptions.ConnectionError))
                                 and not isinstance(error, requests.exceptions.SSLError))
                    if temporary: vpn.auth_probe_transient_failures = probe+1
                    if not temporary or probe == probe_attempts-1: raise
                    session = vpn.session
                    if isinstance(session, DeadlineSession):
                        remaining = session.deadline-time.monotonic()
                        if remaining <= 0 or session.cancelled.wait(min(1,remaining)):
                            raise AuthenticationError('media_auth_cancelled')
                    else: sleep(1)
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
            error.auth_failure_diagnostics = authentication_failure(error, vpn=vpn)
            history.append({k: v for k, v in error.auth_failure_diagnostics.items()
                            if k not in ('auth_attempts', 'attempt_failures')})
            error.auth_failure_diagnostics.update(auth_attempts=attempt+1,
                                                  attempt_failures=list(history))
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


class DeadlineSession(requests.Session):
    """Each auth request/redirect checks the same cancellation and wall deadline."""
    def __init__(self, cancelled, deadline):
        super().__init__()
        self.cancelled, self.deadline = cancelled, deadline

    def send(self, request, **kwargs):
        remaining = self.deadline-time.monotonic()
        if self.cancelled.is_set() or remaining <= 0:
            raise AuthenticationError('media_auth_cancelled')
        requested = kwargs.get('timeout') or (10, 10)
        if not isinstance(requested, tuple): requested = (requested, requested)
        kwargs['timeout'] = tuple(min(float(v or cap), cap, remaining)
                                  for v, cap in zip(requested, (10, 30)))
        return super().send(request, **kwargs)


def fresh_media_session(cancelled, deadline):
    """One new ticket flow from configured credentials, never a media Location."""
    def factory():
        vpn = WebVPNSession()
        vpn.session.close()
        vpn.session = DeadlineSession(cancelled, deadline)
        from src.runtime import config
        vpn.session.headers.update({'User-Agent': config.USER_AGENT})
        return vpn
    # One fresh login, with one bounded retry of the no-credential preflight
    # inside the existing 75-second deadline; no retry after login begins.
    return authenticated_session(max_attempts=1, factory=factory, probe_attempts=2)
