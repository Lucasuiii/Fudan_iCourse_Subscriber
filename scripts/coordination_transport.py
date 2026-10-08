"""Bounded retries and fixed diagnostics for coordination reads.

Writes are reconciled by their caller, never blindly replayed here. Exceptions
keep private subprocess output internally; public reports contain only codes.
"""
import json
import re
import subprocess
import time


def transport_code(error):
    if isinstance(error, json.JSONDecodeError):
        return 'invalid_response'
    if isinstance(error, subprocess.TimeoutExpired):
        return 'timeout'
    if not isinstance(error, subprocess.CalledProcessError):
        return 'unknown'
    raw = error.stderr or b''
    value = (raw.decode(errors='replace') if isinstance(raw, bytes) else raw).lower()
    if re.search(r'http(?:\s|[^\w])*401\b|http(?:\s|[^\w])*403\b', value) or any(
            s in value for s in ('authentication failed', 'permission denied', 'could not read username')):
        return 'authorization'
    if any(s in value for s in ('certificate', 'ssl', 'tls')):
        return 'tls'
    if re.search(r'(?:http|error:)[^\n]{0,60}\b(?:408|429|500|502|503|504)\b', value):
        return 'service_unavailable'
    if any(s in value for s in ('connection reset', 'could not resolve host', 'failed to connect',
                                'timed out', 'timeout', 'remote end hung up', 'bad gateway')):
        return 'connection'
    return 'command_failed'


RETRYABLE = frozenset(('timeout', 'service_unavailable', 'connection', 'invalid_response'))
OPERATIONS = frozenset(('git_ls_remote', 'git_fetch', 'git_show', 'git_push', 'github_read', 'github_dispatch'))


class CoordinationError(RuntimeError):
    def __init__(self, operation, code, attempts):
        if operation not in OPERATIONS or code not in RETRYABLE | {'authorization', 'tls', 'command_failed', 'invalid_response'}:
            raise ValueError('Invalid coordination diagnostic')
        self.operation, self.code, self.attempts = operation, code, attempts
        super().__init__('Coordination operation failed; private response withheld')


def read_with_retry(operation, read):
    for attempt in range(1, 4):
        try:
            return read()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            code = transport_code(error)
            if code not in RETRYABLE or attempt == 3:
                raise CoordinationError(operation, code, attempt) from error
            time.sleep(2 * attempt)


def read_json(read):
    return read_with_retry('github_read', lambda: json.loads(read()))


def diagnostic(error):
    if isinstance(error, CoordinationError):
        return {'operation': error.operation, 'failure': error.code, 'attempts': error.attempts}
    return None
