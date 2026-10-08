"""Encrypted Git fast-forward CAS, isolated from main/data and REST quotas.

Each candidate commit has exactly the read commit as its parent. A competing
push is divergent and rejected atomically. No force pushes, audio or secrets in
Git: only encrypted block claims/results on run-specific coordination refs.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from scripts.coordination_transport import CoordinationError, RETRYABLE, read_with_retry, transport_code
import time

MAX_STATE = 2 * 1024 * 1024


class GitHubQueueStore:
    def __init__(self, run_id, slot, key, *, remote_url=None):
        repo = os.environ['GITHUB_REPOSITORY']
        if (not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo)
                or not str(run_id).isdigit() or type(slot) is not int or not 0 <= slot < 256
                or len(key) != 32):
            raise ValueError('Invalid queue storage identity')
        self.ref = f'refs/heads/codex/asr-queue-{run_id}-{slot}'
        self.remote = remote_url or 'https://github.com/'+repo+'.git'
        self.aad = f'icourse-asr-queue-v1:{repo}:{run_id}:{slot}'.encode()
        self.key = key
        self.temp = tempfile.TemporaryDirectory(prefix='icourse-queue-git-')
        self.path = Path(self.temp.name)
        from scripts.production_db import auth_env
        self.env = auth_env()
        self.env.update(GIT_AUTHOR_NAME='github-actions[bot]', GIT_COMMITTER_NAME='github-actions[bot]',
                        GIT_AUTHOR_EMAIL='41898282+github-actions[bot]@users.noreply.github.com',
                        GIT_COMMITTER_EMAIL='41898282+github-actions[bot]@users.noreply.github.com')
        self.command(['git', 'init', '-q'])

    def command(self, args, data=None):
        return subprocess.run(args, cwd=self.path, env=self.env, input=data, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120).stdout

    def close(self):
        self.temp.cleanup()

    def head(self):
        rows = read_with_retry('git_ls_remote', lambda: self.command(
            ['git', 'ls-remote', '--heads', self.remote, self.ref])).decode().splitlines()
        if not rows:
            return None  # Confirmed absence only; transport errors raise.
        if len(rows) != 1 or rows[0].split()[1] != self.ref:
            raise ValueError('Unexpected queue reference')
        return rows[0].split()[0]

    def seal(self, state):
        raw = json.dumps(state, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        if len(raw) > MAX_STATE:
            raise ValueError('Queue checkpoint too large')
        nonce = os.urandom(12)
        return b'AQ01'+nonce+AESGCM(self.key).encrypt(nonce, raw, self.aad)

    def unseal(self, data):
        if not 32 <= len(data) <= MAX_STATE+32 or data[:4] != b'AQ01':
            raise ValueError('Invalid encrypted queue checkpoint')
        return json.loads(AESGCM(self.key).decrypt(data[4:16], data[16:], self.aad))

    def read(self):
        head = self.head()
        if head is None:
            return None, None
        read_with_retry('git_fetch', lambda: self.command(['git', 'fetch', '-q', '--depth=1', self.remote, head]))
        paths = self.command(['git', 'ls-tree', '-r', '--name-only', head]).decode().splitlines()
        if paths != ['queue.enc']:
            raise ValueError('Unexpected queue tree')
        return head, self.unseal(self.command(['git', 'show', head+':queue.enc']))

    def compare_and_swap(self, previous, state):
        if previous and not re.fullmatch(r'[0-9a-f]{40}', previous):
            raise ValueError('Invalid queue revision')
        blob = self.command(['git', 'hash-object', '-w', '--stdin'], self.seal(state)).decode().strip()
        tree = self.command(['git', 'mktree'], f'100644 blob {blob}\tqueue.enc\n'.encode()).decode().strip()
        args = ['git', 'commit-tree', tree]
        if previous:
            args += ['-p', previous]
        commit = self.command(args, b'Encrypted ASR queue checkpoint\n').decode().strip()
        for attempt in range(1, 4):
            try:
                self.command(['git', 'push', '-q', self.remote, commit+':'+self.ref])
                return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                # Reconcile even a timeout before replaying this exact commit.
                # If the read fails too, the write outcome remains unknown.
                current = self.head()
                if current == commit:
                    return True
                if current != previous:
                    return False
                code = transport_code(error)
                if code not in RETRYABLE or attempt == 3:
                    raise CoordinationError('git_push', code, attempt) from error
                time.sleep(2 * attempt)
