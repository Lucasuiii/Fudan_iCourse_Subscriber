"""Execute exactly one journal-authorized stage against the frozen parent input.

Actions retains its real child run ID for artifact upload. Only the pipeline
subprocess receives parent run/attempt encryption and recovery identities.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import time
from scripts import production_pool as pool


def authorize(parent, nonce, actual_run, *, read, inspect, sleep=time.sleep):
    if not re.fullmatch('[0-9]+', parent) or not re.fullmatch('[0-9a-f]{32}', nonce):
        raise ValueError('Invalid stage request')
    info = inspect(actual_run)
    if info['run_attempt'] != 1:
        raise ValueError('Rerun the parent workflow, not a single pool child')
    parent_info = inspect(parent)
    if parent_info['path'].split('@')[0] not in ('.github/workflows/check.yml', '.github/workflows/parallel_pilot.yml'):
        raise ValueError('Unauthorized parent workflow')
    for _ in range(12):
        state = read(); pool.validate(state)
        if state['run_id'] != parent or state['sha'] != info['head_sha'] or state['sha'] != parent_info['head_sha']:
            raise ValueError('Frozen stage source mismatch')
        if info['path'].split('@')[0] != '.github/workflows/'+pool.WORKFLOW or info['display_title'] != f'icourse-stage-{parent}-{nonce}':
            raise ValueError('Stage run identity mismatch')
        ticket = next((t for t in state['tickets'] if t['nonce'] == nonce), None)
        if ticket is None: raise ValueError('Stage has no durable reservation')
        if ticket.get('run'):
            if ticket['run'] != actual_run or ticket['status'] == 'completed':
                raise ValueError('Stage reservation belongs to another or ended run')
            if ticket['attempt'] != parent_info['run_attempt']:
                raise ValueError('Parent attempt changed while a child was active')
            return state, ticket
        sleep(10)  # Wait for the controller to register this exact run; no model calls.
    raise TimeoutError('Stage identity was not confirmed; reservation remains charged')


def context():
    parent = os.environ['POOL_PARENT_RUN_ID']; nonce = os.environ['POOL_TICKET']
    actual = os.environ['GITHUB_RUN_ID']
    store = pool.store_for(parent)
    try:
        state, ticket = authorize(parent, nonce, actual, read=lambda: pool.read_state(store)[1],
            inspect=lambda run: pool.api(f'repos/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{run}'))
    finally: store.close()
    env = dict(os.environ, GITHUB_RUN_ID=parent, GITHUB_RUN_ATTEMPT=str(ticket['attempt']),
               COURSE_SLOT=str(ticket['slot']), SHARD_ID=str(ticket['worker']), SHARD_MODE='shared',
               POOL_ARTIFACTS='true', **state['flags'])
    return ticket, env


def bootstrap():
    ticket, env = context()
    # No credentials or private content are written to GITHUB_ENV.
    with open(os.environ['GITHUB_ENV'], 'a') as target:
        target.write('POOL_STAGE='+ticket['stage']+'\n')
        target.write('COURSE_SLOT='+str(ticket['slot'])+'\n')
        target.write('SHARD_ID='+str(ticket['worker'])+'\n')
    with open(os.environ['GITHUB_OUTPUT'], 'a') as target:
        target.write('stage='+ticket['stage']+'\n')
    print('Reserved stage identity verified')


def execute():
    ticket, env = context()
    from scripts import production_qwen as pipeline
    from scripts.sharded_qwen_pilot import environment
    mode = ticket['stage']; began = time.monotonic()
    with environment(env):
        if mode == 'prepare':
            pipeline.artifact('qwen-production-queue', pipeline.root()/'inbox', required=True)
        elif mode in ('asr', 'gather'):
            name = 'worker-input' if mode == 'asr' else 'prepare'
            pipeline.artifact(f'qwen-production-{name}-{ticket["slot"]}', pipeline.root()/'inbox', required=True)
        else:
            pipeline.artifact(f'qwen-production-state-{ticket["slot"]}', pipeline.root()/'inbox', required=True)
        download_seconds = time.monotonic()-began
        began = time.monotonic()
        command = {'prepare': 'prepare', 'asr': 'worker', 'gather': 'gather', 'publish': 'publish'}[mode]
        result = subprocess.run([sys.executable, '-m', 'scripts.production_qwen', command], env=env)
        pipeline.out('stage-audit.json').write_text(json.dumps({'stage': mode, 'task_slot': ticket['slot'],
            'worker_id': ticket['worker'], 'parent_attempt': ticket['attempt'], 'input_download_seconds': download_seconds, 'pipeline_seconds': time.monotonic()-began,
            'return_code': result.returncode}))
    if result.returncode: raise SystemExit(result.returncode)


def main(command):
    try:
        {'bootstrap': bootstrap, 'execute': execute}[command]()
    except Exception as error:
        # Authorization/API/artifact failures occur before production_qwen's
        # own exception boundary. Keep their fixed diagnostics as well.
        from scripts import production_qwen as pipeline
        from scripts.coordination_transport import diagnostic
        audit = {'entry': command if command in ('bootstrap', 'execute') else 'unknown',
                 'error_code': pipeline.failure_code(error)}
        if diagnostic(error): audit['coordination'] = diagnostic(error)
        try: pipeline.out('stage-failure.json').write_bytes(pipeline.shards.encoded(audit))
        except Exception: pass  # Diagnostic writes must preserve the primary error.
        raise


if __name__ == '__main__':
    try: main(sys.argv[1])
    except Exception as error:
        print(f'Reserved stage failed ({type(error).__name__}); private details withheld')
        raise SystemExit(1)
