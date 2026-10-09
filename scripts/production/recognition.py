"""Formal recognition stage, with explicit runtime services and frozen inputs."""
from __future__ import annotations
import os
import shutil


def shared_local_checkpoints(runtime, plan):
    """Read every prior worker before any replacement worker may claim blocks."""
    saved = {}
    for worker_id in range(len(plan['shards'])):
        destination = runtime.root()/'shared-recovery'/str(worker_id)
        if runtime.artifact(f'qwen-production-shared-{plan["course_slot"]}-{worker_id}',
                    destination, run=str(plan['run_id'])):
            with runtime.shards.environment({'GITHUB_RUN_ID': str(plan['run_id']),
                                     'COURSE_SLOT': str(plan['course_slot'])}):
                local = runtime.read_json(runtime.decode(destination/'shared-local.enc', f'shared-local-{worker_id}')['local.json'])
            if local['plan_hash'] != runtime.fingerprint(plan) or local['worker_id'] != worker_id:
                raise ValueError('Dynamic local checkpoint belongs to another input')
            saved[worker_id] = local['chunks']
    return saved


def shared_results(runtime, plan, *, require_complete=True):
    """Read-only complete original blocks for gather and private export."""
    from scripts.shared_asr_worker import store_for
    from src.pipeline.asr_queue import SharedQueue
    store = store_for(plan)
    try:
        return SharedQueue(plan, store).results(require_complete=require_complete)
    finally:
        store.close()


def worker(runtime):
    files = runtime.read_preparation(); spec = runtime.read_json(files['specification.json'])
    if int(os.environ['SHARD_ID']) == -1:
        if spec['mode'] == 'failed': raise ValueError('Preparation failed')
        return
    plan = spec['plan']; runtime.validate_plan(plan)
    if plan.get('execution') == 'shared_queue':
        from scripts.shared_asr_worker import run_worker, store_for
        from src.pipeline.asr_queue import SharedQueue
        worker_id = int(os.environ['SHARD_ID'])
        attempt = int(os.environ.get('GITHUB_RUN_ATTEMPT', '1'))
        previous = runtime.shared_local_checkpoints(plan) if attempt > 1 else {}
        store = store_for(plan)
        try:
            SharedQueue(plan, store).restore_rows([row for rows in previous.values() for row in rows], attempt)
            report = run_worker(plan, files, store, worker_id, attempt,
                                previous_rows=previous.get(worker_id, []))
        finally:
            store.close()
        runtime.out('worker-audit.json').write_bytes(runtime.shards.encoded(report))
        return
    shard_id = int(os.environ['SHARD_ID'])
    assigned = plan['shards'][shard_id]['chunk_ids']
    audio = {'manifest.json': runtime.shards.encoded(plan)}
    audio.update({f'chunk-{i}.flac': files[f'chunk-{i}.flac'] for i in assigned})
    if f'completed-{shard_id}.json' in files:
        audio['completed.json'] = files[f'completed-{shard_id}.json']
    runtime.encode(audio, f'input-{shard_id}', runtime.root()/'inbox'/f'shard-{shard_id}.enc')
    if runtime.artifact(f'qwen-production-asr-{os.environ["COURSE_SLOT"]}-{shard_id}', runtime.root()/'previous'):
        shutil.copyfile(runtime.root()/'previous'/'worker-result.enc', runtime.out('worker-result.enc'))
    runtime.shards.worker()

