"""Dynamic original-block worker; completed blocks are committed immediately."""
import hashlib
import io
import os
import time
from scripts import sharded_qwen_pilot as shards
from scripts.asr_queue_store import GitHubQueueStore
from src.pipeline.asr_queue import SharedQueue, initial_queue, validate_queue
from scripts.qwen_sharding import fingerprint


def store_for(plan):
    return GitHubQueueStore(plan['run_id'], plan['course_slot'], shards.key())


def initialize(plan, files):
    import json
    store = store_for(plan)
    try:
        _, state = store.read()
        if state is None:
            if int(os.environ.get('GITHUB_RUN_ATTEMPT', '1')) > 1:
                raise ValueError('Lost shared queue checkpoint; cannot reset completed blocks')
            prior = [json.loads(files[f'completed-{s["shard_id"]}.json']) for s in plan['shards']
                     if f'completed-{s["shard_id"]}.json' in files]
            state = initial_queue(plan, prior)
            if not store.compare_and_swap(None, state):
                _, state = store.read()
        validate_queue(plan, state)
    finally:
        store.close()


def run_worker(plan, files, store, worker_id, attempt, *, transcriber=None, timeout=19500, previous_rows=()):
    """Dependencies injected for multi-worker/concurrent recovery verification."""
    import soundfile as sf
    from src.ai.qwen_transcriber import QwenTranscriber, QwenTokenBudgetError
    if type(worker_id) is not int or not 0 <= worker_id < len(plan['shards']):
        raise ValueError('Invalid dynamic worker slot')
    queue = SharedQueue(plan, store)
    began = time.monotonic()
    decoded = []
    local_rows = list(previous_rows)
    input_audio_bytes = sum(len(files[f'chunk-{b["chunk_id"]}.flac']) for b in plan['blocks']
                            if f'chunk-{b["chunk_id"]}.flac' in files)
    input_audio_blocks = sum(f'chunk-{b["chunk_id"]}.flac' in files for b in plan['blocks'])
    claimed_audio_bytes = 0
    claimed_audio_blocks = 0
    role = f'shared-local-{worker_id}'
    def save_local():
        shards.seal({'local.json': shards.encoded({'plan_hash': fingerprint(plan),
                    'worker_id': worker_id, 'chunks': local_rows})}, role, shards.root()/'out'/'shared-local.enc')
    save_local()
    phase = 'restore'; failure = None; cleanup_errors = []; active_block = None
    try:
        queue.restore_rows(previous_rows, attempt)
        while True:
            if time.monotonic()-began > timeout:
                raise TimeoutError('Shared worker time budget reached')
            phase = 'claim'
            claim = queue.claim(f'worker-{worker_id}', attempt)
            if claim is None:
                break  # Other workers finish their claims; no idle poll/model init.
            block, token = claim
            active_block = {k:block[k] for k in ('chunk_id','start','end')}
            saved_locally = False
            try:
                if transcriber is None:
                    phase = 'model_construct'
                    transcriber = QwenTranscriber()
                    transcriber.set_terms(plan['recognition_terms'])
                phase = 'audio_validate'
                blob = files[f'chunk-{block["chunk_id"]}.flac']
                claimed_audio_bytes += len(blob)
                claimed_audio_blocks += 1
                if hashlib.sha256(blob).hexdigest() != block['flac_sha256']:
                    raise ValueError('Original audio block hash changed')
                def load(_):
                    samples, rate = sf.read(io.BytesIO(blob), dtype='float32')
                    if rate != 16000 or samples.ndim != 1:
                        raise ValueError('Unexpected shared audio format')
                    return samples
                def completed(rows):
                    nonlocal saved_locally, phase
                    phase = 'local_checkpoint'
                    local_rows.append(rows[0])
                    save_local()  # Keep decoded output even if remote CAS is unavailable.
                    saved_locally = True
                    phase = 'remote_checkpoint'
                    queue.finish(block['chunk_id'], token, rows[0])
                phase = 'recognize'
                transcriber.recognize_blocks([block], load, timeout=max(0, timeout-(time.monotonic()-began)),
                    checkpoint=completed, keep_model=True)
                decoded.append(block['chunk_id'])
            except BaseException:
                # Completed records survive this release. A hard-killed Runner
                # is reclaimed on the next attempt, never by a timed-out lease.
                if not saved_locally:
                    try:
                        queue.release(block['chunk_id'], token)
                    except Exception:
                        pass  # Preserve the original failure.
                raise
    except BaseException as error:
        failure = error
        raise
    finally:
        # A final read/release failure must not replace the original ASR error.
        try:
            if transcriber is not None: transcriber.release_model()
        except Exception as error:
            cleanup_errors.append(error)
        snapshot_saved = False
        try:
            state = queue.snapshot()
            shards.seal({'queue.json': shards.encoded(state)}, 'shared-queue', shards.root()/'out'/'shared-queue.enc')
            snapshot_saved = True
        except Exception as error:
            cleanup_errors.append(error)
        primary = failure or (cleanup_errors[0] if cleanup_errors else None)
        from scripts.production_qwen import failure_code
        from scripts.coordination_transport import diagnostic
        audit = {'worker_id': worker_id, 'run_attempt': attempt,
                 'input_audio_bytes': input_audio_bytes, 'input_audio_blocks': input_audio_blocks,
                 'claimed_audio_bytes': claimed_audio_bytes, 'claimed_audio_blocks': claimed_audio_blocks,
                 'unused_input_audio_bytes': max(0, input_audio_bytes-claimed_audio_bytes),
                 'phase': phase if failure else 'cleanup' if cleanup_errors else 'complete',
                 'local_completed_blocks': sum(not r.get('missing_intervals') for r in local_rows),
                 'local_recognition_complete': not any(r.get('missing_intervals') for r in local_rows),
                 'local_failed_blocks': sum(bool(r.get('missing_intervals')) for r in local_rows),
                 'local_terminal_blocks': len(local_rows), 'queue_snapshot_saved': snapshot_saved,
                 'active_block': active_block if failure else None,
                 'block_diagnostics': [{'chunk_id':r['chunk_id'], 'start':r['start'], 'end':r['end'],
                     'quality_state':r.get('quality_state'),
                     'recognition_attempts':r.get('recognition_attempts',[]),
                     'missing_intervals':r.get('missing_intervals',[])}
                     for r in local_rows if r.get('recognition_attempts') or r.get('missing_intervals')],
                 'error_code': failure_code(primary) if primary else None,
                 'secondary_error_codes': [failure_code(e) for e in cleanup_errors]}
        if isinstance(primary,QwenTokenBudgetError): audit['recognition_failure'] = primary.diagnostic
        if diagnostic(primary): audit['coordination'] = diagnostic(primary)
        target = shards.root()/'out'/'worker-audit.json'
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(shards.encoded(audit))
        except Exception:
            if primary is None: raise
        if failure is None and cleanup_errors: raise cleanup_errors[0]
    return {**audit, 'decoded_chunk_ids': decoded,
            'failed_chunk_ids': [r['chunk_id'] for r in local_rows if r.get('missing_intervals')], 'seconds': time.monotonic()-began,
            'worker_id': worker_id, 'run_attempt': attempt}
