"""Formal recovery stage, with explicit runtime services and frozen inputs."""
from __future__ import annotations
import json
import os
import subprocess


def historical_asr_cost(runtime, db, course, date):
    import math
    import statistics
    from scripts.production_pool import DEFAULT_RTF
    from src.ai.qwen_transcriber import MODEL, REVISION
    estimates = []
    rows = db.conn.execute('''SELECT m.value FROM meta m JOIN lectures l
        ON m.key = 'qwen_pipeline:' || l.sub_id WHERE l.course_id=? AND l.date < ?
        AND l.processed_at IS NOT NULL AND l.deleted_at IS NULL ORDER BY l.date DESC LIMIT 10''', (str(course), date))
    for row in rows:
        try:
            value = json.loads(row[0]); cost = value['asr_cost']
            if not value.get('complete') or (cost['model'], cost['revision']) != (MODEL, REVISION) or cost['blocks'] < 5:
                continue
            ratio = cost['decode_seconds']/cost['audio_seconds']
            if math.isfinite(ratio) and .25 <= ratio <= 10: estimates.append(ratio)
        except (KeyError, ValueError, TypeError, ZeroDivisionError): continue
    return statistics.median(estimates) if estimates else DEFAULT_RTF


def last_finalization_attempt(runtime, run, slot, *, prior_only=False):
    """A lost runner checkpoint must not silently refund a later cloud attempt."""
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        return 0
    if os.environ.get('POOL_ARTIFACTS') == 'true' or str(run) in runtime._POOL_RUNS:
        from scripts.production_pool import finalization_attempt
        value = finalization_attempt(str(run), slot, prior_only,
            allow_legacy=str(run) != os.environ['GITHUB_RUN_ID'] and str(run) not in runtime._POOL_RUNS)
        if value is not None: return value
    repo = os.environ['GITHUB_REPOSITORY']
    if run == os.environ['GITHUB_RUN_ID']:
        attempts = int(os.environ.get('GITHUB_RUN_ATTEMPT', '1')) - int(prior_only)
    else:
        info = json.loads(subprocess.check_output(['gh','api',f'repos/{repo}/actions/runs/{run}'],
                                                  stderr=subprocess.PIPE, timeout=60))
        attempts = info['run_attempt']
    if attempts > 20:
        raise ValueError('Finalization history exceeds bounded recovery audit')
    latest = 0
    for attempt in range(1, attempts+1):
        pages = json.loads(subprocess.check_output(['gh','api','--paginate','--slurp',
            f'repos/{repo}/actions/runs/{run}/attempts/{attempt}/jobs?per_page=100'],
            stderr=subprocess.PIPE, timeout=120))
        for page in pages:
            for job in page['jobs']:
                if not job['name'].endswith(f'finalize-{slot}'):
                    continue
                steps = job.get('steps')
                if steps is None:
                    # No visibility is not evidence that quota was unused.
                    latest = max(latest, attempt)
                elif any(step['name'] == 'Finalize through LectureRunner with saved quota'
                         and step.get('conclusion') != 'skipped' and step.get('started_at')
                         for step in steps):
                    latest = max(latest, attempt)
    return latest


def validate_checkpoint_age(runtime, saved, run, slot, *, prior_only=False):
    stamp = int(runtime.read_json(saved.get('attempt.json', b'0')))
    if runtime.last_finalization_attempt(run, slot, prior_only=prior_only) > stamp:
        raise ValueError('A later finalization lost its quota checkpoint; automatic refund forbidden')

