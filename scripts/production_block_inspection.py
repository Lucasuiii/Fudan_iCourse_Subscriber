"""Read ended workers' claim/release history; export no audio or ASR text.

Only safe block diagnostics leave Actions. Existing encryption keys stay on the
Runner, all APIs/Git operations are reads, no publisher/login/model is imported.
"""
import json
import os
import sys
from pathlib import Path
from scripts import production_pool as pool, production_qwen as pipeline
from scripts import sharded_qwen_pilot as shards
from scripts.qwen_sharding import validate_plan
from src.pipeline.asr_queue import validate_queue
from scripts.coordination_transport import read_with_retry

HISTORY_LIMIT=128


def released_claims(plan, history):
    """Newest-first decrypted snapshots; only claimed -> pending is a release."""
    found={}
    for newer,older in zip(history,history[1:]):
        validate_queue(plan,newer);validate_queue(plan,older)
        if newer['revision'] != older['revision']+1:
            raise ValueError('Nonconsecutive queue history')
        for block in plan['blocks']:
            key=str(block['chunk_id']);after=newer['blocks'][key];before=older['blocks'][key]
            if after['status']=='pending' and before['status']=='claimed':
                identity=(before['owner'],before['attempt'])
                if identity not in found:
                    found[identity]={'chunk_id':block['chunk_id'],'start':block['start'],
                        'end':block['end'],'release_revision':newer['revision']}
    return found


def inspect():
    run=os.environ['SOURCE_RUN_ID'];slot=int(os.environ['SOURCE_SLOT'])
    from scripts.production_result_export import identity
    identity(run,slot)
    info=pool.api(f'repos/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{run}')
    if info['status']!='completed' or info['path'].split('@')[0]!='.github/workflows/parallel_pilot.yml':
        raise ValueError('Block inspection requires an ended formal batch')
    store=pool.store_for(run)
    try:state=pool.read_state(store)[1]
    finally:store.close()
    if state['run_id']!=run or state['sha']!=info['head_sha'] or str(slot) not in state['courses']:
        raise ValueError('Block inspection source mismatch')
    # Do not inspect or clean history while any sibling is still running.
    for ticket in state['tickets']:
        if ticket['status']!='completed' or not ticket.get('run'):
            raise ValueError('Block inspection source has unresolved children')
        child=pool.api(f'repos/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{ticket["run"]}')
        if (child['status']!='completed' or child['head_sha']!=state['sha']
                or child['path'].split('@')[0]!='.github/workflows/'+pool.WORKFLOW
                or child['display_title']!=f'icourse-stage-{run}-{ticket["nonce"]}'
                or child['run_attempt']!=1):
            raise ValueError('Block inspection child mismatch')
    target=pipeline.root()/'block-inspection-source'
    pipeline.artifact(f'qwen-production-worker-plan-{slot}',target,run=run,required=True)
    with shards.environment({'GITHUB_RUN_ID':run,'COURSE_SLOT':str(slot)}):
        files=pipeline.decode(target/'worker-plan.enc','worker-plan')
    spec=json.loads(files['specification.json']);plan=spec['plan'];validate_plan(plan)
    if plan['run_id']!=run or plan['course_slot']!=slot or plan.get('execution')!='shared_queue':
        raise ValueError('Block inspection immutable plan mismatch')
    from scripts.shared_asr_worker import store_for
    queue=store_for(plan)
    try:
        head=queue.head()
        if not head:raise ValueError('Block inspection queue missing')
        read_with_retry('git_fetch',lambda:queue.command(['git','fetch','-q',f'--depth={HISTORY_LIMIT}',queue.remote,head]))
        commits=queue.command(['git','rev-list',f'--max-count={HISTORY_LIMIT}',head]).decode().splitlines()
        history=[]
        for commit in commits:
            if queue.command(['git','ls-tree','-r','--name-only',commit]).decode().splitlines()!=['queue.enc']:
                raise ValueError('Unexpected inspection queue tree')
            history.append(queue.unseal(queue.command(['git','show',commit+':queue.enc'])))
        releases=released_claims(plan,history)
        validate_queue(plan,history[0])
        counts={status:sum(r['status']==status for r in history[0]['blocks'].values())
                for status in ('pending','claimed','complete','failed')}
    finally:queue.close()
    workers=[]
    for ticket in state['tickets']:
        if ticket['slot']!=slot or ticket['stage']!='asr' or ticket['conclusion']!='failure':continue
        destination=target/str(ticket['worker'])
        pipeline.artifact(f'qwen-production-shared-{slot}-{ticket["worker"]}',destination,run=run,required=True)
        audit=json.loads((destination/'worker-audit.json').read_text())
        if audit['worker_id']!=ticket['worker'] or audit['run_attempt']!=ticket['attempt']:
            raise ValueError('Block inspection worker audit mismatch')
        row={'worker_id':ticket['worker'],'child_run':ticket['run'],'error_code':audit['error_code'],
             'phase':audit['phase'],'local_completed_blocks':audit['local_completed_blocks'],
             'generation_phase':'not_recorded','triggered_token_limit':None}
        if audit['error_code']=='qwen_token_budget' and audit['phase']=='recognize':
            row['last_released_block']=releases.get((f'worker-{ticket["worker"]}',ticket['attempt']))
        workers.append(row)
    payload={'source_run_id':run,'source_commit':state['sha'],'source_slot':slot,
             'queue_counts':counts,'history_states_read':len(history),
             'historical_token_limits':{'normal':2048,'unhinted_echo_retry':256},'workers':workers}
    pipeline.out('block-inspection.json').write_bytes(shards.encoded(payload))
    print('Ended block history inspected; only safe IDs, intervals and counts exported')


if __name__=='__main__':
    os.environ['QWEN_PRODUCTION_TASK']='true'
    try:inspect()
    except Exception as error:
        print(f'Block inspection failed ({type(error).__name__}); private details withheld')
        sys.exit(1)
