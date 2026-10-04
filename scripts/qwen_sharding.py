"""Immutable original-block plans and strict cross-runner result assembly."""
from __future__ import annotations
import hashlib
import json
import math
from scripts.qwen_segmentation import deduplicated_chunk_rows, reaches_acquisition_limit
from src.ai.qwen_transcriber import MODEL, REVISION, RATE

MAX_COURSES = 5
MAX_RUNNERS = 15


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def parse_baselines(raw):
    values = json.loads(raw)
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_COURSES:
        raise ValueError('Select one to five baseline artifacts')
    output = []
    for value in values:
        run = str(value.get('run_id', ''))
        artifact = value.get('artifact', '')
        if not run.isdigit() or artifact not in ('qwen-asr-encrypted-result-0',
                                               'qwen-asr-encrypted-result-1',
                                               'qwen-asr-encrypted-result'):
            raise ValueError('Invalid baseline reference')
        item = {'run_id': run, 'artifact': artifact}
        if any((v['run_id'], v['artifact']) == (run, artifact) for v in output):
            raise ValueError('Duplicate baseline reference')
        if 'reuse_source_slot' in value:
            source_slot = value['reuse_source_slot']
            if type(source_slot) is not int or not 0 <= source_slot < MAX_COURSES:
                raise ValueError('Invalid ASR reuse source slot')
            item['reuse_source_slot'] = source_slot
        output.append(item)
    return output


def build_plan(baseline, *, reference, course_slot, run_id, audio_sha256, mode='2'):
    if (baseline.get('source') != 'authorized_full_runtime' or baseline.get('complete') is not True
            or baseline.get('model') != MODEL or baseline.get('revision') != REVISION
            or baseline.get('acquisition_limit_reached')):
        raise ValueError('A complete production-recognizer baseline is required')
    return build_audio_plan(baseline, reference=reference, course_slot=course_slot,
                            run_id=run_id, audio_sha256=audio_sha256, mode=mode)


def build_audio_plan(audio, *, reference, course_slot, run_id, audio_sha256, mode='2', allow_partial=False):
    """Plan independently acquired VAD blocks without claiming ASR is complete."""
    baseline = audio
    selection = baseline.get('selection') or {}
    if any(not str(selection.get(k, '')).isdigit() for k in ('course_id', 'sub_id')):
        raise ValueError('Baseline lacks an exact private lecture selection')
    duration = baseline.get('audio_seconds')
    if (not isinstance(duration, (int, float)) or not math.isfinite(duration)
            or duration <= 0 or duration > 10800.1
            or (reaches_acquisition_limit(duration) and not allow_partial)):
        raise ValueError('Invalid baseline audio duration')
    rows = baseline.get('full_chunks')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Baseline lacks original chunks')
    blocks = []
    for i, row in enumerate(rows):
        start, end = row['start'], row['end']
        if (not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (start, end))
                or not 0 <= start < end <= duration or end-start > 122.01
                or (blocks and start < blocks[-1]['start'])):
            raise ValueError('Invalid original block timeline')
        samples = round(end*RATE)-round(start*RATE)
        if samples <= 0:
            raise ValueError('Empty original audio block')
        blocks.append({'chunk_id': i, 'start': start, 'end': end, 'samples': samples})
    pending = sum(b['samples'] for b in blocks)/RATE
    if mode not in ('2', 'auto'):
        raise ValueError('Unknown shard strategy')
    count = min(len(blocks), 2 if mode == '2' or pending <= 3600 else 3)
    groups, loads = [[] for _ in range(count)], [0]*count
    for block in sorted(blocks, key=lambda b: (-b['samples'], b['chunk_id'])):
        index = min(range(count), key=lambda i: (loads[i], i))
        groups[index].append(block['chunk_id'])
        loads[index] += block['samples']
    terms = baseline.get('recognition_terms', [])
    if not isinstance(terms, list) or len(terms) > 30 or any(not isinstance(t, str) for t in terms):
        raise ValueError('Invalid baseline terminology')
    plan = {'schema': 1, 'run_id': str(run_id), 'course_slot': course_slot,
            'reference': reference, 'selection': selection, 'model': MODEL, 'revision': REVISION,
            'recognition_terms': terms, 'audio_seconds': duration, 'audio_sha256': audio_sha256,
            'vad_windows': baseline.get('vad_windows', []), 'blocks': blocks,
            'pending_audio_seconds': pending, 'strategy': mode,
            'shards': [{'shard_id': i, 'chunk_ids': sorted(ids), 'audio_seconds': loads[i]/RATE}
                       for i, ids in enumerate(groups)]}
    if allow_partial:
        plan['allow_partial_comparison'] = True
    return plan


def validate_plan(plan):
    """Validate decrypted plans before trusting IDs as file names or allocations."""
    if (plan.get('schema') != 1 or plan.get('model') != MODEL or plan.get('revision') != REVISION
            or not str(plan.get('run_id', '')).isdigit()
            or type(plan.get('course_slot')) is not int or not 0 <= plan['course_slot'] < MAX_COURSES):
        raise ValueError('Invalid plan identity')
    blocks, shards = plan['blocks'], plan['shards']
    if not blocks or not 1 <= len(shards) <= 3:
        raise ValueError('Invalid shard plan')
    for i, block in enumerate(blocks):
        if (type(block['chunk_id']) is not int or block['chunk_id'] != i
                or not all(isinstance(block[k], (int, float)) and math.isfinite(block[k]) for k in ('start', 'end'))
                or not 0 <= block['start'] < block['end'] <= plan['audio_seconds']
                or block['end']-block['start'] > 122.01
                or type(block['samples']) is not int
                or block['samples'] != round(block['end']*RATE)-round(block['start']*RATE)):
            raise ValueError('Invalid block')
    allocated = []
    for i, shard in enumerate(shards):
        if type(shard['shard_id']) is not int or shard['shard_id'] != i or not shard['chunk_ids']:
            raise ValueError('Invalid shard identity')
        if any(type(n) is not int or not 0 <= n < len(blocks) for n in shard['chunk_ids']):
            raise ValueError('Invalid shard allocation')
        allocated.extend(shard['chunk_ids'])
    if sorted(allocated) != list(range(len(blocks))):
        raise ValueError('Missing or duplicate block allocation')


def validate_result(plan, result, shard_id, *, require_complete=False):
    validate_plan(plan)
    if type(shard_id) is not int or not 0 <= shard_id < len(plan['shards']):
        raise ValueError('Invalid result shard')
    if (result.get('plan_hash') != fingerprint(plan) or result.get('shard_id') != shard_id
            or type(result.get('shard_id')) is not int):
        raise ValueError('Result does not match immutable plan')
    expected = set(plan['shards'][shard_id]['chunk_ids'])
    seen = set()
    for row in result['chunks']:
        n = row.get('chunk_id')
        if type(n) is not int or n not in expected or n in seen:
            raise ValueError('Unexpected or duplicate result block')
        block = plan['blocks'][n]
        if (row.get('start') != block['start'] or row.get('end') != block['end']
                or not isinstance(row.get('text'), str)):
            raise ValueError('Result changed original timestamps or text format')
        seen.add(n)
    if result.get('complete') is True and seen != expected:
        raise ValueError('Shard claimed success with missing blocks')
    if require_complete and (result.get('complete') is not True or seen != expected):
        raise ValueError('Shard incomplete; summary forbidden')
    return seen


def assemble(plan, results, baseline):
    validate_plan(plan)
    if len(results) != len(plan['shards']):
        raise ValueError('Missing shard results')
    seen = set()
    rows = []
    for result in results:
        shard = result['shard_id']
        if shard in seen:
            raise ValueError('Duplicate shard result')
        validate_result(plan, result, shard, require_complete=True)
        seen.add(shard)
        rows.extend(result['chunks'])
    rows.sort(key=lambda r: r['chunk_id'])
    cleaned = deduplicated_chunk_rows(rows)
    text = '\n'.join(r['text'] for r in cleaned)
    if not text.strip():
        raise ValueError('All decoded blocks empty; summary forbidden')
    same_timeline = (len(baseline['full_chunks']) == len(rows) and all(
        old['start'] == new['start'] and old['end'] == new['end']
        for old, new in zip(baseline['full_chunks'], rows)))
    changed = ([i for i, (old, new) in enumerate(zip(baseline['full_chunks'], rows))
                if old.get('text', '') != new['text']] if same_timeline else None)
    return {'source': 'authorized_sharded_runtime', 'model': MODEL, 'revision': REVISION,
            'complete': True, 'selection': plan['selection'], 'recognition_terms': plan['recognition_terms'],
            'audio_seconds': plan['audio_seconds'], 'pending_audio_seconds': plan['pending_audio_seconds'],
            'audio_sha256': plan['audio_sha256'], 'vad_windows': plan['vad_windows'],
            'full_chunks': rows, 'transcript': text,
            'segments': [{'start_ms': round(r['start']*1000), 'end_ms': round(r['end']*1000), 'text': r['text']}
                         for r in cleaned],
            'sharding': {'plan_hash': fingerprint(plan), 'shards': len(results), 'strategy': plan['strategy'],
                         'asr_reuse_run_id': plan.get('asr_reuse_run_id'),
                         'worker_decode_seconds': sum(r.get('seconds', 0) for r in results),
                         'slowest_worker_seconds': max(r.get('seconds', 0) for r in results),
                         'worker_attempts': [r.get('attempts', []) for r in results]},
            'baseline_comparison': {'reference': plan['reference'], 'baseline_asr_seconds': baseline.get('seconds'),
                'baseline_summary_complete': bool(baseline.get('test_summary')),
                'baseline_asr_complete': baseline.get('complete') is True,
                'same_audio_duration_verified': abs(plan['audio_seconds']-baseline['audio_seconds']) <= 0.1,
                'baseline_audio_seconds': baseline['audio_seconds'],
                'baseline_cloud_review': baseline.get('cloud_review'),
                'changed_chunk_ids': changed, 'same_block_timeline_verified': same_timeline,
                'baseline_chunk_count': len(baseline['full_chunks']), 'sharded_chunk_count': len(rows),
                'transcript_equal': text == baseline.get('transcript'),
                'quality_accuracy_verified': False, 'same_original_audio_hash_verified': False,
                'note': 'Same exact lecture; block timelines checked separately. Old baseline did not retain an audio hash.'},
            'empty_chunks': [r['chunk_id'] for r in rows if not r['text'].strip()]}
