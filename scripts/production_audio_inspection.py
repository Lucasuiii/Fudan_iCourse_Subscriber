"""Inspect already prepared audio; no acquisition, ASR, LLM or publication.

The database key stays on the runner. Only bounded listening clips and
diagnostics are encrypted for the user's ephemeral recipient key.
"""
from __future__ import annotations
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def audio_stats(path):
    import numpy as np
    import soundfile as sf
    seconds = []
    total = 0; square_sum = 0.0; peak = 0.0
    with sf.SoundFile(path) as audio:
        if audio.samplerate != 16000 or audio.channels != 1:
            raise ValueError('Unexpected prepared audio format')
        for samples in audio.blocks(blocksize=16000, dtype='float32'):
            if not np.isfinite(samples).all():
                raise ValueError('Nonfinite prepared audio')
            energy = float(np.sum(samples.astype('float64')**2))
            seconds.append(math.sqrt(energy/len(samples)))
            total += len(samples); square_sum += energy
            peak = max(peak, float(np.max(np.abs(samples))))
    if not total: raise ValueError('Empty prepared audio')
    return {'samples': total, 'audio_seconds': total/16000,
            'rms': math.sqrt(square_sum/total), 'peak': peak,
            'one_second_rms_percentiles': [float(np.percentile(seconds, n)) for n in (10,50,90,99)],
            'seconds_below_rms_1e_5': sum(r < 1e-5 for r in seconds)}, seconds


def listening_offsets(levels, duration):
    # Include the strongest two separated windows, plus an even timeline sample.
    # These are diagnostic excerpts, not a claim of classroom completeness.
    starts = []
    limit = max(0, duration-10)
    for index in sorted(range(len(levels)), key=lambda i: -levels[i]):
        start = min(limit, max(0, index-5))
        if all(abs(start-other) >= 15 for other in starts): starts.append(start)
        if len(starts) == 2: break
    for fraction in (.05,.23,.41,.59,.77,.95):
        start = min(limit, duration*fraction)
        if all(abs(start-other) >= 10 for other in starts): starts.append(start)
    return sorted(starts[:8])


def inspect():
    from scripts import production_qwen as pipeline
    from scripts import sharded_qwen_pilot as shards
    from scripts.production_result_export import encrypt, identity
    from src.ai.qwen_transcriber import QwenTranscriber
    run, slot = os.environ['SOURCE_RUN_ID'], int(os.environ['SOURCE_SLOT'])
    identity(run, slot)
    recipient = base64.b64decode(os.environ['RECIPIENT_PUBLIC_KEY'], validate=True)
    if len(recipient) != 32: raise ValueError('Invalid recipient public key')
    info = json.loads(subprocess.check_output(['gh','api',
        f'repos/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{run}'], stderr=subprocess.PIPE, timeout=60))
    if info['status'] != 'completed' or info['path'].split('@')[0] != '.github/workflows/parallel_pilot.yml':
        raise ValueError('Source is not a completed production pilot')
    target = pipeline.root()/'audio-inspection-source'
    pipeline.artifact(f'qwen-production-prepare-{slot}', target, run=run, required=True)
    with shards.environment({'GITHUB_RUN_ID':run, 'COURSE_SLOT':str(slot)}):
        files = pipeline.decode(target/'prepared.enc', 'prepared')
    spec = json.loads(files['specification.json']); plan = spec['plan']
    from scripts.qwen_sharding import validate_plan
    validate_plan(plan)
    blob = files['lecture.flac']; digest = hashlib.sha256(blob).hexdigest()
    if digest != plan['audio_sha256']: raise ValueError('Prepared audio hash changed')
    flac = pipeline.root()/'inspection.flac'; flac.write_bytes(blob)
    del files, blob
    metrics, levels = audio_stats(flac)
    if abs(metrics['audio_seconds']-plan['audio_seconds']) > .1:
        raise ValueError('Prepared audio duration changed')
    raw = pipeline.root()/'inspection.raw'
    shards.command(['ffmpeg','-nostdin','-v','error','-i',str(flac),'-f','f32le',
                    '-ar','16000','-ac','1','-y',str(raw)], timeout=300)
    transcriber = QwenTranscriber()
    with raw.open('rb') as audio:
        windows = transcriber.prepare_pcm_stream(audio.read, lambda: True, lambda: b'', lambda: 0,
                                                 audio_path=str(raw), timeout=300)
    # This calls only Silero VAD; never initialize Qwen or recognize any block.
    if transcriber._model is not None: raise RuntimeError('Audio inspection initialized ASR')
    lecture = spec['lecture']
    payload = {'source_run_id':run, 'source_commit':info['head_sha'], 'source_slot':slot,
        'course_id':spec['course_id'], 'course_title':spec['course_title'],
        'date':lecture.get('date'), 'sub_title':lecture.get('sub_title'), 'sub_id':lecture['sub_id'],
        'audio_sha256':digest, 'metrics':metrics,
        'original_vad_windows':len(plan['vad_windows']), 'original_blocks':len(plan['blocks']),
        'repeat_vad_windows':len(transcriber.last_vad_windows), 'repeat_blocks':len(windows),
        'repeat_vad_speech_seconds':sum(b-a for a,b in transcriber.last_vad_windows),
        'repeat_vad_intervals':transcriber.last_vad_windows, 'clips':[]}
    for i,start in enumerate(listening_offsets(levels, metrics['audio_seconds'])):
        clip = pipeline.root()/f'inspection-{i}.mp3'
        shards.command(['ffmpeg','-nostdin','-v','error','-ss',str(start),'-i',str(flac),
                        '-t','10','-ar','16000','-ac','1','-codec:a','libmp3lame','-b:a','32k',
                        '-y',str(clip)], timeout=30)
        data = clip.read_bytes()
        payload['clips'].append({'start_seconds':start, 'duration_seconds':min(10,metrics['audio_seconds']-start),
                                'sha256':hashlib.sha256(data).hexdigest(), 'mp3_base64':base64.b64encode(data).decode()})
    pipeline.out('audio-inspection.enc').write_bytes(encrypt(shards.encoded(payload), recipient, run, slot))
    print('Existing audio inspected and bounded clips exported encrypted; no ASR or acquisition')


if __name__ == '__main__':
    os.environ['QWEN_PRODUCTION_TASK'] = 'true'
    try: inspect()
    except Exception as error:
        print(f'Audio inspection failed ({type(error).__name__}); private details withheld')
        sys.exit(1)
