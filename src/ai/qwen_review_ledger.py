"""Whole-lecture rescue reservations survive summary failures and job retries."""
from pathlib import Path
import subprocess
import tempfile
from src.runtime import config
from src.ai.segment_rescue import MAX_CLOUD_SECONDS, MAX_CLOUD_CLIPS


def validate_ledger(state):
    import math
    attempts = state.get('attempts', [])
    seconds = sum(item['seconds'] for item in attempts)
    identities = [(i['interval']['start_ms'], i['interval']['end_ms']) for i in attempts]
    if (len(attempts) > MAX_CLOUD_CLIPS or not math.isfinite(seconds)
            or seconds < 0 or seconds > MAX_CLOUD_SECONDS + 1e-6
            or len(set(identities)) != len(identities)
            or any(not 0 < i['seconds'] <= 60
                   or abs(i['seconds']-(i['interval']['end_ms']-i['interval']['start_ms'])/1000) > 1e-6
                   or i['status'] not in ('reserved', 'complete', 'failed') for i in attempts)
            or abs(state.get('seconds', 0)-seconds) > 1e-6):
        raise ValueError('Invalid whole-lecture cloud quota checkpoint')


def review_prepared(material, pages, summarizer, state, checkpoint):
    from scripts.qwen_quality import review_quality
    from scripts.qwen_audio_alignment import align_suspects
    from src.ai.doubao_asr import rescue_intervals_pcm
    validate_ledger(state)
    if not config.DOUBAO_ASR_API_KEY or state.get('complete'):
        return state.get('material', {})
    if not callable(checkpoint):
        raise ValueError('Cloud review requires durable checkpoints')
    report = {'full_chunks': material['full_chunks'], 'vad_windows': material['vad_windows']}
    attempts = state.setdefault('attempts', [])
    used = {(a['interval']['start_ms'], a['interval']['end_ms']) for a in attempts}

    def material_state():
        variants, weak = [], []
        for item in attempts:
            interval, segments = item['interval'], item.get('segments', [])
            if interval.get('kind') == 'weak':
                if segments: weak.append((interval, segments))
            elif any(s['start_ms'] < interval['quote_end_ms'] and interval['quote_start_ms'] < s['end_ms'] for s in segments):
                variants.append({'original_quote': interval['text'], 'cloud_text': ' '.join(s['text'] for s in segments)})
        return {'variants': variants, 'weak_rescues': weak, 'unresolved': state.get('unresolved', []),
                'uncertain_calls': sum(i['status'] == 'reserved' for i in attempts)}

    def rescue(intervals):
        for interval in intervals:
            identity = (interval['start_ms'], interval['end_ms'])
            seconds = (interval['end_ms']-interval['start_ms'])/1000
            if identity in used or not 0 < seconds <= 60:
                continue
            if len(attempts) >= MAX_CLOUD_CLIPS or state.get('seconds', 0)+seconds > MAX_CLOUD_SECONDS:
                continue
            item = {'interval': interval, 'seconds': seconds, 'status': 'reserved'}
            attempts.append(item); used.add(identity)
            state['seconds'] = state.get('seconds', 0)+seconds
            state['material'] = material_state()
            checkpoint()  # Reserve before transport; unknown outcomes never refund.
            rescues, _, failed = rescue_intervals_pcm(material['audio_path'], config.DOUBAO_ASR_API_KEY,
                [interval], max_seconds=seconds, max_clips=1, hotwords=material['recognition_terms'])
            item.update(status='failed' if failed else 'complete', segments=rescues[0][1] if rescues else [])
            state['failed'] = failed; state['material'] = material_state()
            checkpoint()
            if failed: break

    try:
        if 'weak_intervals' not in state:
            from src.ai.segment_rescue import select_weak_windows
            weak = select_weak_windows(material.get('weak_windows', []), material.get('audio_seconds', 0),
                max_clips=MAX_CLOUD_CLIPS//2 if len(material.get('transcript', '')) >= 200 else MAX_CLOUD_CLIPS)
            state['weak_intervals'] = [dict(w, kind='weak') for w in weak]
            checkpoint()
        if not state.get('failed'): rescue(state['weak_intervals'])
        if 'intervals' not in state:
            intervals, unresolved = [], []
            remaining = MAX_CLOUD_SECONDS-state.get('seconds', 0)
            clips = MAX_CLOUD_CLIPS-len(attempts)
            if not state.get('failed') and remaining > 0 and clips > 0 and len(material.get('transcript', '')) >= 200:
                provider = summarizer.providers[0]
                suspects = review_quality(summarizer._clients[provider['name']], provider['models'][0],
                                         report, {'ppt': pages}, max_suspects=clips, input_budget=96000)
                if suspects:
                    with tempfile.TemporaryDirectory(prefix='icourse-review-') as tmp:
                        wav = Path(tmp)/'audio.wav'
                        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'f32le', '-ar', '16000',
                                        '-ac', '1', '-i', material['audio_path'], '-y', str(wav)],
                                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
                        intervals, _, unresolved, _ = align_suspects(report, suspects, wav, lambda _: None, budget=remaining)
            state.update(intervals=intervals, unresolved=unresolved)
            checkpoint()
        if not state.get('failed'): rescue(state['intervals'])
        state['material'] = material_state(); state['complete'] = True
        checkpoint()
        return state['material']
    except Exception as error:
        state['error_type'] = type(error).__name__
        state['material'] = material_state()
        checkpoint()
        return state['material']
