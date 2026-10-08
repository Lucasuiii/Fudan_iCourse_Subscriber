"""Qwen-only CPU recognizer: bounded two-minute blocks, silence-aware VAD."""
from __future__ import annotations
import gc
import os
from pathlib import Path
import re
import subprocess
import time

from src.runtime import config
from scripts.qwen_segmentation import plan_long_chunks, deduplicated_chunk_rows
from scripts.qwen_quality import context_echo, low_information, bounded_retry, generation_audit

MODEL='Qwen/Qwen3-ASR-1.7B'
REVISION='7278e1e70fe206f11671096ffdd38061171dd6e5'
RATE=16000
NORMAL_TOKENS=2048
UNHINTED_TOKENS=256
RESCUE_SECONDS=600


class QwenTokenBudgetError(RuntimeError):
    """No generated text is retained when the conservative budget guard trips."""
    def __init__(self, phase, limit, tokens, generated_tokens=None):
        super().__init__('Qwen chunk reached its token budget')
        self.diagnostic = {'error_code': 'qwen_token_budget', 'generation_phase': phase,
                           'token_limit': limit, 'observed_text_tokens': tokens,
                           'budget_guard': 'generated_limit' if generated_tokens is not None else 'text_margin'}
        if generated_tokens is not None:
            self.diagnostic['generated_tokens'] = generated_tokens


class IncompleteQwenRecognitionError(RuntimeError):
    def __init__(self, missing):
        super().__init__('Qwen recognition incomplete; missing audio intervals')
        self.missing_intervals = missing



class QwenTranscriber:
    def __init__(self, backend=None, model_dir=None, num_threads=None):
        if backend and backend!='qwen':
            raise ValueError('This branch supports Qwen only')
        self._model=None
        self._num_threads=num_threads or config.ASR_NUM_THREADS
        self._last_duration=0.0
        self._media_duration=None
        self._last_speech_windows=[]
        self.last_vad_windows=[]
        self.last_chunks=[]
        self.last_prepare_stats={}
        self._terms=[]

    @property
    def last_audio_duration(self): return self._last_duration

    @property
    def last_media_duration(self): return self._media_duration

    @property
    def last_speech_windows(self): return self._last_speech_windows

    def set_terms(self, terms):
        self._terms=list(dict.fromkeys(t.strip() for t in terms if isinstance(t,str) and t.strip()))[:30]

    def reset_lecture_state(self):
        self._last_speech_windows=[];self.last_chunks=[];self.last_vad_windows=[]
        self._last_duration=0.0;self._media_duration=None

    def _init(self):
        if self._model is not None: return
        import torch
        from huggingface_hub import snapshot_download
        from qwen_asr import Qwen3ASRModel
        torch.set_num_threads(self._num_threads)
        # Set interop once; subsequent lecture loads share the same process.
        try: torch.set_num_interop_threads(1)
        except RuntimeError: pass
        path=snapshot_download(MODEL,revision=REVISION,allow_patterns=['*.json','*.safetensors','*.txt'])
        self._model=Qwen3ASRModel.from_pretrained(path,dtype=torch.float32,device_map='cpu',
            attn_implementation='eager',max_inference_batch_size=1,max_new_tokens=NORMAL_TOKENS)

    def release_model(self):
        self._model=None
        gc.collect()

    def _recognize(self, samples, *, unhinted=False):
        import torch
        from transformers import StoppingCriteriaList
        context=('术语：'+'、'.join(self._terms)) if self._terms and not unhinted else ''
        with generation_audit(self._model) as generations, torch.inference_mode():
            result=self._model.transcribe(audio=(samples,RATE),context=context,language='Chinese')[0]
        row={'text':result.text,'quality_state':'recognized'}
        def check(result, phase, limit, generations):
            tokens=len(self._model.processor.tokenizer.encode(result.text,add_special_tokens=False))
            capped=next((g for g in generations if g['generated_tokens']>=g['token_limit']),None)
            if capped or tokens>=limit-8:
                raise QwenTokenBudgetError(phase,capped['token_limit'] if capped else limit,tokens,
                    capped['generated_tokens'] if capped else None)
        check(result,'normal',NORMAL_TOKENS,generations)
        if context and context_echo(result.text,context):
            with bounded_retry(self._model,StoppingCriteriaList) as state, \
                    generation_audit(self._model) as generations, torch.inference_mode():
                result=self._model.transcribe(audio=(samples,RATE),context='',language='Chinese')[0]
            check(result,'unhinted_echo_retry',UNHINTED_TOKENS,generations)
            row.update(text=result.text,quality_state='unhinted_retry')
            if state['timed_out']:
                row.update(text='',quality_state='retry_timeout')
        if context and context_echo(row['text'],context):
            row.update(text='',quality_state='unresolved_context_echo')
        if low_information(row['text']):
            if row['quality_state'] not in ('retry_timeout','unresolved_context_echo'):
                row['quality_state']='low_information'
            row['text']=''
        return row

    def _recognize_resilient(self, samples, block, deadline):
        """At most one full unhinted retry, then 30s clips and 15s leaves.

        Immutable block identity and sample coverage are kept. Failed generations
        contribute no text; successful subclips survive with explicit gaps.
        Model/setup, corrupt audio and coordination failures still propagate.
        """
        attempts=[]
        offset=round(block['start']*RATE)
        def attempt(a,b,*,rescue=False,kind='original'):
            start=block['start'] if a==0 else (offset+a)/RATE
            end=block['end'] if b==len(samples) else (offset+b)/RATE
            if time.monotonic()>=deadline:
                return None, {'start':start,'end':end,'error_code':'worker_deadline'}
            try:
                if rescue:
                    from transformers import StoppingCriteriaList
                    # Separate bounds from the echo retry's small 256-token cap.
                    with bounded_retry(self._model,StoppingCriteriaList,
                            seconds=min(RESCUE_SECONDS,60+8*(b-a)/RATE,max(0,deadline-time.monotonic())),
                            tokens=NORMAL_TOKENS,clock=time.monotonic) as state:
                        row=self._recognize(samples[a:b],unhinted=True)
                    if state['timed_out']:
                        row={'text':'','quality_state':'retry_timeout'}
                else:
                    row=self._recognize(samples[a:b])
                if row.get('quality_state') in ('retry_timeout','unresolved_context_echo'):
                    issue={'error_code':row['quality_state']}
                else:
                    return row,None
            except QwenTokenBudgetError as error:
                issue=error.diagnostic
            issue=dict(issue,start=start,end=end,attempt_kind=kind)
            attempts.append(issue)
            return None,issue
        row,issue=attempt(0,len(samples))
        if row is not None: return row
        row,issue=attempt(0,len(samples),rescue=True,kind='unhinted_full_retry')
        if row is not None:
            return dict(row,quality_state='bounded_retry',recognition_attempts=attempts)
        if len(samples)<=15*RATE or issue['error_code']=='worker_deadline':
            return {'text':'','quality_state':'missing_audio','missing_intervals':[issue],
                    'recognition_attempts':attempts}
        parts=[];missing=[]
        step=(15 if len(samples)<=30*RATE else 30)*RATE
        for a in range(0,len(samples),step):
            b=min(len(samples),a+step)
            row,issue=attempt(a,b,rescue=True,kind='split_15s' if step==15*RATE else 'split_30s')
            if row is not None:
                parts.append(row['text']);continue
            # Never retry an exhausted deadline or an already short leaf.
            if b-a<=15*RATE or issue['error_code']=='worker_deadline':
                missing.append(issue);continue
            for c in range(a,b,15*RATE):
                d=min(b,c+15*RATE)
                row,issue=attempt(c,d,rescue=True,kind='split_15s')
                if row is None: missing.append(issue)
                else: parts.append(row['text'])
        return {'text':'\n'.join(t for t in parts if t),
                'quality_state':'missing_audio' if missing else 'split_retry',
                'missing_intervals':missing,'recognition_attempts':attempts}

    def _drain_vad(self, vad, windows):
        while not vad.empty():
            segment=vad.front
            windows.append((segment.start/RATE,(segment.start+len(segment.samples))/RATE))
            vad.pop()

    def prepare_pcm_stream(self, read_fn, is_eof_fn, stderr_provider, return_code_fn,
                            timeout=18000, wait_on_empty_sec=0.1, label='tail', audio_path=None,
                            idle_timeout=None):
        import numpy as np
        import sherpa_onnx
        from src.ai.transcriber import NoAudioStreamError
        if not audio_path:
            raise ValueError('Qwen requires the local PCM file for bounded disk reads')
        self._last_speech_windows=[];self.last_chunks=[];self.last_vad_windows=[]
        self._last_duration=0;self._media_duration=None
        cfg=sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model=config.SILERO_VAD_PATH
        cfg.silero_vad.threshold=0.5
        cfg.silero_vad.min_silence_duration=0.8
        cfg.silero_vad.min_speech_duration=0.25
        cfg.silero_vad.max_speech_duration=28.0
        cfg.sample_rate=RATE;cfg.num_threads=1
        vad=sherpa_onnx.VoiceActivityDetector(cfg,buffer_size_in_seconds=60)
        began=time.monotonic();progress=began;total=0;pending=b'';eof=False
        self.last_prepare_stats={}
        try:
            while True:
                now=time.monotonic()
                if now-began>timeout: raise TimeoutError('Audio preparation deadline exceeded')
                raw=read_fn(512*4)
                if not raw and is_eof_fn():
                    # Drain final samples written between an empty read and exit.
                    raw=read_fn(512*4)
                    if not raw:
                        eof=True
                        break
                if raw:
                    progress=time.monotonic()
                    pending+=raw
                    if len(pending)<512*4: continue
                    block,pending=pending[:512*4],pending[512*4:]
                    vad.accept_waveform(np.frombuffer(block,dtype=np.float32))
                    total+=512;self._drain_vad(vad,self.last_vad_windows)
                else:
                    if idle_timeout is not None and now-progress>idle_timeout:
                        raise TimeoutError('Audio preparation stalled')
                    time.sleep(wait_on_empty_sec)
        finally:
            ended=time.monotonic()
            self.last_prepare_stats={'elapsed_seconds':ended-began,
                'idle_seconds':ended-progress,'timeout_seconds':timeout,
                'idle_timeout_seconds':idle_timeout,'stream_eof':eof}
        if pending:
            if len(pending)%4: raise RuntimeError('Incomplete PCM sample')
            samples=np.frombuffer(pending,dtype=np.float32)
            total+=len(samples)
            vad.accept_waveform(np.pad(samples,(0,512-len(samples))))
        vad.flush();self._drain_vad(vad,self.last_vad_windows)
        del vad
        stderr=stderr_provider().decode(errors='replace')
        if 'does not contain any stream' in stderr or 'matches no streams' in stderr:
            raise NoAudioStreamError('No audio stream')
        if return_code_fn() not in (None,0): raise RuntimeError('Audio download failed')
        if total==0: raise RuntimeError('No decoded audio')
        self._last_duration=total/RATE
        match=re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)',stderr)
        if match: self._media_duration=int(match[1])*3600+int(match[2])*60+float(match[3])
        self.last_vad_windows=[(a,min(b,self._last_duration)) for a,b in self.last_vad_windows if a<self._last_duration]
        return plan_long_chunks(self.last_vad_windows,self._last_duration)

    def recognize_blocks(self, blocks, load_samples, *, checkpoint=None, timeout=18000, keep_model=False):
        """Decode immutable original blocks; leave global ordering/dedupe to caller."""
        self.last_chunks=[]
        self._last_speech_windows=[]
        began=time.monotonic()
        try:
            if timeout <= 0:
                raise TimeoutError('Qwen shard timeout')
            self._init()
            for i, block in enumerate(blocks):
                if time.monotonic()-began>timeout:
                    raise TimeoutError('Qwen shard timeout')
                samples=load_samples(block)
                expected=round(block['end']*RATE)-round(block['start']*RATE)
                if len(samples)!=expected:
                    raise RuntimeError('Incomplete Qwen audio block')
                started=time.monotonic()
                row=self._recognize_resilient(samples,block,began+timeout)
                row.update(start=block['start'],end=block['end'],
                           chunk_id=block['chunk_id'],decode_seconds=time.monotonic()-started)
                self.last_chunks.append(row)
                for a,b in self.last_vad_windows:
                    a,b=max(a,row['start']),min(b,row['end'])
                    if b>a:
                        self._last_speech_windows.append({'start_ms':round(a*1000),'end_ms':round(b*1000),
                            'text':row['text'],'chunk_id':row['chunk_id']})
                if checkpoint:
                    checkpoint(self.last_chunks)
                print(f'[Qwen] Block {i+1}/{len(blocks)} completed.',flush=True)
                del samples;gc.collect()
        finally:
            if not keep_model:
                self.release_model()
        return self.last_chunks

    def _consume_pcm_stream(self, read_fn, is_eof_fn, stderr_provider, return_code_fn,
                            timeout=18000, wait_on_empty_sec=0.1, label='tail', audio_path=None):
        import numpy as np
        began=time.monotonic()
        chunks=self.prepare_pcm_stream(read_fn,is_eof_fn,stderr_provider,return_code_fn,
            timeout=timeout,wait_on_empty_sec=wait_on_empty_sec,label=label,audio_path=audio_path)
        if not chunks:
            return '',[]
        blocks=[{'chunk_id':i,'start':a,'end':b} for i,(a,b) in enumerate(chunks)]
        with open(audio_path,'rb') as audio:
            def load(block):
                audio.seek(round(block['start']*RATE)*4)
                count=round(block['end']*RATE)-round(block['start']*RATE)
                return np.frombuffer(audio.read(count*4),dtype=np.float32).copy()
            self.recognize_blocks(blocks,load,timeout=max(0,timeout-(time.monotonic()-began)))
        missing=[dict(gap,chunk_id=row['chunk_id']) for row in self.last_chunks
                 for gap in row.get('missing_intervals',[])]
        if missing:
            raise IncompleteQwenRecognitionError(missing)
        rows=deduplicated_chunk_rows(self.last_chunks)
        segments=[{'start_ms':round(r['start']*1000),'end_ms':round(r['end']*1000),'text':r['text']}
                  for r in rows]
        return '\n'.join(r['text'] for r in rows),segments

    def transcribe_tail(self, audio_path, ffmpeg_proc, stderr_chunks, timeout=18000):
        began=time.monotonic()
        while not os.path.exists(audio_path):
            if ffmpeg_proc.poll() is not None: raise RuntimeError('Audio download did not create PCM')
            if time.monotonic()-began>60: raise TimeoutError('Audio file unavailable')
            time.sleep(0.1)
        with open(audio_path,'rb') as stream:
            return self._consume_pcm_stream(stream.read,lambda:ffmpeg_proc.poll() is not None,
                lambda:b''.join(stderr_chunks),lambda:ffmpeg_proc.returncode,
                timeout=timeout,audio_path=audio_path)

    @staticmethod
    def probe_duration(url,http_headers=None,timeout=30):
        command=['ffprobe','-v','error']
        if http_headers: command+=['-headers',http_headers]
        command+=['-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',url]
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
            return float(result.stdout.strip()) if result.returncode==0 else None
        except (subprocess.TimeoutExpired,ValueError): return None
