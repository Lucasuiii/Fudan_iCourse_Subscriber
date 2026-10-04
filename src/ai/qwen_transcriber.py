"""Qwen-only CPU recognizer: bounded two-minute blocks, silence-aware VAD."""
from __future__ import annotations
import gc
import os
from pathlib import Path
import re
import subprocess
import time

from src.runtime import config
from scripts.qwen_segmentation import plan_long_chunks, join_chunk_text
from scripts.qwen_quality import context_echo, low_information, bounded_retry

MODEL='Qwen/Qwen3-ASR-1.7B'
REVISION='7278e1e70fe206f11671096ffdd38061171dd6e5'
RATE=16000


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
            attn_implementation='eager',max_inference_batch_size=1,max_new_tokens=2048)

    def release_model(self):
        self._model=None
        gc.collect()

    def _recognize(self, samples):
        import torch
        from transformers import StoppingCriteriaList
        context=('术语：'+'、'.join(self._terms)) if self._terms else ''
        with torch.inference_mode():
            result=self._model.transcribe(audio=(samples,RATE),context=context,language='Chinese')[0]
        row={'text':result.text,'quality_state':'recognized'}
        limit=2048
        if context and context_echo(result.text,context):
            with bounded_retry(self._model,StoppingCriteriaList) as state, torch.inference_mode():
                result=self._model.transcribe(audio=(samples,RATE),context='',language='Chinese')[0]
            row.update(text=result.text,quality_state='unhinted_retry')
            limit=256
            if state['timed_out']:
                row.update(text='',quality_state='retry_timeout')
        tokens=len(self._model.processor.tokenizer.encode(result.text,add_special_tokens=False))
        if tokens>=limit-8:
            # Never persist a token-limit-cut lecture as an apparently complete one.
            raise RuntimeError('Qwen chunk reached its token budget')
        if context and context_echo(row['text'],context):
            row.update(text='',quality_state='unresolved_context_echo')
        if low_information(row['text']):
            if row['quality_state'] not in ('retry_timeout','unresolved_context_echo'):
                row['quality_state']='low_information'
            row['text']=''
        return row

    def _drain_vad(self, vad, windows):
        while not vad.empty():
            segment=vad.front
            windows.append((segment.start/RATE,(segment.start+len(segment.samples))/RATE))
            vad.pop()

    def _consume_pcm_stream(self, read_fn, is_eof_fn, stderr_provider, return_code_fn,
                            timeout=18000, wait_on_empty_sec=0.1, label='tail', audio_path=None):
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
        began=time.monotonic();total=0;pending=b''
        while True:
            if time.monotonic()-began>timeout: raise TimeoutError('Qwen lecture timeout')
            raw=read_fn(512*4)
            if raw:
                pending+=raw
                if len(pending)<512*4: continue
                block,pending=pending[:512*4],pending[512*4:]
                vad.accept_waveform(np.frombuffer(block,dtype=np.float32))
                total+=512;self._drain_vad(vad,self.last_vad_windows)
            elif is_eof_fn(): break
            else: time.sleep(wait_on_empty_sec)
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
        chunks=plan_long_chunks(self.last_vad_windows,self._last_duration)
        if not chunks: return '',[]
        self._init()
        try:
            with open(audio_path,'rb') as audio:
                for i,(start,end) in enumerate(chunks):
                    if time.monotonic()-began>timeout: raise TimeoutError('Qwen lecture timeout')
                    audio.seek(round(start*RATE)*4)
                    count=round(end*RATE)-round(start*RATE)
                    samples=np.frombuffer(audio.read(count*4),dtype=np.float32).copy()
                    if len(samples)!=count: raise RuntimeError('Incomplete Qwen audio block')
                    row=self._recognize(samples)
                    row.update(start=start,end=end)
                    self.last_chunks.append(row)
                    # Speech coverage is actual VAD, NOT the padded chunk extent.
                    for a,b in self.last_vad_windows:
                        a,b=max(a,start),min(b,end)
                        if b>a:
                            self._last_speech_windows.append({'start_ms':round(a*1000),'end_ms':round(b*1000),
                                'text':row['text'], 'chunk_id':i})
                    print(f'[Qwen] Block {i+1}/{len(chunks)} completed.',flush=True)
                    del samples;gc.collect()
        finally:
            self.release_model()  # leave RAM for OCR / the separate forced aligner
        segments=[{'start_ms':round(r['start']*1000),'end_ms':round(r['end']*1000),'text':r['text']}
                  for r in self.last_chunks if r['text']]
        return join_chunk_text(self.last_chunks),segments

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
