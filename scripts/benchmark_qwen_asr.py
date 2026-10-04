"""Manual, isolated CPU benchmark. No database or email calls.

Course selection is a Secret. Audio stays on the ephemeral runner. The only
artifact is AES-GCM encrypted with a separate test key; logs contain metrics.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

MODEL = "Qwen/Qwen3-ASR-1.7B"
REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
WINDOWS = [(62.022, 76.128), (176.294, 187.328),
           (209.926, 217.824), (511.110, 523.744)]
TERMS = "数值算法与案例分析。术语：良态问题、病态问题、扰动、希尔伯特矩阵、Hilbert矩阵、逆矩阵、条件数、delta、范数。"


def recognition_context(course_title=None):
    if course_title:
        from src.ai.course_glossary import course_terms
        terms = course_terms(course_title)
        return '术语：' + '、'.join(terms) if terms else ''
    # New lecture topics are unknown; do not reuse previous lecture's hotwords.
    return '' if os.environ.get('LATEST_LECTURE') == 'true' else TERMS


def review_hotwords(report, evidence):
    """Reuse the selected course's hints, never the old numerical test terms."""
    selection = report.get('selection') or evidence.get('selection') or {}
    title = selection.get('course_title')
    if title:
        from src.ai.course_glossary import course_terms
        return course_terms(title)
    return [] if os.environ.get('LATEST_LECTURE') == 'true' else TERMS.split('术语：')[-1].rstrip('。').split('、')


def latest_request(detail, request, today=None):
    """Latest listed non-future lecture, NOT gated on playback_status.

    If its playback cannot be resolved, fail rather than silently test older audio.
    """
    today = today or datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    candidates = []
    for lecture in detail.get('lectures', []):
        date = str(lecture.get('date', ''))
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            continue
        if date > today or not str(lecture.get('sub_id', '')).isdigit():
            continue
        match = re.search(r'第\s*(\d+)', str(lecture.get('sub_title', '')))
        candidates.append((date, int(match.group(1)) if match else -1,
                           int(lecture['sub_id']), lecture))
    if not candidates:
        raise ValueError('No dated non-future lecture available')
    lecture = max(candidates, key=lambda item: item[:3])[3]
    return {**request, 'sub_id': str(lecture['sub_id']), 'offset': 0,
            'duration': 1800, 'selection': {'course_id': request.get('course_id'), 'course_title': detail.get('title'),
             'sub_title': lecture.get('sub_title'), 'date': lecture['date'],
             'sub_id': str(lecture['sub_id']), 'offset': 0, 'duration': 1800}}


def workspace() -> Path:
    root = Path(os.environ["RUNNER_TEMP"]) / "qwen-benchmark"
    root.mkdir(mode=0o700, exist_ok=True)
    return root


def parse_request(raw: str) -> dict:
    value = json.loads(raw)
    for field in ("course_id", "sub_id"):
        if not str(value.get(field, "")).isdigit():
            raise ValueError("Invalid private course selection")
    offset, duration = float(value["offset"]), float(value["duration"])
    if not (0 <= offset <= 6 * 3600 and 0 < duration <= 600):
        raise ValueError("Test slice must be at most 600 seconds")
    return {"course_id": str(value["course_id"]), "sub_id": str(value["sub_id"]),
            "offset": offset, "duration": duration}


def selected_test_request():
    """Two immutable private selections for one parallel test workflow."""
    slot = os.environ.get('TEST_COURSE_SLOT', '-1')
    if slot == '-1':
        return parse_request(os.environ['QWEN_ASR_TEST_REQUEST'])
    if slot not in ('0', '1'):
        raise ValueError('Invalid parallel test slot')
    requests = json.loads(os.environ['QWEN_ASR_TEST_REQUESTS'])
    if not isinstance(requests, list) or len(requests) != 2:
        raise ValueError('Parallel test requires exactly two private requests')
    parsed = [parse_request(json.dumps(request)) for request in requests]
    if parsed[0]['course_id'] == parsed[1]['course_id']:
        raise ValueError('Parallel test courses must be distinct')
    return parsed[int(slot)]


def sample_seconds() -> int:
    value=os.environ.get('SAMPLE_MINUTES','10')
    if value not in ('10','30'):
        raise ValueError('Sample length must be 10 or 30 minutes')
    if value=='30' and os.environ.get('LONG_CHUNK_SAMPLE')!='true':
        raise ValueError('30-minute samples require the isolated long-chunk mode')
    return int(value)*60


def auth_phase(url: str) -> str:
    """Only allowlisted step names, never URL/token/account data in logs."""
    path = urlparse(url).path.rstrip("/")
    for suffix, label in (("/authenticate", "auth_context"),
                          ("/queryAuthMethods", "auth_methods"),
                          ("/getJsPublicKey", "public_key"),
                          ("/authExecute", "credential_exchange"),
                          ("/authnEngine", "cas_ticket"),
                          ("/casapi/index.php", "icourse_cas"),
                          ("/infosimple", "verify_icourse")):
        if path.endswith(suffix):
            return label
    return "portal_or_redirect"


def configure_auth_session(session, events):
    """Benchmark-only timeout policy; do not replay credential POSTs or tickets."""
    original = session.request

    def request(method, url, **kwargs):
        timeout = kwargs.get("timeout", 60)
        read_timeout = timeout[1] if isinstance(timeout, tuple) else timeout
        kwargs["timeout"] = (15, max(20, min(90, read_timeout * 1.5)))
        event = {"phase": auth_phase(url), "method": method.upper()}
        events.append(event)
        began = time.perf_counter()
        try:
            response = original(method, url, **kwargs)
            event["status"] = response.status_code
            return response
        except Exception as error:
            event["error_type"] = type(error).__name__
            raise
        finally:
            event["seconds"] = round(time.perf_counter() - began, 2)

    session.request = request


def fetch() -> None:
    from src.api.webvpn import WebVPNSession
    from src.api.icourse import ICourseClient
    request = selected_test_request()
    if os.environ.get('LONG_CHUNK_SAMPLE')=='true':
        request['duration']=sample_seconds()
    print("Acquiring one privately selected authorized audio slice", flush=True)
    failures = []
    for attempt in range(5):
        print(f"Authentication attempt {attempt + 1}/5", flush=True)
        private_log = io.StringIO()
        events = []
        vpn = WebVPNSession()
        configure_auth_session(vpn.session, events)
        try:
            with contextlib.redirect_stdout(private_log), contextlib.redirect_stderr(private_log):
                if not vpn.login() or not vpn.authenticate_icourse():
                    raise RuntimeError("Authentication did not complete")
            break
        except Exception as error:
            phase = events[-1]["phase"] if events else "initialization"
            failures.append({"attempt": attempt + 1, "events": events,
                             "error_type": type(error).__name__,
                             "private_error": str(error)[-2000:],
                             "private_log": private_log.getvalue()[-6000:]})
            save_encrypted({"stage": "authentication", "attempts": failures})
            print(f"Authentication attempt failed ({type(error).__name__}, phase={phase})", flush=True)
            vpn.session.close()
            if attempt == 4:
                raise
            time.sleep((15, 30, 60, 90)[attempt])
    print("Authentication complete; resolving selected playback", flush=True)
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        client = ICourseClient(vpn)
        if os.environ.get('LATEST_LECTURE') == 'true':
            request = latest_request(client.get_course_detail(request['course_id']), request)
        if os.environ.get('FULL_LECTURE') == 'true':
            request['offset'], request['duration'] = 0, 10800
            if request.get('selection'):
                request['selection'].update(offset=0, duration=10800)
        url = client.get_video_url(request["course_id"], request["sub_id"])
        if not url:
            raise RuntimeError("No playback available")
        media, headers = client.get_stream_params(url)
        # Input-side seeking: pull media index and the selected interval only.
        full = os.environ.get("FULL_LECTURE") == "true"
        process = subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-headers", headers, "-ss", str(0 if full else request["offset"]), "-i", media,
            "-t", str(10800 if full else request["duration"]), "-vn", "-ac", "1", "-ar", "16000",
            "-y", str(workspace() / "audio.wav"),
        ], stdout=quiet, stderr=subprocess.PIPE,
           timeout=1200 if full or request['duration']>600 else 420)
        if process.returncode:
            save_encrypted({"stage": "audio_acquisition", "returncode": process.returncode,
                            "private_diagnostic": process.stderr.decode(errors="replace")[-8000:]})
            raise RuntimeError("Audio acquisition failed; encrypted diagnostic saved")
    print("Authorized slice acquisition completed; no audio artifact uploaded", flush=True)
    if os.environ.get("QUALITY_SAMPLE") == "true":
        fetch_evidence(client, request)
    else:
        (workspace()/'evidence.json').write_text(
            json.dumps({'selection': request.get('selection')}, ensure_ascii=False), encoding='utf-8')


def fetch_evidence(client, request):
    """Optional bounded local OCR; no screenshot URLs stored in the report."""
    from src.ai.ocr import ocr_image_text
    from src.api.icourse import fetch_ppt_image
    from scripts.qwen_quality import usable_ppt
    evidence = {"official_subtitles": [], "ppt": [], "unavailable": [],
                "selection": request.get('selection')}
    offset, stop = request['offset'], request['offset'] + request['duration']
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        try:
            subtitles=client.get_transcript_segments(request['sub_id'])
            for row in subtitles or []:
                if row['end_ms']>offset*1000 and row['start_ms']<stop*1000:
                    evidence['official_subtitles'].append({
                        'start':row['start_ms']/1000-offset,'end':row['end_ms']/1000-offset,
                        'text':str(row['text'])[:500]})
            evidence['official_subtitles']=evidence['official_subtitles'][:600]
            if subtitles is None:
                evidence['unavailable'].append('official_subtitles')
        except Exception:
            evidence['unavailable'].append('official_subtitles')
        try:
            pages=client.get_ppt_list(request['course_id'],request['sub_id'])
            preceding=[p for p in pages if p['created_sec']<=offset]
            chosen=[p for p in preceding[-1:] if p['created_sec']>=offset-300]+[p for p in pages if offset<p['created_sec']<stop]
            for page in chosen[:6]:
                try:
                    image=fetch_ppt_image(client,page,max_attempts=1,timeout=30)
                    if not image or len(image)>8*1024*1024:
                        evidence['unavailable'].append('ppt_page')
                        continue
                    text=ocr_image_text(image)[:1000]
                    if usable_ppt(text,page['created_sec']-offset):
                        evidence['ppt'].append({'start':page['created_sec']-offset,'text':text})
                    else:
                        evidence['unavailable'].append('ppt_filtered')
                except Exception:
                    evidence['unavailable'].append('ppt_page')
        except Exception:
            evidence['unavailable'].append('ppt')
    (workspace()/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False),encoding='utf-8')
    print(f"Reference materials: subtitles={len(evidence['official_subtitles'])}, OCR_pages={len(evidence['ppt'])}",flush=True)


def save_encrypted(report: dict) -> None:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key = base64.b64decode(os.environ["QWEN_ASR_TEST_KEY"], validate=True)
    if len(key) != 32:
        raise ValueError("Invalid test encryption key")
    nonce = os.urandom(12)
    raw = json.dumps(report, ensure_ascii=False).encode()
    (workspace() / "result.enc").write_bytes(b"QASR1" + nonce + AESGCM(key).encrypt(nonce, raw, b"qwen-asr-benchmark-v1"))


def infer() -> None:
    if os.environ.get('RUNTIME_SAMPLE') == 'true':
        infer_runtime_sample()
        return
    if os.environ.get("FULL_LECTURE") == "true" or os.environ.get("LONG_CHUNK_SAMPLE") == "true":
        infer_lecture()
        return
    import soundfile as sf
    import torch
    import sherpa_onnx
    from huggingface_hub import snapshot_download
    from qwen_asr import Qwen3ASRModel
    public_sample = os.environ.get("PUBLIC_SAMPLE") == "true"
    if public_sample:
        import requests
        response = requests.get("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav", timeout=60)
        response.raise_for_status()
        (workspace() / "audio.wav").write_bytes(response.content)
        print("Using Qwen official public Mandarin sample; NOT a classroom quality test", flush=True)
    audio, sr = sf.read(workspace() / "audio.wav", dtype="float32")
    if sr != 16000 or audio.ndim != 1 or (not public_sample and not 590 <= len(audio) / sr <= 600.1):
        raise ValueError("Audio duration or format does not match the 10-minute test")
    windows = [(0, len(audio) / sr)] if public_sample else WINDOWS
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    report = {"model": MODEL, "revision": REVISION, "torch": torch.__version__,
              "device": "cpu", "threads": 4, "dtype": "float32", "clips": [],
              "audio_seconds": len(audio) / sr, "full_chunks": []}
    report["source"] = "official_public_sample" if public_sample else "authorized_classroom_slice"
    save_encrypted(report)  # Check encryption before expensive inference.
    print(f"CPU threads=4; available RAM={os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024**3:.2f} GiB", flush=True)
    began = time.perf_counter()
    model_path = snapshot_download(MODEL, revision=REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt"])
    model = Qwen3ASRModel.from_pretrained(model_path, dtype=torch.float32,
        device_map="cpu", attn_implementation="eager", max_inference_batch_size=1,
        max_new_tokens=512)
    report["load_including_download_seconds"] = time.perf_counter() - began
    print("Qwen 1.7B loaded on CPU", flush=True)
    for hinted in (False, True):
        for index, (start, end) in enumerate(windows):
            began = time.perf_counter()
            result = model.transcribe(audio=(audio[round(start * sr):round(end * sr)], sr),
                context=TERMS if hinted else "", language="Chinese")[0]
            elapsed = time.perf_counter() - began
            report["clips"].append({"backend": "Qwen3-ASR-1.7B", "hinted": hinted,
                "start": start, "end": end, "text": result.text, "seconds": elapsed})
            report["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            save_encrypted(report)
            print(f"Qwen clip={index + 1}, hinted={hinted}, seconds={elapsed:.3f}, peak_RSS={report['peak_rss_gib']:.2f} GiB", flush=True)
    timed = [row for row in report["clips"] if row["backend"] == "Qwen3-ASR-1.7B" and row["hinted"]]
    ratio = sum(row["seconds"] for row in timed) / sum(row["end"] - row["start"] for row in timed)
    report["selected_clips_rtf"] = ratio
    # Avoid turning a CPU viability test into an unbounded full-course job.
    if ratio <= 3 and not public_sample:
        for start in range(0, 600, 30):
            end = min(start + 30, len(audio) / sr)
            began = time.perf_counter()
            result = model.transcribe(audio=(audio[round(start * sr):round(end * sr)], sr),
                context=TERMS, language="Chinese")[0]
            elapsed = time.perf_counter() - began
            report["full_chunks"].append({"start": start, "end": end, "text": result.text, "seconds": elapsed})
            save_encrypted(report)
            print(f"Full slice chunk={start // 30 + 1}/20, seconds={elapsed:.3f}", flush=True)
        report["full_slice_seconds"] = sum(row["seconds"] for row in report["full_chunks"])
    elif not public_sample:
        report["full_slice_skipped"] = "Selected clips took over 3x real time on CPU"
        print("Skipping full slice: selected-clip CPU real-time factor exceeds 3", flush=True)
    else:
        report["full_slice_skipped"] = "Public smoke sample only; no classroom slice used"
    report["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    save_encrypted(report)
    print(f"Benchmark completed: selected RTF={ratio:.3f}, peak RSS={report['peak_rss_gib']:.2f} GiB", flush=True)


def infer_runtime_sample():
    """Actual production recognizer, public smoke or isolated full classroom."""
    import soundfile as sf
    from types import SimpleNamespace
    from src.ai.transcriber import Transcriber
    public = os.environ.get('PUBLIC_SAMPLE') == 'true'
    if not public and os.environ.get('FULL_LECTURE') != 'true':
        raise ValueError('Private runtime tests require the full-lecture mode')
    wav=workspace()/'audio.wav'
    if public:
        import requests
        response=requests.get('https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav',timeout=60)
        response.raise_for_status()
        wav.write_bytes(response.content)
    info=sf.info(wav)
    if info.samplerate != 16000 or info.channels != 1 or info.duration <= 0:
        raise ValueError('Runtime audio must be nonempty mono 16kHz')
    evidence_path=workspace()/'evidence.json'
    evidence=json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
    selection=evidence.get('selection') or {}
    raw=workspace()/'audio.raw'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(wav),'-f','f32le','-ac','1','-ar','16000','-y',str(raw)],
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True,timeout=60)
    transcriber=Transcriber()
    from src.ai.course_glossary import course_terms
    terms=course_terms(selection.get('course_title', ''))
    transcriber.set_terms(terms)
    began=time.perf_counter()
    report={'source':'official_public_runtime_smoke' if public else 'authorized_full_runtime',
            'model':MODEL,'revision':REVISION,'complete':False,'selection':selection,
            'recognition_terms':terms,'reference_evidence':evidence,
            'acquisition_limit_reached':not public and info.duration >= 10800}
    save_encrypted(report)  # Validate encryption before expensive inference.
    try:
        transcript,segments=transcriber.transcribe_tail(str(raw),SimpleNamespace(poll=lambda:0,returncode=0),[])
        if not transcript.strip(): raise ValueError('Production runtime transcript empty')
        if report['acquisition_limit_reached']:
            raise ValueError('Audio reached the full-lecture acquisition cap')
        report.update(transcript=transcript,segments=segments,complete=True)
    except Exception as error:
        report['error_type']=type(error).__name__
        raise
    finally:
        report.update(full_chunks=transcriber.last_chunks,vad_windows=transcriber.last_vad_windows,
                      audio_seconds=transcriber.last_audio_duration,seconds=time.perf_counter()-began,
                      peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        save_encrypted(report)
        transcriber.release_model()
    print(f"Production Qwen runtime passed: audio={report['audio_seconds']:.1f}s, runtime={report['seconds']:.1f}s, peak={report['peak_rss_gib']:.2f}GiB",flush=True)


def infer_lecture() -> None:
    """One authorized lecture, serial CPU inference, VAD and encrypted checkpoints."""
    import gc
    import soundfile as sf
    import torch
    import sherpa_onnx
    from huggingface_hub import snapshot_download
    from qwen_asr import Qwen3ASRModel
    from scripts.qwen_segmentation import plan_long_chunks, join_chunk_text
    from scripts.qwen_quality import context_echo, low_information, bounded_retry
    from transformers import StoppingCriteriaList

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    path = workspace() / "audio.wav"
    info = sf.info(path)
    if info.samplerate != 16000 or info.channels != 1 or not 0 < info.duration <= 10800.1:
        raise ValueError("Invalid full lecture audio")
    sample_only = os.environ.get("LONG_CHUNK_SAMPLE") == "true"
    expected=sample_seconds() if sample_only else None
    if sample_only and not expected-10 <= info.duration <= expected+0.1:
        raise ValueError("Audio duration does not match the bounded sample length")
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(Path(os.environ["RUNNER_TEMP"]) / "silero_vad.onnx")
    config.silero_vad.threshold = 0.5
    config.silero_vad.min_silence_duration = 0.8
    config.silero_vad.min_speech_duration = 0.25
    config.silero_vad.max_speech_duration = 28.0
    config.sample_rate = 16000
    config.num_threads = 1
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)
    windows = []

    def drain():
        while not vad.empty():
            segment = vad.front
            windows.append((segment.start / 16000,
                            min(info.duration, (segment.start + len(segment.samples)) / 16000)))
            vad.pop()

    began = time.perf_counter()
    for block in sf.blocks(path, blocksize=512, dtype="float32"):
        vad.accept_waveform(block)
        drain()
    vad.flush()
    drain()
    chunks = plan_long_chunks(windows, info.duration)
    # Compute union duration, not summed overlapping windows.
    union_end, included = 0.0, 0.0
    for start, end in chunks:
        included += max(0, end - max(start, union_end))
        union_end = max(union_end, end)
    report = {"model": MODEL, "revision": REVISION,
              "source": "authorized_long_chunk_slice" if sample_only else "authorized_full_lecture",
              "chunk_target_seconds": 120, "max_new_tokens": 2048,
              "requested_sample_seconds": expected,
              "audio_seconds": info.duration, "vad_seconds": time.perf_counter() - began,
              "acquisition_limit_reached": info.duration >= 10800,
              "included_audio_seconds": included, "skipped_audio_seconds": info.duration - included,
              "vad_windows": windows, "planned_chunks": chunks, "full_chunks": [],
              "complete": False, "device": "cpu", "dtype": "float32", "threads": 4}
    if (workspace() / 'evidence.json').exists():
        report['selection'] = json.loads((workspace() / 'evidence.json').read_text()).get('selection')
    context = recognition_context((report.get('selection') or {}).get('course_title'))
    report['recognition_context'] = context
    save_encrypted(report)
    del vad
    gc.collect()
    print(f"VAD completed: audio={info.duration:.1f}s, chunks={len(chunks)}, skipped={info.duration-included:.1f}s", flush=True)
    if not chunks:
        report["complete"] = True
        report["transcript"] = ""
        save_encrypted(report)
        return
    began = time.perf_counter()
    model_path = snapshot_download(MODEL, revision=REVISION,
                                  allow_patterns=["*.json", "*.safetensors", "*.txt"])
    model = Qwen3ASRModel.from_pretrained(model_path, dtype=torch.float32,
        device_map="cpu", attn_implementation="eager", max_inference_batch_size=1,
        max_new_tokens=2048)
    report["load_including_download_seconds"] = time.perf_counter() - began
    with sf.SoundFile(path) as source:
        for index, (start, end) in enumerate(chunks):
            source.seek(round(start * 16000))
            samples = source.read(round(end * 16000) - round(start * 16000), dtype="float32")
            began = time.perf_counter()
            for attempt in range(1 if sample_only else 2):
                try:
                    with torch.inference_mode():
                        result = model.transcribe(audio=(samples, 16000), context=context, language="Chinese")[0]
                    break
                except Exception:
                    if sample_only or attempt == 1:
                        report["failed_chunk"] = index
                        save_encrypted(report)
                        raise
                    gc.collect()
            row = {"start": start, "end": end, "text": result.text,
                   "seconds": time.perf_counter() - began,
                   "attempts": attempt + 1,
                   "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
            row['initial_seconds'] = row['seconds']
            token_limit = 2048
            row["context_echo"] = context_echo(result.text, context)
            if os.environ.get("QUALITY_SAMPLE") == "true" and row['context_echo']:
                row['original_text'] = row['text']
                retry_start=time.perf_counter()
                with bounded_retry(model, StoppingCriteriaList) as retry, torch.inference_mode():
                    result=model.transcribe(audio=(samples,16000),context='',language='Chinese')[0]
                token_limit = 256
                row['text']=result.text
                row['unhinted_retry_seconds']=time.perf_counter()-retry_start
                row['retry_timed_out'] = retry['timed_out'] or row['unhinted_retry_seconds'] >= 60
                row['context_echo']=context_echo(result.text,context)
                row['quality_state']='unresolved_context_echo' if row['context_echo'] else 'unhinted_retry'
            row['text_token_count'] = len(model.processor.tokenizer.encode(result.text, add_special_tokens=False))
            row['possible_truncation'] = row['text_token_count'] >= token_limit - 8
            row['low_information'] = low_information(row['text'])
            # Keep rejected observations encrypted, but exclude them from review/summary.
            if row.get('retry_timed_out') or row['context_echo'] or row['low_information'] or (token_limit == 256 and row['possible_truncation']):
                row['rejected_text'] = row['text']
                row['quality_state'] = ('retry_timeout' if row.get('retry_timed_out') else
                                        'unresolved_context_echo' if row['context_echo'] else
                                        'low_information' if row['low_information'] else 'retry_token_limit')
                row['text'] = ''
            row['seconds'] = time.perf_counter() - began
            # Linux current RSS distinguishes retained allocations from the peak.
            row["rss_gib"] = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024**3
            report["full_chunks"].append(row)
            save_encrypted(report)
            print(f"Lecture chunk={index+1}/{len(chunks)}, seconds={row['seconds']:.2f}, RSS={row['rss_gib']:.2f}GiB", flush=True)
            del samples, result
            gc.collect()
    report["transcript"] = join_chunk_text(report["full_chunks"])
    report["full_slice_seconds"] = sum(row["seconds"] for row in report["full_chunks"])
    report["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    report["complete"] = info.duration < 10800
    save_encrypted(report)
    print("Full lecture completed; private text is encrypted only", flush=True)


def quality_review():
    """One review request, at most 120s cloud audio; keep variants, no DB writes."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from openai import OpenAI
    from scripts.qwen_quality import review_quality
    from scripts.qwen_audio_alignment import align_suspects
    from src.ai.doubao_asr import rescue_intervals_pcm
    d=(workspace()/'result.enc').read_bytes()
    key=base64.b64decode(os.environ['QWEN_ASR_TEST_KEY'],validate=True)
    report=json.loads(AESGCM(key).decrypt(d[5:17],d[17:],b'qwen-asr-benchmark-v1'))
    if os.environ.get('REUSE_RUN_ID'):
        import soundfile as sf
        if (report.get('source')!='authorized_long_chunk_slice' or not report.get('complete')
                or abs(sf.info(workspace()/'audio.wav').duration-report.get('audio_seconds',0))>0.1):
            raise ValueError('Cached slice does not match test mode or duration')
        report['cached_asr_source_run']=os.environ['REUSE_RUN_ID']
        for field in ('cloud_review','rescue_comparisons','localization','alignment_items'):
            report.pop(field,None)
    evidence=json.loads((workspace()/'evidence.json').read_text())
    report['reference_evidence']=evidence
    # Cached encrypted ASR + selected quotes allow testing the failing suffix
    # without paying for another full local transcription or review request.
    selected=report.get('review_suspects') if os.environ.get('REUSE_RUN_ID') else None
    full = os.environ.get('FULL_LECTURE') == 'true'
    budget = 600 if full else 120
    report['quality_limits'] = {'cloud_seconds':budget, 'max_suspects':12 if full else 4}
    if selected is None:
        selected=review_quality(OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'],base_url='https://api.deepseek.com/v1'),
                                'deepseek-v4-flash',report,evidence,
                                max_suspects=12 if full else 4, input_budget=96000 if full else 30000)
    report['review_suspects']=selected
    save_encrypted(report)
    intervals,located,unresolved,metrics=align_suspects(report,selected,workspace()/'audio.wav',save_encrypted,budget=budget)
    report['audio_alignment_metrics']=metrics
    report['localization']={'located':located,'unresolved':unresolved}
    save_encrypted(report)
    raw=workspace()/'audio.raw'
    hotwords=review_hotwords(report,evidence)
    if intervals:
        subprocess.run(['ffmpeg','-v','error','-i',str(workspace()/'audio.wav'),
                        '-f','f32le','-ac','1','-ar','16000','-y',str(raw)],
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True,timeout=60)
        rescues,attempted,failed=rescue_intervals_pcm(str(raw),os.environ.get('DOUBAO_ASR_API_KEY',''),
                                                   intervals,max_seconds=budget,max_clips=12 if full else 10,
                                                   hotwords=hotwords)
    else:
        rescues,attempted,failed=[],0,False
    report['cloud_review']={'rescues':rescues,'attempted_audio_seconds':attempted,'failed':failed,
                            'hotword_hints_enabled':bool(hotwords),
                            'selection_limit':'Whole-chunk audio forced alignment locates quoted text; not a correctness guarantee. Unresolved locations are not uploaded. Originals retained.'}
    report['rescue_comparisons']=[{'chunk_id':w['chunk_id'],'original_quote':w['text'],
        'cloud_text':' '.join(s['text'] for s in segments),
        'quote_start_ms':w['quote_start_ms'],'quote_end_ms':w['quote_end_ms'],
        'returned_speech_overlaps_quote':any(s['start_ms']<w['quote_end_ms'] and w['quote_start_ms']<s['end_ms'] for s in segments),
        'state':'variants_require_review' if segments else 'no_cloud_text'} for w,segments in rescues]
    save_encrypted(report)
    print(f"Quality check: suspects={len(selected)}, located={len(located)}, unresolved={len(unresolved)}, cloud_audio={attempted:.1f}s, cloud_failed={failed}",flush=True)


def summary_material(report, evidence):
    """Keep cloud variants separate; never blindly replace the local transcript."""
    if not report.get('complete') or not report.get('transcript', '').strip():
        raise ValueError('Cannot summarize incomplete or empty transcription')
    material = '本地 ASR 正文（按录音顺序）：\n' + report['transcript']
    pages = [p for p in evidence.get('ppt', []) if p.get('text')]
    material += '\n\nPPT OCR 辅助材料：\n' + json.dumps(pages, ensure_ascii=False)
    material += ('\n\n局部云端复核（只是另一识别版本，不保证正确；只可结合上下文判断，'
                 '不可靠的公式不要补成确定结论）：\n'
                 + json.dumps(report.get('rescue_comparisons', []), ensure_ascii=False))
    material += ('\n\n未解决的定位疑点：\n'
                 + json.dumps(report.get('localization', {}).get('unresolved', []), ensure_ascii=False))
    rejected = [{'start':r['start'],'end':r['end'],'state':r.get('quality_state')}
                for r in report.get('full_chunks', []) if not r.get('text')]
    material += ('\n\n被排除的空白/低信息/超时片段（不能据此编造缺失内容）：\n'
                 + json.dumps(rejected, ensure_ascii=False))
    if len(material) > 96000:
        raise ValueError('Summary material exceeds single-call test budget')
    return material


def record_summary_response(report, response, elapsed):
    """Save partial output/finish metadata BEFORE validating completeness."""
    choice = response.choices[0] if response.choices else None
    content = choice.message.content if choice else None
    diagnostic = {'finish_reason':choice.finish_reason if choice else None,
                  'markdown':content or '', 'seconds':elapsed,
                  'usage':response.usage.model_dump() if response.usage else None,
                  'reasoning_chars':len(getattr(choice.message,'reasoning_content',None) or '') if choice else 0}
    report.setdefault('summary_attempts', []).append(diagnostic)
    complete = bool(choice and choice.finish_reason == 'stop' and content and content.strip())
    if complete:
        report['test_summary'] = {**diagnostic, 'model':'deepseek-v4-flash',
                                  'production_written':False, 'email_sent':False}
    return complete


def generate_summary():
    """One explicit summary request; encrypted output only, no production pipeline."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from openai import OpenAI
    from src.ai.summarizer import load_system_prompt
    data = (workspace() / 'result.enc').read_bytes()
    key = base64.b64decode(os.environ['QWEN_ASR_TEST_KEY'], validate=True)
    report = json.loads(AESGCM(key).decrypt(data[5:17], data[17:], b'qwen-asr-benchmark-v1'))
    evidence_path = workspace() / 'evidence.json'
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else report.get('reference_evidence', {})
    material = summary_material(report, evidence)
    selection = report.get('selection') or {}
    automatic = os.environ.get('AUTO_COURSE_TERMS') == 'true'
    system = load_system_prompt()
    user = f"课程：{selection.get('course_title','数值算法与案例分析')}\n课次：{selection.get('sub_title','')}\n<course_material>\n" + material + '\n</course_material>'
    sources = {}
    if automatic:
        from src.ai.automatic_glossary import INSTRUCTION, validated_keywords
        sources = {'asr':[report['transcript']],
                   'ppt':[p['text'] for p in evidence.get('ppt',[]) if p.get('text')],
                   'cloud':[c['cloud_text'] for c in report.get('rescue_comparisons',[]) if c.get('cloud_text')]}
        system += '\n\n' + INSTRUCTION
        user = json.dumps({'course':selection.get('course_title'),'material':material,
                           'evidence_sources':{'asr':'material中的本地 ASR 正文',
                              'ppt':'material中的PPT OCR辅助材料',
                              'cloud':'material中的局部云端复核cloud_text字段'}},ensure_ascii=False)
    client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url='https://api.deepseek.com',
                    max_retries=0, timeout=600)
    began = time.perf_counter()
    try:
        response = client.chat.completions.create(model='deepseek-v4-flash', temperature=0.2,
        extra_body={'thinking':{'type':'enabled'}}, reasoning_effort='high',
        max_tokens=64000, messages=[{'role':'system','content':system},{'role':'user','content':user}],
        **({'response_format':{'type':'json_object'}} if automatic else {}))
    except Exception as error:
        report.setdefault('summary_attempts', []).append({'error_type':type(error).__name__,
            'seconds':time.perf_counter()-began, 'private_error':str(error)[-2000:]})
        save_encrypted(report)
        raise
    if automatic and response.choices:
        report['keyword_response'] = response.choices[0].message.content
        report['keyword_response_diagnostic'] = {
            'finish_reason':response.choices[0].finish_reason,
            'seconds':time.perf_counter()-began,
            'usage':response.usage.model_dump() if response.usage else None}
        save_encrypted(report)
        data=json.loads(response.choices[0].message.content)
        if not isinstance(data,dict) or not isinstance(data.get('summary'),str):
            raise ValueError('Invalid summary/keyword envelope')
        response.choices[0].message.content = data['summary']
        report['automatic_keywords'] = validated_keywords(data.get('keywords',[]),sources,data['summary'])
    complete = record_summary_response(report, response, time.perf_counter()-began)
    report['summary_attempts'][-1].update(thinking='enabled', reasoning_effort='high', max_tokens=64000)
    if complete:
        report['test_summary'].update(thinking='enabled', reasoning_effort='high', max_tokens=64000)
    save_encrypted(report)
    if not complete:
        print('Summary incomplete; encrypted partial output and finish metadata saved', flush=True)
        raise ValueError('Incomplete summary response')
    print('Isolated summary completed; text encrypted, no database or email writes', flush=True)


def clean() -> None:
    # Only this workflow's exact private outputs, never a broad temp directory.
    for name in ("audio.wav", "audio.raw", "evidence.json", "result.enc"):
        (workspace() / name).unlink(missing_ok=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) == 2 else "invalid"
    try:
        {"fetch": fetch, "infer": infer, "review": quality_review, "summary": generate_summary, "clean": clean}[mode]()
    except Exception as error:
        # Exception bodies and command arguments may contain private URLs.
        print(f"Benchmark {mode} failed ({type(error).__name__}); private details withheld", flush=True)
        raise SystemExit(1)
