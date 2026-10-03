"""Manual, isolated CPU benchmark. No database, summary API, or email calls.

Course selection is a Secret. Audio stays on the ephemeral runner. The only
artifact is AES-GCM encrypted with a separate test key; logs contain metrics.
"""
from __future__ import annotations

import base64
import contextlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

MODEL = "Qwen/Qwen3-ASR-1.7B"
REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
WINDOWS = [(62.022, 76.128), (176.294, 187.328),
           (209.926, 217.824), (511.110, 523.744)]
TERMS = "数值算法与案例分析。术语：良态问题、病态问题、扰动、希尔伯特矩阵、Hilbert矩阵、逆矩阵、条件数、delta、范数。"


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


def fetch() -> None:
    from src.api.webvpn import WebVPNSession
    from src.api.icourse import ICourseClient
    request = parse_request(os.environ["QWEN_ASR_TEST_REQUEST"])
    print("Acquiring one privately selected authorized audio slice", flush=True)
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        vpn = WebVPNSession()
        if not vpn.login() or not vpn.authenticate_icourse():
            raise RuntimeError("Authentication did not complete")
        client = ICourseClient(vpn)
        url = client.get_video_url(request["course_id"], request["sub_id"])
        if not url:
            raise RuntimeError("No playback available")
        media, headers = client.get_stream_params(url)
        # Input-side seeking: pull media index and the selected interval only.
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "quiet",
            "-headers", headers, "-ss", str(request["offset"]), "-i", media,
            "-t", str(request["duration"]), "-vn", "-ac", "1", "-ar", "16000",
            "-y", str(workspace() / "audio.wav"),
        ], stdout=quiet, stderr=quiet, timeout=420, check=True)
    print("Authorized slice acquisition completed; no audio artifact uploaded", flush=True)


def save_encrypted(report: dict) -> None:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key = base64.b64decode(os.environ["QWEN_ASR_TEST_KEY"], validate=True)
    if len(key) != 32:
        raise ValueError("Invalid test encryption key")
    nonce = os.urandom(12)
    raw = json.dumps(report, ensure_ascii=False).encode()
    (workspace() / "result.enc").write_bytes(b"QASR1" + nonce + AESGCM(key).encrypt(nonce, raw, b"qwen-asr-benchmark-v1"))


def infer() -> None:
    import soundfile as sf
    import torch
    import sherpa_onnx
    from huggingface_hub import snapshot_download
    from qwen_asr import Qwen3ASRModel
    audio, sr = sf.read(workspace() / "audio.wav", dtype="float32")
    if sr != 16000 or audio.ndim != 1 or not 590 <= len(audio) / sr <= 600.1:
        raise ValueError("Audio duration or format does not match the 10-minute test")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    report = {"model": MODEL, "revision": REVISION, "torch": torch.__version__,
              "device": "cpu", "threads": 4, "dtype": "float32", "clips": [],
              "audio_seconds": len(audio) / sr, "full_chunks": []}
    save_encrypted(report)  # Check encryption before expensive inference.
    print(f"CPU threads=4; available RAM={os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024**3:.2f} GiB", flush=True)
    sense_dir = Path(os.environ["RUNNER_TEMP"]) / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    baseline = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(sense_dir / "model.int8.onnx"), tokens=str(sense_dir / "tokens.txt"),
        num_threads=4, use_itn=True, debug=False)
    for index, (start, end) in enumerate(WINDOWS):
        began = time.perf_counter()
        stream = baseline.create_stream()
        stream.accept_waveform(sr, audio[round(start * sr):round(end * sr)])
        baseline.decode_stream(stream)
        report["clips"].append({"backend": "SenseVoiceSmall-int8", "start": start,
                                "end": end, "text": stream.result.text,
                                "seconds": time.perf_counter() - began})
    del baseline
    print("SenseVoice baseline finished on the same four selected clips", flush=True)
    began = time.perf_counter()
    model_path = snapshot_download(MODEL, revision=REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt"])
    model = Qwen3ASRModel.from_pretrained(model_path, dtype=torch.float32,
        device_map="cpu", attn_implementation="eager", max_inference_batch_size=1,
        max_new_tokens=512)
    report["load_including_download_seconds"] = time.perf_counter() - began
    print("Qwen 1.7B loaded on CPU", flush=True)
    for hinted in (False, True):
        for index, (start, end) in enumerate(WINDOWS):
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
    if ratio <= 3:
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
    else:
        report["full_slice_skipped"] = "Selected clips took over 3x real time on CPU"
        print("Skipping full slice: selected-clip CPU real-time factor exceeds 3", flush=True)
    report["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    save_encrypted(report)
    print(f"Benchmark completed: selected RTF={ratio:.3f}, peak RSS={report['peak_rss_gib']:.2f} GiB", flush=True)


def clean() -> None:
    # Only this workflow's exact private outputs, never a broad temp directory.
    for name in ("audio.wav", "result.enc"):
        (workspace() / name).unlink(missing_ok=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) == 2 else "invalid"
    try:
        {"fetch": fetch, "infer": infer, "clean": clean}[mode]()
    except Exception as error:
        # Exception bodies and command arguments may contain private URLs.
        print(f"Benchmark {mode} failed ({type(error).__name__}); private details withheld", flush=True)
        raise SystemExit(1)
