"""Seed-ASR 2.0 file transcription without publishing iCourse media URLs.

The downloader has already decoded the authorized recording to local 16 kHz
mono float32 PCM.  Send bounded MP3 chunks as base64 directly to Volcengine;
never hand its service a signed iCourse or WebVPN URL.
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
import time
import uuid

import requests


BASE_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel"
RESOURCE_ID = "volc.seedasr.auc"
SAMPLE_RATE = 16_000
BYTES_PER_SECOND = SAMPLE_RATE * 4  # float32 mono
CHUNK_SECONDS = 30 * 60
MAX_ENCODED_BYTES = 20 * 1024 * 1024


class CloudASRError(RuntimeError):
    """A cloud failure that can safely fall back to local recognition."""


class CloudAudioError(RuntimeError):
    """The downloaded audio is missing or incomplete; do not transcribe it."""


class CloudAudioIncompleteError(CloudAudioError):
    """ffmpeg exited successfully but captured less than half the media."""

    def __init__(self, actual: float, expected: float):
        super().__init__("audio download is incomplete")
        self.actual = actual
        self.expected = expected


def wait_for_complete_audio(path, process, stderr_chunks, timeout=7200,
                            expected_duration_s=None):
    """Return (actual, expected) after the complete ffmpeg download finishes."""
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise CloudAudioError("audio download timed out") from exc
    stderr = b"".join(stderr_chunks)
    if process.returncode != 0:
        if b"does not contain any stream" in stderr:
            raise CloudAudioError("video has no audio stream")
        raise CloudAudioError("audio download failed")
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise CloudAudioError("audio download produced no file") from exc
    if size < BYTES_PER_SECOND:
        raise CloudAudioError("audio download produced no usable audio")
    actual = size / BYTES_PER_SECOND
    match = re.search(rb"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    expected = float(expected_duration_s or 0)
    if not expected and match:
        h, m, s = match.groups()
        expected = int(h) * 3600 + int(m) * 60 + float(s)
    # Media and decoded audio timelines can disagree on iCourse recordings.
    # Only a severe shortfall blocks transcription; lesser discrepancies are
    # reported by the runner without discarding otherwise usable audio.
    if expected and actual / expected < 0.50:
        raise CloudAudioIncompleteError(actual, expected)
    return actual, expected


def _encode_chunk(path: str, start_s: int, duration_s: float) -> bytes:
    """Encode one local PCM interval; keep API bodies small and bounded."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", "16000",
             "-ac", "1", "-ss", str(start_s), "-i", path,
             "-t", str(duration_s), "-vn", "-codec:a", "libmp3lame",
             "-b:a", "48k", "-f", "mp3", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CloudASRError("MP3 encoding failed") from exc
    if result.returncode or not result.stdout:
        raise CloudASRError("MP3 encoding failed")
    if len(result.stdout) > MAX_ENCODED_BYTES:
        raise CloudASRError("encoded audio chunk exceeds upload limit")
    return result.stdout


def _recognize_chunk(audio: bytes, api_key: str, offset_ms: int,
                     duration_ms: int, session: requests.Session,
                     poll_timeout=300) -> list[dict]:
    task_id = str(uuid.uuid4())
    headers = {
        "X-Api-Key": api_key,
        "X-Api-Resource-Id": RESOURCE_ID,
        "X-Api-Request-Id": task_id,
        "X-Api-Sequence": "-1",
    }
    payload = {
        "user": {"uid": "icourse-private"},
        "audio": {
            "format": "mp3",
            "data": base64.b64encode(audio).decode("ascii"),
            "language": "zh-CN",
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_punc": True,
            "show_utterances": True,
        },
    }
    try:
        response = session.post(
            f"{BASE_URL}/submit", headers=headers, json=payload,
            timeout=(15, 120),
        )
        response.raise_for_status()
        if response.headers.get("X-Api-Status-Code") != "20000000":
            raise CloudASRError("cloud ASR submit rejected")
        deadline = time.monotonic() + poll_timeout
        while time.monotonic() < deadline:
            query = session.post(
                f"{BASE_URL}/query", headers=headers, json={},
                timeout=(15, 60),
            )
            query.raise_for_status()
            status = query.headers.get("X-Api-Status-Code")
            if status in ("20000001", "20000002"):
                time.sleep(5)
                continue
            if status == "20000003":
                return []
            if status != "20000000":
                raise CloudASRError("cloud ASR query failed")
            result = query.json().get("result") or {}
            utterances = result.get("utterances") or []
            segments = [
                {"start_ms": offset_ms + int(item["start_time"]),
                 "end_ms": offset_ms + int(item["end_time"]),
                 "text": str(item.get("text", "")).strip()}
                for item in utterances
                if isinstance(item, dict) and item.get("text")
                and "start_time" in item and "end_time" in item
            ]
            if segments:
                return segments
            text = str(result.get("text") or "").strip()
            return [{"start_ms": offset_ms,
                     "end_ms": offset_ms + duration_ms,
                     "text": text}] if text else []
        raise CloudASRError("cloud ASR query timed out")
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        # Never put response bodies, signed URLs, or token-bearing headers in
        # exceptions/logs; the runner reports only the exception class.
        raise CloudASRError("cloud ASR transport or response error") from exc


def transcribe_pcm(path: str, api_key: str, actual_duration_s: float,
                   *, session: requests.Session | None = None,
                   ) -> tuple[str, list[dict]]:
    """Recognize all audio in chronological chunks; no partial success."""
    if not api_key:
        raise CloudASRError("cloud ASR key not configured")
    segments: list[dict] = []
    own_session = session is None
    session = session or requests.Session()
    try:
        start = 0
        while start < actual_duration_s:
            duration = min(CHUNK_SECONDS, actual_duration_s - start)
            audio = _encode_chunk(path, start, duration)
            segments.extend(_recognize_chunk(
                audio, api_key, int(start * 1000), int(duration * 1000),
                session,
            ))
            start += CHUNK_SECONDS
    finally:
        if own_session:
            session.close()
    text = " ".join(s["text"] for s in segments).strip()
    if not text:
        raise CloudASRError("cloud ASR returned no speech")
    return text, segments
