"""Verify the configured Seed-ASR 2.0 API with synthetic audio only.

This script deliberately never touches iCourse, the encrypted database, or
the mailer. It prints only an outcome, never provider response bodies or keys.
"""

from __future__ import annotations

import math
import os
import struct
import tempfile

import requests

from src.ai.doubao_asr import (
    CloudASRError,
    SAMPLE_RATE,
    _encode_chunk,
    _recognize_chunk,
)


def main() -> None:
    api_key = os.environ.get("DOUBAO_ASR_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DOUBAO_ASR_API_KEY is not configured")
    duration = 3
    with tempfile.NamedTemporaryFile(suffix=".raw") as raw:
        # A quiet synthetic tone, not a recording or any private information.
        for index in range(SAMPLE_RATE * duration):
            sample = 0.04 * math.sin(2 * math.pi * 440 * index / SAMPLE_RATE)
            raw.write(struct.pack("<f", sample))
        raw.flush()
        encoded = _encode_chunk(raw.name, 0, duration)
    with requests.Session() as session:
        _recognize_chunk(encoded, api_key, 0, duration * 1000, session)
    print("Seed-ASR 2.0 API smoke check passed (synthetic audio only).")


if __name__ == "__main__":
    try:
        main()
    except CloudASRError:
        raise SystemExit(
            "Seed-ASR 2.0 API smoke check failed; verify API key and "
            "volc.seedasr.auc activation. No lecture was processed."
        )
