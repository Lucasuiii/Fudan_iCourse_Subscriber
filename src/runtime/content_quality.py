"""Conservative quality gate for lectures with almost no usable material.

The gate deliberately classifies *evidence*, not the real-world cause.  A
silent recording with an unchanged classroom screen may mean that the teacher
was absent, but it can also mean a broken microphone or the wrong audio track.
Only the former is obvious to a human; the pipeline calls both cases
``no_content`` and never claims that a class was cancelled.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re


MIN_LONG_RECORDING_SECONDS = 20 * 60
MIN_TRANSCRIPT_CHARS = 120
MIN_TRANSCRIPT_CHARS_PER_MINUTE = 2.0
MAX_SPARSE_SEGMENTS_PER_MINUTE = 0.10
MIN_PPT_TEXT_CHARS = 80

_DIAGNOSTIC_RE = re.compile(r"\[注意：.*?\]", re.DOTALL)


@dataclass(frozen=True)
class ContentQualityDecision:
    """Result returned by :func:`assess_content_quality`.

    ``action`` is one of:

    - ``summarize``: at least one source contains enough material;
    - ``skip_no_content``: a complete, long recording has consistently empty
      audio and visual evidence;
    - ``retry``: material is too weak to summarize, but the evidence is not
      strong enough for a permanent no-content decision.
    """

    action: str
    transcript_chars: int
    transcript_chars_per_minute: float
    segment_count: int
    ppt_chars: int
    ppt_page_count: int


def _content_chars(text: str) -> int:
    """Count letters and numbers after removing pipeline diagnostics."""
    cleaned = _DIAGNOSTIC_RE.sub("", text or "")
    return sum(1 for char in cleaned if char.isalnum())


def assess_content_quality(
    *,
    transcript: str,
    segments: list[dict] | None,
    ppt_pages: list[dict] | None,
    transcript_source: str,
    actual_audio_seconds: float = 0,
    expected_audio_seconds: float = 0,
    ppt_failed_count: int = 0,
) -> ContentQualityDecision:
    """Decide whether an LLM has enough reliable lecture material.

    The terminal ``skip_no_content`` decision is intentionally narrow.  It is
    available only after a full local-ASR pass over a known-complete recording
    of at least 20 minutes.  Both the transcript and the usable PPT OCR must be
    sparse, and very few speech segments may have survived.  Any healthy
    source is enough to keep the normal summary path.
    """
    transcript_chars = _content_chars(transcript)
    segment_count = sum(
        1 for segment in (segments or [])
        if _content_chars(str(segment.get("text", "")))
    )
    ppt_pages = ppt_pages or []
    ppt_chars = sum(
        _content_chars(str(page.get("text", ""))) for page in ppt_pages
    )
    ppt_page_count = len(ppt_pages)

    duration = max(float(actual_audio_seconds or 0), 0.0)
    minutes = duration / 60 if duration else 0.0
    chars_per_minute = transcript_chars / minutes if minutes else 0.0

    decision_args = dict(
        transcript_chars=transcript_chars,
        transcript_chars_per_minute=chars_per_minute,
        segment_count=segment_count,
        ppt_chars=ppt_chars,
        ppt_page_count=ppt_page_count,
    )

    # Cached, official and hybrid transcripts do not carry a trustworthy full
    # local-ASR duration here. Their own completeness policy remains in charge.
    if transcript_source != "local_asr":
        return ContentQualityDecision("summarize", **decision_args)

    transcript_sparse = (
        transcript_chars < MIN_TRANSCRIPT_CHARS
        or chars_per_minute < MIN_TRANSCRIPT_CHARS_PER_MINUTE
    )
    ppt_sparse = ppt_chars < MIN_PPT_TEXT_CHARS
    if not transcript_sparse or not ppt_sparse:
        return ContentQualityDecision("summarize", **decision_args)

    # A short recording may be a legitimate announcement. Do not auto-skip it.
    if duration < MIN_LONG_RECORDING_SECONDS:
        return ContentQualityDecision("summarize", **decision_args)

    expected = max(float(expected_audio_seconds or 0), 0.0)
    audio_complete = expected > 0 and duration / expected >= 0.90
    sparse_segment_limit = max(
        3,
        math.ceil(minutes * MAX_SPARSE_SEGMENTS_PER_MINUTE),
    )
    segments_sparse = segment_count <= sparse_segment_limit

    # Failed PPT downloads/OCR and unknown audio completeness are technical
    # uncertainty, not evidence that no class took place. Retry instead.
    if ppt_failed_count or not audio_complete or not segments_sparse:
        return ContentQualityDecision("retry", **decision_args)

    return ContentQualityDecision("skip_no_content", **decision_args)
