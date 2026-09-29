"""Pure policy helpers for selecting official subtitles versus local ASR."""

from __future__ import annotations


OFFICIAL_MAX_GAP_SECONDS = 20 * 60
OFFICIAL_MIN_CHARS_PER_MINUTE = 5
OFFICIAL_MAX_HYBRID_SHARE = 0.60


def assess_official_transcript(
    segments: list[dict] | None,
    duration_s: float = 0,
) -> tuple[str, list[tuple[float, float]]]:
    """Classify official subtitles as complete, hybrid-fillable, or unusable.

    ``duration_s`` should be the probed media duration when available. Without
    it we can still detect head/middle gaps, but cannot make a trustworthy tail
    decision. Hybrid ASR is limited to cases where long missing spans cover no
    more than 60% of the lecture; sparse/stub subtitles use full local ASR.
    """
    if not segments:
        return "full_asr", []

    ordered = sorted(segments, key=lambda s: int(s.get("start_ms", 0)))
    duration_s = max(float(duration_s or 0), 0.0)
    end_limit = duration_s or max(
        float(int(s.get("end_ms", 0))) / 1000 for s in ordered
    )

    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for seg in ordered:
        start = max(0.0, float(int(seg.get("start_ms", 0))) / 1000)
        end = max(start, float(int(seg.get("end_ms", 0))) / 1000)
        if start - cursor > OFFICIAL_MAX_GAP_SECONDS:
            gaps.append((cursor, min(start, end_limit)))
        cursor = max(cursor, end)
    if duration_s and duration_s - cursor > OFFICIAL_MAX_GAP_SECONDS:
        gaps.append((cursor, duration_s))

    # A few isolated rows can have no 20-minute hole while still being an
    # unusably sparse stub. This conservative density floor catches that case.
    if duration_s:
        visible_chars = sum(
            len("".join(str(s.get("text", "")).split())) for s in ordered
        )
        chars_per_minute = visible_chars / max(duration_s / 60, 1)
        if chars_per_minute < OFFICIAL_MIN_CHARS_PER_MINUTE:
            return "full_asr", []

    if not gaps:
        return "complete", []
    if duration_s and sum(end - start for start, end in gaps) <= (
        duration_s * OFFICIAL_MAX_HYBRID_SHARE
    ):
        return "hybrid", gaps
    return "full_asr", []


def merge_timed_segments(
    official: list[dict], asr_segments: list[dict]
) -> list[dict]:
    """Merge official and gap-ASR segments in timeline order."""
    return sorted(
        [*official, *asr_segments],
        key=lambda s: (int(s.get("start_ms", 0)), int(s.get("end_ms", 0))),
    )


def supplement_asr_gaps(
    asr_segments: list[dict], official: list[dict] | None,
    duration_s: float, min_gap_s: int = 120,
) -> list[dict]:
    """Use official text only inside long ASR holes, visibly marked as such.

    A whole official transcript never replaces ASR. In particular, a short
    announcement or a genuinely silent interval must not be overwritten by
    miscellaneous subtitles outside a substantial ASR gap.
    """
    if not asr_segments or not official or duration_s <= 0:
        return asr_segments
    if assess_official_transcript(official, duration_s)[0] != "complete":
        return asr_segments
    ordered = sorted(asr_segments, key=lambda s: int(s["start_ms"]))
    end_ms = int(duration_s * 1000)
    holes = []
    cursor = 0
    for segment in ordered:
        start = int(segment["start_ms"])
        if start - cursor >= min_gap_s * 1000:
            holes.append((cursor, start))
        cursor = max(cursor, int(segment["end_ms"]))
    if end_ms - cursor >= min_gap_s * 1000:
        holes.append((cursor, end_ms))
    supplements = []
    for start, end in holes:
        candidates = [s for s in official
                      if int(s["start_ms"]) >= start
                      and int(s["end_ms"]) <= end
                      and str(s.get("text", "")).strip()]
        if candidates:
            supplements.append({
                "start_ms": start, "end_ms": end,
                "text": "[官方字幕补充，可能有识别错误] "
                        + " ".join(str(s["text"]) for s in candidates),
            })
    return sorted([*ordered, *supplements], key=lambda s: int(s["start_ms"]))
