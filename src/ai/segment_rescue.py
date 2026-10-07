"""Select only clearly weak local-ASR speech windows for cloud rescue.

Qwen does not expose calibrated confidence here. An empty or near-empty
result on a VAD-confirmed speech window is a conservative, observable proxy;
silent stretches are never sent to the cloud merely because the transcript is
sparse.  The budget is per lecture, not per retry or account.
"""

MAX_CLOUD_SECONDS = 15 * 60
MAX_CLOUD_CLIPS = 40


def cloud_budget_limits(profile='production'):
    """Whole-lecture defaults and the preserved historical experiment profile."""
    if profile == 'production':
        return MAX_CLOUD_SECONDS, MAX_CLOUD_CLIPS
    if profile == 'pilot15':
        return 15 * 60, 18
    raise ValueError('Unknown cloud budget profile')


MIN_EMPTY_SPEECH_SECONDS = 5
MIN_WEAK_SPEECH_SECONDS = 10


def select_weak_windows(windows: list[dict], audio_seconds: float,
                        *, max_seconds: int = MAX_CLOUD_SECONDS,
                        max_clips: int = MAX_CLOUD_CLIPS) -> list[dict]:
    """Return chronological, non-overlapping VAD windows within the cap."""
    candidates = []
    audio_end = int(max(0, audio_seconds) * 1000)
    for window in windows:
        start = max(0, int(window.get("start_ms", 0)))
        end = min(audio_end, int(window.get("end_ms", 0)))
        seconds = (end - start) / 1000
        text = str(window.get("text") or "").strip()
        if not text and seconds >= MIN_EMPTY_SPEECH_SECONDS:
            priority = 0
        elif len(text) <= 2 and seconds >= MIN_WEAK_SPEECH_SECONDS:
            priority = 1
        else:
            continue
        candidates.append((priority, -seconds, start, {
            "start_ms": start, "end_ms": end, "text": text,
        }))

    selected = []
    remaining_ms = max(0, max_seconds) * 1000
    for _, _, _, window in sorted(candidates):
        length = window["end_ms"] - window["start_ms"]
        if (len(selected) >= max_clips or length > remaining_ms
                or any(window["start_ms"] < old["end_ms"]
                       and old["start_ms"] < window["end_ms"]
                       for old in selected)):
            continue
        selected.append(window)
        remaining_ms -= length
    return sorted(selected, key=lambda item: item["start_ms"])


def merge_rescued_segments(local: list[dict], rescues: list[tuple[dict, list[dict]]]
                           ) -> list[dict]:
    """Replace only weak windows with nonempty cloud text; keep all others."""
    replacements = {
        (window["start_ms"], window["end_ms"]): segments
        for window, segments in rescues if segments
    }
    merged = [segment for segment in local
              if (segment["start_ms"], segment["end_ms"])
              not in replacements]
    for segments in replacements.values():
        merged.extend(segments)
    return sorted(merged, key=lambda item: (item["start_ms"], item["end_ms"]))
