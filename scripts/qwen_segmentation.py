"""Conservative, pause-aware chunk planning for the isolated Qwen test."""
import math


def plan_chunks(windows, duration, maximum=28.0, padding=1.0):
    """Keep VAD pauses; merge adjacent speech only within the size limit.

    Continuous speech gets bounded chunks with at most two seconds overlap.
    Quiet speech is not rejected with an amplitude threshold.
    """
    if not math.isfinite(duration) or duration <= 0 or maximum <= 0 or padding < 0:
        raise ValueError("Invalid chunk limits")
    merged = []
    for start, end in sorted(windows):
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
            raise ValueError("Invalid speech window")
        if merged and start - merged[-1][1] <= 0.8 and end - merged[-1][0] <= maximum:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    chunks = []
    for start, end in merged:
        cursor = start
        while cursor < end:
            stop = min(cursor + maximum, end)
            chunks.append((max(0, cursor - padding), min(duration, stop + padding)))
            cursor = stop
    return chunks


def join_chunk_text(rows):
    """Trim only exact substantial repeats across overlapping audio windows.

    Do not manufacture sentence timestamps; rows retain their audio intervals.
    Short repeated words may be deliberate, so keep them.
    """
    output = []
    previous = None
    for row in rows:
        text = row.get("text", "").strip()
        if previous and row["start"] < previous["end"]:
            prior = previous.get("text", "").strip()
            for size in range(min(len(prior), len(text), 120), 5, -1):
                if prior[-size:] == text[:size]:
                    text = text[size:]
                    break
        if text:
            output.append(text)
        previous = row
    return "\n".join(output)
