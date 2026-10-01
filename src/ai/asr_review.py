"""Bounded text-only review; the model can select existing speech IDs only."""

import json
from pathlib import Path

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "asr_review.md"
MAX_WINDOWS = 200
MAX_INPUT_CHARS = 30_000


def review_windows(client, model: str, windows: list[dict], ppt_pages: list[dict],
                   excluded: set[tuple[int, int]],
                   *, disable_thinking: bool = False,
                   terms: list[str] | None = None) -> list[dict]:
    # Keep neighbouring windows for context. Sample evenly only for very long
    # lectures; eligible IDs remain tied to original VAD windows.
    indices = list(range(len(windows)))
    if len(indices) > MAX_WINDOWS:
        indices = sorted({round(i * (len(windows) - 1) / (MAX_WINDOWS - 1))
                          for i in range(MAX_WINDOWS)})
    shown = {}
    rows = []
    for index in indices:
        window = windows[index]
        text = str(window.get("text") or "").strip()
        if not text:
            continue
        duration = window["end_ms"] - window["start_ms"]
        eligible = (1_000 <= duration <= 60_000
                    and (window["start_ms"], window["end_ms"]) not in excluded)
        rows.append([index, round(window["start_ms"] / 1000),
                     int(eligible), text[:80]])
        if eligible:
            shown[index] = window
    if not shown:
        return []
    ppt = "\n".join(str(page.get("text") or "") for page in ppt_pages)[:6_000]
    payload = json.dumps({"columns": ["id", "start_seconds", "eligible", "text"],
                          "segments": rows, "ppt": ppt,
                          "terminology_reference": (terms or [])[:30]}, ensure_ascii=False)
    if len(payload) > MAX_INPUT_CHARS:
        # No extra calls or truncated JSON when the budget is exceeded.
        return []
    options = {}
    if disable_thinking:
        options = {"extra_body": {"thinking": {"type": "disabled"}},
                   "response_format": {"type": "json_object"}}
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": PROMPT_PATH.read_text(encoding="utf-8")},
                  {"role": "user", "content": payload}],
        temperature=0,
        max_tokens=1_000,
        timeout=45,
        **options,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(raw)
    suspects = result.get("suspects", [])
    if not isinstance(suspects, list):
        return []
    selected = []
    seen = set()
    for item in suspects[:10]:
        if not isinstance(item, dict):
            continue
        index = item.get("id")
        if (type(index) is int and index in shown and index not in seen
                and isinstance(item.get("reason"), str) and item["reason"].strip()):
            selected.append(shown[index])
            seen.add(index)
    usage = getattr(response, "usage", None)
    if usage is not None:
        print(f"[ASR review] tokens: prompt={usage.prompt_tokens}, "
              f"completion={usage.completion_tokens}; suspects={len(selected)}")
    return selected
