"""Qwen-only runtime facade, shared audio errors and legacy text utilities.

Transcriber aliases QwenTranscriber below. The recognizer does not use the
legacy text-cleanup utility: meaningful short mathematical tokens are retained.
"""

import os
import re
import subprocess
import threading
import time
from typing import Callable, Optional

from src.runtime import config


SAMPLE_RATE= 16000
WINDOW_SIZE = 512  # VAD window in samples (~32 ms at 16 kHz)
BYTES_PER_SAMPLE = 4  # float32
BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE
SILENCE_GAP_THRESHOLD_SEC = 30 * 60  # 30 min of no speech → suspected cutoff


# ── Shared resource meter state (network delta tracking) ──────────────────
_rm_net_last: tuple[float, int, int] | None = None


def _resource_meter(bound: str = "") -> str:
    """CPU + memory + network throughput suffix for progress lines.
    ``bound`` indicates the limiting resource (e.g. "cpu", "io").
    """
    global _rm_net_last
    try:
        import psutil
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        now = time.time()
        net = psutil.net_io_counters()
        if _rm_net_last is not None:
            dt = now - _rm_net_last[0]
            up = (net.bytes_sent - _rm_net_last[1]) / max(dt, 0.1) / 1024
            down = (net.bytes_recv - _rm_net_last[2]) / max(dt, 0.1) / 1024
            ns = f" down={down:.0f}KB/s"
        else:
            ns = ""
        _rm_net_last = (now, net.bytes_sent, net.bytes_recv)
        tag = f"[{bound.upper()}] " if bound else ""
        return f"  {tag}(cpu={cpu:.0f}% mem={mem:.0f}%{ns})"
    except Exception:
        return ""


# ── Generic per-segment text post-processing ─────────────────────────────
# Applied to every recognized segment regardless of backend.  The cleanups
# below address noise that the LLM otherwise has to spend tokens ignoring:
#
#   • SenseVoice's zh-en-ja-ko-yue model hallucinates kana / hangul tokens
#     in pure-Chinese audio ("うん", "あの", "그래") because the model was
#     trained to handle code-switching and biases toward emitting non-empty
#     output even for filler sounds.
#   • FireRed inserts ``<sil>`` between recognized chunks.
#   • Both backends transcribe occasional bilingual filler English ("Yeah",
#     "OK", "well") from bilingual classroom speech that adds nothing.
#
# What we do NOT touch: real technical English (CNN, FCN, YOLO, RGB, VGG…).
# The English filter is a fixed whitelist of fillers, so tech terms survive
# verbatim.  This is by design — anything subject-specific lives at the
# prompt / LLM layer, not here.
#
# Module-level compiled patterns so we don't recompile per call.

# Japanese hiragana + katakana + half-width katakana
_JP_NOISE_RE = re.compile(r"[぀-ゟ゠-ヿｦ-ﾟ]+")
# Korean: precomposed hangul syllables + jamo blocks
_KR_NOISE_RE = re.compile(r"[가-힯ᄀ-ᇿ㄰-㆏]+")
# English + Chinese filler / interjection words.  Word-bounded where
# possible so tech terms ("CNN", "RNN") are never touched.
#
#   [\s.。,，;；!！?？]*  — leading punctuation / whitespace (greedy)
#   \b(…)\b                 — filler word (word-bounded for ASCII)
#   [\s.。,，;；!！?？、。]+ — trailing punctuation sequence (at least 1)
#
# All three are removed together so we don't leave orphan punctuation
# behind.  For Chinese fillers that can't be word-bounded, we require
# at least ONE punctuation neighbour.
#
# CJK interjections — ONLY pure onomatopoeia / filler sounds, NOT sentence
# particles ("吗""呢""吧""嘛""呀" are meaningful and must stay).
_CN_FILLER_WORDS = r"嗯|呃|啊+|哦+|哈+|呵+|唉|哎|哟|嗨|喔+|噢+|啧|嘶|啧|唔+"
_EN_FILLER_WORDS = (
    r"yeah|yep|yup|ok|okay|uh+|um+|hmm+|ohh*|huh+|hey+|"
    r"you know|i mean|"
    r"the|it|its|that|this|these|those|"
    r"of|to|for|a|an|be|was|were|do|does|did|"
    r"just|only|very|really|actually"
)
# Pattern: filler word surrounded by punctuation on either side.
# Requires AT LEAST ONE punctuation neighbour (leading or trailing) so
# we don't touch fillers embedded in real content.  The entire match
# (optional leading punct + filler + mandatory trailing punct) is
# removed, so no orphan punctuation remains.
_FILLER_PUNCT_RE = re.compile(
    r"(?:"
    r"[\s.。,，;；!！?？、。]+"
    r"\b(?:" + _EN_FILLER_WORDS + r")\b"
    r"[\s.。,，;；!！?？、。]*"
    r"|"
    r"[\s.。,，;；!！?？、。]*"
    r"\b(?:" + _EN_FILLER_WORDS + r")\b"
    r"[\s.。,，;；!！?？、。]+"
    r")",
    re.IGNORECASE,
)
_CN_FILLER_PUNCT_RE = re.compile(
    r"(?:"
    r"[\s.。,，;；!！?？、。]+"
    r"(?:" + _CN_FILLER_WORDS + r")"
    r"[\s.。,，;；!！?？、。]*"
    r"|"
    r"[\s.。,，;；!！?？、。]*"
    r"(?:" + _CN_FILLER_WORDS + r")"
    r"[\s.。,，;；!！?？、。]+"
    r")",
)
# Angle-bracket tokens emitted as literal text by some backends:
#   <sil>          FireRed silence
#   <|zh|>, <|EMO|>, <|HAPPY|>, <|Speech|> …  SenseVoice format tags
#                  (sherpa-onnx usually strips these, but defense in depth)
_BRACKET_TOK_RE = re.compile(r"<\|?[^<>]*\|?>")
# Collapse the leftover whitespace (incl. ideographic full-width space U+3000)
_WS_COLLAPSE_RE = re.compile(r"[ \t　]+")
# After deletions, runs of dangling punctuation + whitespace pile up (e.g.
# "P. Yeah. Yeah。" → "P. . 。" — the periods are orphans left by the
# removed fillers).  Collapse 2+ adjacent punctuation/space chars to a
# single full-width period so the LLM still sees ONE sentence break.
_ORPHAN_PUNCT_RE = re.compile(r"[\s.。,，;；!！?？]{2,}")
# Trim leading/trailing punctuation+whitespace on each segment — the inter-
# segment join in ``_consume_pcm_stream`` already inserts a space, so any
# punctuation at the edges is noise from a deletion at the boundary.
_EDGE_PUNCT = " \t　.。,，;；!！?？"


def _postprocess_segment(text: str) -> str:
    """Strip cross-backend ASR noise from one recognized segment."""
    text = _BRACKET_TOK_RE.sub("", text)
    # Japanese / Korean kana — always noise in Chinese lectures
    text = _JP_NOISE_RE.sub("", text)
    text = _KR_NOISE_RE.sub("", text)
    # Filler words with punctuation neighbours — remove word + punct together
    text = _FILLER_PUNCT_RE.sub("", text)
    text = _CN_FILLER_PUNCT_RE.sub("", text)
    # Clean up what's left
    text = _ORPHAN_PUNCT_RE.sub("。", text)
    text = _WS_COLLAPSE_RE.sub(" ", text)
    text = text.strip(_EDGE_PUNCT).strip()
    if text in (".", "。", "？", "。", "!", "；", "，"):
        return ""
    return text


# Runtime is Qwen-only on this branch. Error types below remain API-compatible.



class IncompleteAudioError(RuntimeError):
    """Raised when downloaded audio is significantly shorter than expected.

    Carries the partial result so the caller can decide what to do with it
    without reaching into Transcriber internals."""

    def __init__(self, message: str, actual_duration: float,
                 expected_duration: float,
                 transcript: str = "",
                 segments: Optional[list[dict]] = None):
        super().__init__(message)
        self.actual_duration = actual_duration
        self.expected_duration = expected_duration
        self.transcript = transcript
        self.segments = segments or []


class NoAudioStreamError(RuntimeError):
    """Raised when the media contains no audio stream (video-only file)."""


from src.ai.qwen_transcriber import QwenTranscriber as Transcriber
