"""Compatibility import for src.ai.qwen_audio_alignment; implementation lives in src."""
import sys
from src.ai import qwen_audio_alignment as _implementation

sys.modules[__name__] = _implementation
