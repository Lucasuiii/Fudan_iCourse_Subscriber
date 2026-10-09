"""Compatibility import for src.ai.qwen_quality; implementation lives in src."""
import sys
from src.ai import qwen_quality as _implementation

sys.modules[__name__] = _implementation
