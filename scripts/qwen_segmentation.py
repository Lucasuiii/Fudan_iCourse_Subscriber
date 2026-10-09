"""Compatibility import for src.ai.qwen_segmentation; implementation lives in src."""
import sys
from src.ai import qwen_segmentation as _implementation

sys.modules[__name__] = _implementation
