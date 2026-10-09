"""Compatibility import for src.pipeline.qwen_plan; implementation lives in src."""
import sys
from src.pipeline import qwen_plan as _implementation

sys.modules[__name__] = _implementation
