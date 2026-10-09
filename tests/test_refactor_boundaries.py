"""Import and checkpoint boundaries, without models, campus access or SMTP."""
import importlib
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CoreImportTests(unittest.TestCase):
    def test_core_plans_and_recognition_import_without_script_entrypoints(self):
        # A core-only consumer must not import CLI modules or construct models.
        code = """
import sys
sys.modules['scripts'] = None
from src.pipeline.asr_queue import SharedQueue
from src.pipeline.prepared_lecture import assemble_material
from src.ai.qwen_transcriber import QwenTranscriber
from src.ai.qwen_missing_fallback import repair_missing
from src.ai.qwen_review_ledger import validate_ledger
from src.pipeline.runner_budget import desired_workers
assert desired_workers(7200, 60, 2) == 4
assert 'torch' not in sys.modules and 'qwen_asr' not in sys.modules
"""
        subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                       check=True, capture_output=True, text=True)

    def test_legacy_helper_imports_share_one_implementation(self):
        # Existing external callers and patch targets retain the same module.
        for old, new in (
            ('scripts.qwen_sharding', 'src.pipeline.qwen_plan'),
            ('scripts.qwen_segmentation', 'src.ai.qwen_segmentation'),
            ('scripts.qwen_quality', 'src.ai.qwen_quality'),
            ('scripts.qwen_audio_alignment', 'src.ai.qwen_audio_alignment'),
        ):
            with self.subTest(old=old):
                self.assertIs(importlib.import_module(old), importlib.import_module(new))


if __name__ == '__main__':
    unittest.main()
