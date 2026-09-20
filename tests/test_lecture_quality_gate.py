import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _load_runner_class():
    """Import the runner without loading native ASR/OCR engines in unit tests."""
    sherpa = types.ModuleType("sherpa_onnx")
    rapid = types.ModuleType("rapidocr_onnxruntime")
    rapid.RapidOCR = object
    imagehash = types.ModuleType("imagehash")
    imagehash.dhash = lambda _image: "0" * 16
    crypto = types.ModuleType("Crypto")
    crypto_cipher = types.ModuleType("Crypto.Cipher")
    crypto_cipher.AES = object
    crypto_cipher.PKCS1_v1_5 = object
    crypto_public_key = types.ModuleType("Crypto.PublicKey")
    crypto_public_key.RSA = object
    with patch.dict(sys.modules, {
        "sherpa_onnx": sherpa,
        "rapidocr_onnxruntime": rapid,
        "imagehash": imagehash,
        "Crypto": crypto,
        "Crypto.Cipher": crypto_cipher,
        "Crypto.PublicKey": crypto_public_key,
    }):
        module = importlib.import_module("src.pipeline.lecture_runner")
    return module.LectureRunner


class LectureQualityGateIntegrationTests(unittest.TestCase):
    def test_no_content_bypasses_llm_and_email_batch(self):
        LectureRunner = _load_runner_class()
        db = MagicMock()
        db.get_lecture.return_value = None
        db.get_done_ppt_pages.return_value = []
        db.get_ppt_status_counts.return_value = {"dedup_dropped": 94}
        scheduler = MagicMock()
        reporter = MagicMock()
        summarizer = MagicMock()
        runner = LectureRunner(
            MagicMock(), db, scheduler, MagicMock(), summarizer, reporter,
        )
        runner._ppt = MagicMock()
        runner._ppt.submit.return_value.drain.return_value = SimpleNamespace(
            failed=0,
        )

        segments = [
            {"start_ms": i * 60_000, "end_ms": i * 60_000 + 5_000,
             "text": "有"}
            for i in range(9)
        ]

        def transcript(*_args, **_kwargs):
            runner._transcript_source = "local_asr"
            runner._asr_actual_duration = 10_054
            runner._asr_expected_duration = 10_054
            return "零星声音" * 6, segments

        runner._get_transcript = transcript
        result = runner.run(
            "course", "课程", {"sub_id": "lecture", "sub_title": "课次"},
        )

        self.assertIsNone(result)
        summarizer.summarize.assert_not_called()
        db.update_transcript.assert_called_once()
        db.mark_processed.assert_called_once_with("lecture")
        db.clear_error.assert_called_once_with("lecture")
        db.update_error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
