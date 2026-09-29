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
    def test_official_tail_detects_clearly_truncated_audio(self):
        LectureRunner = _load_runner_class()
        with self.assertRaisesRegex(
            RuntimeError, "audio ends well before official subtitle timeline"
        ):
            LectureRunner._assert_official_tail(4_000, [
                {"start_ms": 4_850_000, "end_ms": 4_900_000,
                 "text": "课尾内容"},
            ])

    def test_cloud_first_even_when_official_subtitles_are_complete(self):
        LectureRunner = _load_runner_class()
        db = MagicMock()
        db.get_lecture.return_value = None
        client = MagicMock()
        client.get_transcript_segments.return_value = [
            {"start_ms": 0, "end_ms": 60_000,
             "text": "官方错字" * 100},
        ]
        scheduler = MagicMock()
        transcriber = MagicMock()
        reporter = MagicMock()
        runner = LectureRunner(
            client, db, scheduler, transcriber, MagicMock(), reporter,
        )
        handle = SimpleNamespace(path="audio.raw", process=MagicMock(),
                                 stderr_chunks=[])
        scheduler.audio_downloader.get.return_value = handle
        from src.runtime import config
        from src.ai import doubao_asr
        with patch.object(config, "DOUBAO_ASR_API_KEY", "test-key"), \
             patch.object(config, "USE_OFFICIAL_TRANSCRIPT", True), \
             patch.object(doubao_asr, "wait_for_complete_audio",
                          return_value=(60, 60)), \
             patch.object(doubao_asr, "transcribe_pcm",
                          return_value=("云端正确", [
                       {"start_ms": 0, "end_ms": 60_000,
                        "text": "云端正确"},
                   ])):
            self.assertTrue(runner._needs_audio("course", "lecture"))
            text, _ = runner._get_transcript(None, "course", "lecture")
        self.assertEqual(text, "云端正确")
        self.assertEqual(runner._transcript_source, "cloud_asr")
        transcriber.transcribe_tail.assert_not_called()
        db.update_transcript.assert_not_called()

    def test_cloud_failure_uses_local_asr_not_official_subtitles(self):
        LectureRunner = _load_runner_class()
        db = MagicMock()
        client = MagicMock()
        client.get_transcript_segments.return_value = None
        scheduler = MagicMock()
        scheduler.audio_downloader.get.return_value = SimpleNamespace(
            path="audio.raw", process=MagicMock(), stderr_chunks=[],
        )
        transcriber = MagicMock()
        transcriber.transcribe_tail.return_value = (
            "本地结果", [{"start_ms": 0, "end_ms": 60_000,
                        "text": "本地结果"}],
        )
        transcriber.last_audio_duration = 60
        transcriber.last_media_duration = 60
        runner = LectureRunner(
            client, db, scheduler, transcriber, MagicMock(), MagicMock(),
        )
        from src.runtime import config
        from src.ai import doubao_asr
        with patch.object(config, "DOUBAO_ASR_API_KEY", "test-key"), \
             patch.object(doubao_asr, "wait_for_complete_audio",
                          return_value=(60, 60)), \
             patch.object(doubao_asr, "transcribe_pcm",
                          side_effect=doubao_asr.CloudASRError("failed")):
            text, _ = runner._get_transcript(None, "course", "lecture")
        self.assertEqual(text, "本地结果")
        self.assertEqual(runner._transcript_source, "local_asr")
        transcriber.transcribe_tail.assert_called_once()

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
