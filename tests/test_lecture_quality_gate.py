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
    def test_reused_runner_disables_glossary_without_retaining_previous_course_terms(self):
        LectureRunner = _load_runner_class()
        db = MagicMock()
        # A cached summary must remain unchanged while state still resets.
        db.get_lecture.return_value = {'summary': '历史摘要', 'emailed_at': 'already sent'}
        transcriber = MagicMock()
        runner = LectureRunner(MagicMock(), db, MagicMock(), transcriber,
                               MagicMock(), MagicMock())
        with patch('src.ai.automatic_glossary.AutomaticGlossary') as glossary, \
             patch('src.ai.course_glossary.course_terms', return_value=['乙人工词']):
            glossary.return_value.freeze.return_value = {'terms': ['甲自动词', '甲人工词']}
            with patch.dict('os.environ', {'AUTO_COURSE_TERMS': 'true'}):
                runner.run('10', '课程甲', {'sub_id': '1'})
            with patch.dict('os.environ', {'AUTO_COURSE_TERMS': 'false'}):
                runner.run('20', '课程乙', {'sub_id': '2'})
        self.assertEqual(transcriber.set_terms.call_args.args[0], ['乙人工词'])
        self.assertIsNone(runner._automatic_glossary)
        db.update_summary.assert_not_called()

    def test_reused_runner_does_not_feed_previous_review_into_next_summary(self):
        LectureRunner = _load_runner_class()
        db = MagicMock()
        db.get_lecture.return_value = None
        db.get_done_ppt_pages.return_value = []
        db.get_ppt_status_counts.return_value = {}
        summarizer = MagicMock()
        summarizer.summarize.return_value = ('当前课摘要', 'test')
        runner = LectureRunner(MagicMock(), db, MagicMock(), MagicMock(),
                               summarizer, MagicMock())
        runner._ppt = MagicMock()
        runner._ppt.submit.return_value.drain.return_value = SimpleNamespace(failed=0)
        runner._get_transcript = MagicMock(return_value=('当前课正文', []))

        def review(text, segments, pages, **kwargs):
            if not summarizer.summarize.called:
                runner._qwen_review_material = {'variants': [{'cloud_text': '上课复核证据'}]}
                runner._cloud_term_sources.append('上课术语证据')
            return text, segments

        runner._refine_unclear_transcript = review
        with patch.dict('os.environ', {'AUTO_COURSE_TERMS': 'false'}):
            runner.run('10', '课程甲', {'sub_id': '1'})
            runner.run('20', '课程乙', {'sub_id': '2'})
        first, second = summarizer.summarize.call_args_list
        self.assertIn('上课复核证据', first.args[1])
        self.assertNotIn('上课复核证据', second.args[1])
        self.assertEqual(runner._cloud_term_sources, [])

    def test_vad_speech_with_empty_local_result_is_recorded_for_rescue(self):
        with patch.dict(sys.modules, {
            "sherpa_onnx": types.ModuleType("sherpa_onnx"),
        }):
            module = importlib.import_module("src.ai.transcriber")
        transcriber = module.Transcriber.__new__(module.Transcriber)
        transcriber._last_speech_windows = []
        speech = SimpleNamespace(start=16_000, samples=[0.0] * 160_000)
        vad = SimpleNamespace(done=False, front=speech)
        vad.empty = lambda: vad.done
        vad.pop = lambda: setattr(vad, "done", True)
        windows=[]
        transcriber._drain_vad(vad,windows)
        self.assertEqual(windows,[(1.0,11.0)])

    def test_local_asr_returns_substantial_audio_despite_media_mismatch(self):
        with patch.dict(sys.modules, {
            "sherpa_onnx": types.ModuleType("sherpa_onnx"),
        }):
            module = importlib.import_module("src.ai.transcriber")
        transcriber = module.Transcriber.__new__(module.Transcriber)
        transcriber._media_duration = 10087
        transcriber._last_duration = 4061
        transcriber._consume_pcm_stream = MagicMock(
            return_value=("课堂内容" * 100, []),
        )
        import tempfile
        with tempfile.NamedTemporaryFile() as audio:
            self.assertEqual(
                transcriber.transcribe_tail(
                    audio.name, MagicMock(returncode=0), [],
                )[0],
                "课堂内容" * 100,
            )

    def test_official_tail_warns_without_blocking_asr(self):
        LectureRunner = _load_runner_class()
        reporter = MagicMock()
        runner = LectureRunner(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(),
            MagicMock(), reporter,
        )
        runner._warn_official_tail(4_000, [
            {"start_ms": 4_850_000, "end_ms": 4_900_000,
             "text": "课尾内容"},
        ])
        reporter.info.assert_called_once()
        self.assertIn("continuing with ASR", reporter.info.call_args.args[0])

    def test_local_first_rescues_only_weak_speech_window(self):
        LectureRunner = _load_runner_class()
        db = MagicMock()
        db.get_lecture.return_value = None
        client = MagicMock()
        client.get_transcript_segments.return_value = [
            {"start_ms": 0, "end_ms": 4_900_000,
             "text": "官方错字" * 100},
        ]
        scheduler = MagicMock()
        transcriber = MagicMock()
        transcriber.transcribe_tail.return_value = (
            "本地正确 嗯", [
                {"start_ms": 0, "end_ms": 20_000, "text": "本地正确"},
                {"start_ms": 40_000, "end_ms": 60_000, "text": "嗯"},
            ],
        )
        transcriber.last_audio_duration = 4061
        transcriber.last_media_duration = 10087
        transcriber.last_speech_windows = [
            {"start_ms": 0, "end_ms": 20_000, "text": "本地正确"},
            {"start_ms": 40_000, "end_ms": 60_000, "text": "嗯"},
        ]
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
             patch.object(doubao_asr, "rescue_intervals_pcm",
                          return_value=([(
                              transcriber.last_speech_windows[1], [
                                  {"start_ms": 40_000, "end_ms": 60_000,
                                   "text": "云端修复"},
                              ],
                          )], 20.0, False)) as cloud_transcribe:
            self.assertTrue(runner._needs_audio("course", "lecture"))
            text, _ = runner._get_transcript(None, "course", "lecture")
        self.assertEqual(text, "本地正确 云端修复")
        self.assertEqual(runner._transcript_source, "hybrid_asr")
        cloud_transcribe.assert_called_once()
        self.assertEqual(cloud_transcribe.call_args.args[-1], [
            transcriber.last_speech_windows[1],
        ])
        self.assertTrue(any(
            "Audio shorter than media timeline" in call.args[0]
            for call in reporter.info.call_args_list
        ))
        self.assertTrue(any(
            "Official subtitle timeline exceeds audio" in call.args[0]
            for call in reporter.info.call_args_list
        ))
        transcriber.transcribe_tail.assert_called_once()
        db.update_transcript.assert_not_called()

    def test_cloud_rescue_failure_preserves_local_asr(self):
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
        transcriber.last_speech_windows = [
            {"start_ms": 0, "end_ms": 10_000, "text": "本地结果"},
            {"start_ms": 20_000, "end_ms": 30_000, "text": ""},
        ]
        runner = LectureRunner(
            client, db, scheduler, transcriber, MagicMock(), MagicMock(),
        )
        from src.runtime import config
        from src.ai import doubao_asr
        with patch.object(config, "DOUBAO_ASR_API_KEY", "test-key"), \
             patch.object(doubao_asr, "rescue_intervals_pcm",
                          return_value=([], 10.0, True)):
            text, _ = runner._get_transcript(None, "course", "lecture")
        self.assertEqual(text, "本地结果")
        self.assertEqual(runner._transcript_source, "local_asr")
        transcriber.transcribe_tail.assert_called_once()

    def test_silent_recording_never_calls_cloud(self):
        LectureRunner = _load_runner_class()
        scheduler = MagicMock()
        scheduler.audio_downloader.get.return_value = SimpleNamespace(
            path="audio.raw", process=MagicMock(), stderr_chunks=[],
        )
        transcriber = MagicMock()
        transcriber.transcribe_tail.return_value = ("", [])
        transcriber.last_audio_duration = 3600
        transcriber.last_media_duration = 3600
        transcriber.last_speech_windows = []
        runner = LectureRunner(
            MagicMock(), MagicMock(), scheduler, transcriber,
            MagicMock(), MagicMock(),
        )
        from src.runtime import config
        from src.ai import doubao_asr
        with patch.object(config, "DOUBAO_ASR_API_KEY", "test-key"), \
             patch.object(config, "USE_OFFICIAL_TRANSCRIPT", False), \
             patch.object(doubao_asr, "rescue_intervals_pcm") as rescue:
            text, segments = runner._get_transcript(
                None, "course", "lecture",
            )
        self.assertEqual((text, segments), ("", []))
        rescue.assert_not_called()

    def test_llm_rescue_uses_remaining_lecture_budget(self):
        LectureRunner = _load_runner_class()
        summarizer = MagicMock()
        local = [{"start_ms": 0, "end_ms": 20_000, "text": "课程内容" * 60}]
        suspect = local[0]
        summarizer.find_unclear_windows.return_value = [suspect]
        transcriber = MagicMock()
        transcriber.last_speech_windows = local
        runner = LectureRunner(
            MagicMock(), MagicMock(), MagicMock(), transcriber,
            summarizer, MagicMock(),
        )
        runner._asr_audio_path = "audio.raw"
        runner._transcript_source = "local_asr"
        runner._cloud_seconds = 570
        runner._cloud_windows = {(i * 30_000, i * 30_000 + 10_000)
                                 for i in range(1, 9)}
        from src.runtime import config
        from src.ai import doubao_asr
        with patch.object(config, "DOUBAO_ASR_API_KEY", "key"), \
             patch.object(doubao_asr, "rescue_intervals_pcm",
                          return_value=([(suspect, local)], 20, False)) as rescue:
            runner._refine_unclear_transcript(local[0]["text"], local, [],
                                              course_title="高等代数Ⅰ")
        self.assertEqual(rescue.call_args.kwargs,
                     {"max_seconds": 330, "max_clips": 32})
        self.assertEqual(runner._cloud_seconds, 590)
        self.assertEqual(summarizer.find_unclear_windows.call_args.kwargs,
                         {"course_title": "高等代数Ⅰ"})

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
