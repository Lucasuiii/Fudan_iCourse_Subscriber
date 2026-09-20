import unittest

from src.runtime.content_quality import assess_content_quality


def _segments(count: int, text: str = "有") -> list[dict]:
    return [
        {"start_ms": index * 60_000, "end_ms": index * 60_000 + 5_000,
         "text": text}
        for index in range(count)
    ]


class ContentQualityTests(unittest.TestCase):
    def test_complete_long_empty_recording_is_no_content(self):
        decision = assess_content_quality(
            transcript="零星声音" * 6,
            segments=_segments(9),
            ppt_pages=[],
            transcript_source="local_asr",
            actual_audio_seconds=10_054,
            expected_audio_seconds=10_054,
        )
        self.assertEqual(decision.action, "skip_no_content")

    def test_useful_transcript_is_summarized_without_ppt(self):
        decision = assess_content_quality(
            transcript="有效课程内容" * 80,
            segments=_segments(30, "有效课程内容"),
            ppt_pages=[],
            transcript_source="local_asr",
            actual_audio_seconds=5_400,
            expected_audio_seconds=5_400,
        )
        self.assertEqual(decision.action, "summarize")

    def test_useful_ppt_can_rescue_sparse_audio(self):
        decision = assess_content_quality(
            transcript="声音很少",
            segments=_segments(2),
            ppt_pages=[{"text": "完整课件内容" * 20}],
            transcript_source="local_asr",
            actual_audio_seconds=5_400,
            expected_audio_seconds=5_400,
        )
        self.assertEqual(decision.action, "summarize")

    def test_unknown_audio_completeness_retries(self):
        decision = assess_content_quality(
            transcript="声音很少",
            segments=_segments(2),
            ppt_pages=[],
            transcript_source="local_asr",
            actual_audio_seconds=5_400,
            expected_audio_seconds=0,
        )
        self.assertEqual(decision.action, "retry")

    def test_many_speech_segments_with_little_text_retries(self):
        decision = assess_content_quality(
            transcript="识别结果过少",
            segments=_segments(80),
            ppt_pages=[],
            transcript_source="local_asr",
            actual_audio_seconds=5_400,
            expected_audio_seconds=5_400,
        )
        self.assertEqual(decision.action, "retry")

    def test_ppt_failures_are_not_called_no_content(self):
        decision = assess_content_quality(
            transcript="声音很少",
            segments=_segments(2),
            ppt_pages=[],
            transcript_source="local_asr",
            actual_audio_seconds=5_400,
            expected_audio_seconds=5_400,
            ppt_failed_count=3,
        )
        self.assertEqual(decision.action, "retry")

    def test_short_recording_is_not_auto_skipped(self):
        decision = assess_content_quality(
            transcript="临时通知",
            segments=_segments(1),
            ppt_pages=[],
            transcript_source="local_asr",
            actual_audio_seconds=600,
            expected_audio_seconds=600,
        )
        self.assertEqual(decision.action, "summarize")

    def test_official_source_keeps_its_own_completeness_policy(self):
        decision = assess_content_quality(
            transcript="很短",
            segments=_segments(1),
            ppt_pages=[],
            transcript_source="official",
            actual_audio_seconds=5_400,
            expected_audio_seconds=5_400,
        )
        self.assertEqual(decision.action, "summarize")


if __name__ == "__main__":
    unittest.main()
