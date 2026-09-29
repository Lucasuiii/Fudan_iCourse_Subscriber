import unittest

from src.runtime.transcript_policy import (
    assess_official_transcript,
    merge_timed_segments,
    supplement_asr_gaps,
)


def _segment(start_s, end_s, text="有效字幕内容" * 100):
    return {
        "start_ms": int(start_s * 1000),
        "end_ms": int(end_s * 1000),
        "text": text,
    }


class OfficialTranscriptAssessmentTests(unittest.TestCase):
    def test_complete_when_media_tail_is_covered(self):
        mode, gaps = assess_official_transcript(
            [_segment(0, 1800), _segment(1801, 5350)],
            duration_s=5400,
        )
        self.assertEqual(mode, "complete")
        self.assertEqual(gaps, [])

    def test_real_media_duration_catches_tail_missing_after_last_ppt(self):
        mode, gaps = assess_official_transcript(
            [_segment(0, 1800), _segment(1801, 3000)],
            duration_s=5400,
        )
        self.assertEqual(mode, "hybrid")
        self.assertEqual(gaps, [(3000.0, 5400.0)])

    def test_sparse_stub_uses_full_asr(self):
        mode, gaps = assess_official_transcript(
            [_segment(0, 10, text="只有一点")],
            duration_s=5400,
        )
        self.assertEqual(mode, "full_asr")
        self.assertEqual(gaps, [])

    def test_too_many_missing_minutes_use_full_asr(self):
        mode, gaps = assess_official_transcript(
            [_segment(3600, 5400)],
            duration_s=5400,
        )
        self.assertEqual(mode, "full_asr")
        self.assertEqual(gaps, [])

    def test_gap_asr_segments_merge_in_time_order(self):
        merged = merge_timed_segments(
            [_segment(0, 10, "official-a"), _segment(30, 40, "official-b")],
            [_segment(15, 20, "asr")],
        )
        self.assertEqual([s["text"] for s in merged], [
            "official-a", "asr", "official-b",
        ])

    def test_official_is_only_marked_supplement_in_long_asr_gap(self):
        asr = [_segment(0, 20, "cloud-a"),
               _segment(220, 240, "cloud-b")]
        official = [_segment(0, 20),
                    _segment(60, 100, "字幕补充" * 30),
                    _segment(220, 240)]
        merged = supplement_asr_gaps(asr, official, 240)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["text"], "cloud-a")
        self.assertIn("官方字幕补充", merged[1]["text"])
        self.assertEqual(merged[2]["text"], "cloud-b")

    def test_official_never_replaces_a_full_asr_transcript(self):
        asr = [_segment(0, 120, "云端主文本")]
        official = [_segment(0, 120, "官方字幕" * 30)]
        self.assertEqual(supplement_asr_gaps(asr, official, 120), asr)
