import unittest

from src.ai.segment_rescue import merge_rescued_segments, select_weak_windows


class SegmentRescueTests(unittest.TestCase):
    def test_no_vad_speech_means_no_cloud_upload(self):
        self.assertEqual(select_weak_windows([], 6_000), [])

    def test_only_weak_speech_and_no_more_than_fifteen_minutes(self):
        windows = [
            {"start_ms": 0, "end_ms": 20_000, "text": "清楚的课堂讲解"},
            {"start_ms": 30_000, "end_ms": 45_000, "text": ""},
        ]
        windows += [
            {"start_ms": i * 40_000, "end_ms": i * 40_000 + 30_000,
             "text": "嗯"}
            for i in range(2, 30)
        ]
        picked = select_weak_windows(windows, 1_200)
        self.assertLessEqual(len(picked), 40)
        self.assertLessEqual(sum(w["end_ms"] - w["start_ms"]
                                 for w in picked), 900_000)
        self.assertNotIn(windows[0], picked)
        self.assertIn(windows[1], picked)

    def test_default_budget_allows_fifteen_one_minute_windows(self):
        windows = [{"start_ms": i * 60_000, "end_ms": (i + 1) * 60_000,
                    "text": ""} for i in range(18)]
        picked = select_weak_windows(windows, 1080)
        self.assertEqual(len(picked), 15)
        self.assertEqual(sum(w["end_ms"] - w["start_ms"] for w in picked),
                         900_000)

    def test_cloud_replaces_only_matching_weak_window(self):
        local = [
            {"start_ms": 0, "end_ms": 10_000, "text": "保留"},
            {"start_ms": 20_000, "end_ms": 30_000, "text": "嗯"},
        ]
        recovered = merge_rescued_segments(local, [
            (local[1], [{"start_ms": 20_000, "end_ms": 30_000,
                         "text": "矩阵的条件数"}]),
        ])
        self.assertEqual([s["text"] for s in recovered],
                         ["保留", "矩阵的条件数"])

    def test_empty_cloud_result_keeps_local(self):
        weak = {"start_ms": 20_000, "end_ms": 30_000, "text": "嗯"}
        self.assertEqual(merge_rescued_segments([weak], [(weak, [])]), [weak])


if __name__ == "__main__":
    unittest.main()
