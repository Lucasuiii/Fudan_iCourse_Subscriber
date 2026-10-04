import json
import unittest
from unittest.mock import Mock,patch

from scripts.benchmark_qwen_asr import parse_request, auth_phase, configure_auth_session,sample_seconds
from scripts.qwen_segmentation import plan_chunks, plan_long_chunks, join_chunk_text


class QwenBenchmarkTests(unittest.TestCase):
    def test_bounded_sample_duration(self):
        with patch.dict('os.environ',{'SAMPLE_MINUTES':'30','LONG_CHUNK_SAMPLE':'true'}):
            self.assertEqual(sample_seconds(),1800)
        for extra in ({'SAMPLE_MINUTES':'30','LONG_CHUNK_SAMPLE':'false'},
                      {'SAMPLE_MINUTES':'60','LONG_CHUNK_SAMPLE':'true'}):
            with patch.dict('os.environ',extra),self.assertRaises(ValueError):
                sample_seconds()
    def request(self, **extra):
        return json.dumps({"course_id": "1", "sub_id": "2", "offset": 5220, "duration": 600, **extra})

    def test_accepts_bounded_private_slice(self):
        self.assertEqual(parse_request(self.request())["duration"], 600)

    def test_rejects_long_or_invalid_selection(self):
        for extra in ({"duration": 601}, {"duration": 0}, {"offset": -1},
                      {"duration": float("nan")}, {"course_id": "1; echo bad"}):
            with self.assertRaises(ValueError):
                parse_request(self.request(**extra))

    def test_silence_skipped_and_pause_preserved(self):
        self.assertEqual(plan_chunks([], 120), [])
        self.assertEqual(plan_chunks([(10, 20), (40, 50)], 120), [(9, 21), (39, 51)])

    def test_short_pause_merges_and_long_speech_is_bounded(self):
        self.assertEqual(plan_chunks([(0, 10), (10.5, 20)], 60), [(0, 21)])
        chunks = plan_chunks([(0, 95)], 95)
        self.assertTrue(all(end - start <= 30 for start, end in chunks))
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], 95)
        self.assertTrue(all(a[1] >= b[0] for a, b in zip(chunks, chunks[1:])))

    def test_invalid_window_rejected(self):
        for window in ((-1, 2), (2, 1), (0, 61), (0, float('nan'))):
            with self.assertRaises(ValueError):
                plan_chunks([window], 60)

    def test_boundary_dedupe_does_not_delete_nonoverlapping_repetition(self):
        rows = [{"start": 0, "end": 30, "text": "我们讨论希尔伯特矩阵"},
                {"start": 28, "end": 60, "text": "希尔伯特矩阵的条件数"}]
        self.assertEqual(join_chunk_text(rows), "我们讨论希尔伯特矩阵\n的条件数")
        rows[1]['start'] = 30
        self.assertIn("\n希尔伯特矩阵", join_chunk_text(rows))

    def test_short_repetition_preserved(self):
        rows = [{"start": 0, "end": 30, "text": "矩阵"},
                {"start": 28, "end": 60, "text": "矩阵很重要"}]
        self.assertEqual(join_chunk_text(rows), "矩阵\n矩阵很重要")

    def test_long_chunks_keep_short_pauses(self):
        windows = [(i, i+3) for i in range(0, 600, 5)]
        chunks = plan_long_chunks(windows, 600)
        self.assertLessEqual(len(chunks), 7)
        self.assertTrue(all(b-a <= 122 for a,b in chunks))
        self.assertEqual(chunks[0][0], 0)
        self.assertTrue(all(a[1] >= b[0] for a,b in zip(chunks,chunks[1:])))

    def test_long_chunks_skip_long_silence_not_quiet_speech(self):
        self.assertEqual(plan_long_chunks([], 600), [])
        self.assertEqual(plan_long_chunks([(10,20),(50,55)], 600), [(9,21),(49,56)])
        self.assertEqual(plan_long_chunks([(10,20),(25,35)], 600), [(9,36)])

    def test_long_chunks_continuous_speech_and_invalid_input(self):
        chunks = plan_long_chunks([(0,600)], 600)
        self.assertEqual(len(chunks),5)
        self.assertEqual(chunks[-1][1],600)
        with self.assertRaises(ValueError):
            plan_long_chunks([(0,601)],600)

    def test_auth_phase_never_exposes_query_or_unknown_path(self):
        self.assertEqual(auth_phase('https://example.org/idp/authn/authExecute?token=private'), 'credential_exchange')
        self.assertEqual(auth_phase('https://example.org/private-account?ticket=private'), 'portal_or_redirect')

    def test_timeout_policy_and_safe_diagnostics(self):
        session = Mock()
        response = Mock(status_code=200)
        original = Mock(return_value=response)
        session.request = original
        events = []
        configure_auth_session(session, events)
        session.request('GET', 'https://example.org/', timeout=5)
        self.assertEqual(original.call_args.kwargs['timeout'], (15, 20))
        original.side_effect = TimeoutError('private-token')
        with self.assertRaises(TimeoutError):
            session.request('POST', 'https://example.org/authExecute', timeout=60)
        self.assertEqual(original.call_count, 2)  # No credential POST replay.
        self.assertNotIn('private-token', json.dumps(events))
        self.assertEqual(events[-1]['error_type'], 'TimeoutError')


if __name__ == "__main__":
    unittest.main()
