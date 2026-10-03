import json
import unittest

from scripts.benchmark_qwen_asr import parse_request


class QwenBenchmarkTests(unittest.TestCase):
    def request(self, **extra):
        return json.dumps({"course_id": "1", "sub_id": "2", "offset": 5220, "duration": 600, **extra})

    def test_accepts_bounded_private_slice(self):
        self.assertEqual(parse_request(self.request())["duration"], 600)

    def test_rejects_long_or_invalid_selection(self):
        for extra in ({"duration": 601}, {"duration": 0}, {"offset": -1},
                      {"duration": float("nan")}, {"course_id": "1; echo bad"}):
            with self.assertRaises(ValueError):
                parse_request(self.request(**extra))


if __name__ == "__main__":
    unittest.main()
