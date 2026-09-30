import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.ai.asr_review import review_windows


class ASRReviewTests(unittest.TestCase):
    def test_rejects_invented_duplicate_and_excluded_ids(self):
        windows = [
            {"start_ms": i * 30_000, "end_ms": i * 30_000 + 20_000,
             "text": "术语可能识别错误"} for i in range(3)
        ]
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "suspects": [
                    {"id": 999, "reason": "不存在"},
                    {"id": 0, "reason": "已处理"},
                    {"id": True, "reason": "类型错误"},
                    {"id": 1, "reason": "疑似同音误识别"},
                    {"id": 1, "reason": "重复"},
                ],
            })))],
        )
        result = review_windows(client, "model", windows, [], {(0, 20_000)})
        self.assertEqual(result, [windows[1]])
        client.chat.completions.create.assert_called_once()
        self.assertEqual(client.chat.completions.create.call_args.kwargs["max_tokens"], 1_000)

    def test_silence_does_not_trigger_llm_call(self):
        client = MagicMock()
        self.assertEqual(review_windows(client, "model", [], [], set()), [])
        client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
