import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.ai.tavily_enrichment import enrich_summary, find_public_gaps


class TavilyEnrichmentTests(unittest.TestCase):
    def test_only_explicit_public_gaps_are_searched(self):
        summary = (
            "原始材料此处不清晰（待核：贝叶斯定理）\n"
            "原始材料此处不清晰（待核：老师的作业要求）\n"
            "原始材料此处不清晰（待核：条件概率）\n"
            "原始材料此处不清晰（待核：方差分解）"
        )
        self.assertEqual(
            [query for _, query in find_public_gaps(summary)],
            ["贝叶斯定理", "条件概率"],
        )

    def test_knowledge_or_no_key_never_searches(self):
        summary = "### 公式\n\n**补充说明：** 标准定义。"
        with patch("src.ai.tavily_enrichment.requests.post") as post:
            self.assertEqual(
                enrich_summary(summary, api_key="tvly-test", client=MagicMock(), model="m"),
                summary,
            )
            self.assertEqual(
                enrich_summary("原始材料此处不清晰（待核：条件概率）",
                               api_key="", client=MagicMock(), model="m"),
                "原始材料此处不清晰（待核：条件概率）",
            )
            post.assert_not_called()

    @patch("src.ai.tavily_enrichment.requests.post")
    def test_adds_cited_background_only_after_verification(self, post):
        post.return_value.json.return_value = {"results": [{
            "url": "https://example.edu/topic", "content": "条件概率是给定事件后计算的概率。",
        }]}
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"text":"条件概率是在给定条件下计算的概率。","source":1}'
            ))]
        )
        marker = "原始材料此处不清晰（待核：条件概率）"
        result = enrich_summary(marker, api_key="tvly-test", client=client, model="m")
        self.assertIn(marker, result)
        self.assertIn("外部来源，非课堂原话", result)
        self.assertIn("[来源](<https://example.edu/topic>)", result)
        request = post.call_args
        self.assertEqual(request.kwargs["json"]["query"], "条件概率")
        self.assertEqual(request.kwargs["json"]["search_depth"], "basic")
        self.assertFalse(request.kwargs["json"]["include_raw_content"])

    @patch("src.ai.tavily_enrichment.requests.post")
    def test_unverifiable_result_preserves_original(self, post):
        post.return_value.json.return_value = {"results": [{
            "url": "https://example.edu/topic", "content": "一个短片段",
        }]}
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"text":"","source":0}'
            ))]
        )
        marker = "原始材料此处不清晰（待核：条件概率）"
        self.assertEqual(
            enrich_summary(marker, api_key="tvly-test", client=client, model="m"),
            marker,
        )

    @patch("src.ai.tavily_enrichment.requests.post")
    def test_search_failure_does_not_discard_finished_summary(self, post):
        post.side_effect = RuntimeError("temporary provider failure")
        summary = "### 术语\n\n原始材料此处不清晰（待核：条件概率）"
        self.assertEqual(
            enrich_summary(summary, api_key="tvly-test", client=MagicMock(), model="m"),
            summary,
        )


if __name__ == "__main__":
    unittest.main()
