"""
针对本次修复点的回归测试:
1. compute_relevance_score / build_email_html 中的 keyword_weights 必须被正确使用,
   邮件内展示的相关度分数应与排序时的分数一致(此前 build_email_html 内部调用
   compute_relevance_score 时漏传 keyword_weights, 导致邮件展示分数与实际排序权重不符)。

运行方式: python -m unittest test_arxiv_digest.py -v
"""

import unittest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import arxiv_digest
from arxiv_digest import (
    compute_relevance_score,
    sort_papers_by_relevance,
    build_email_html,
    resolve_llm_model,
)


class FakePaper:
    """模拟 arxiv.Result 对象, 只提供本次测试用到的属性"""

    def __init__(self, title, summary, entry_id="http://arxiv.org/abs/9999.00001v1",
                 categories=None, primary_category="cs.RO"):
        self.title = title
        self.summary = summary
        self.entry_id = entry_id
        self.categories = categories or [primary_category]
        self.primary_category = primary_category
        self.authors = []
        self.published = datetime(2026, 9, 1, tzinfo=timezone.utc)
        self.updated = self.published
        self.comment = None
        self.journal_ref = None
        self.pdf_url = entry_id.replace("abs", "pdf")

    def get_short_id(self):
        return self.entry_id.split("/")[-1]


class TestKeywordWeights(unittest.TestCase):
    def setUp(self):
        self.keywords = ["unmanned surface vessel", "USV"]
        self.keyword_weights = {"unmanned surface vessel": 3, "USV": 2}
        # 标题命中 USV, 摘要命中 unmanned surface vessel
        self.paper = FakePaper(
            title="A Study on USV Navigation",
            summary="This paper focuses on the unmanned surface vessel control problem.",
        )

    def test_compute_relevance_score_uses_weights(self):
        weighted_score = compute_relevance_score(self.paper, self.keywords, self.keyword_weights)
        unweighted_score = compute_relevance_score(self.paper, self.keywords, {})
        # 加权后得分必须严格大于不加权(默认权重1)的得分, 否则说明权重未生效
        self.assertGreater(weighted_score, unweighted_score)

    def test_build_email_html_score_matches_weighted_score(self):
        expected_score = compute_relevance_score(self.paper, self.keywords, self.keyword_weights)
        html_with_weights = build_email_html(
            [self.paper], self.keywords, llm_cfg={}, keyword_weights=self.keyword_weights
        )
        html_without_weights = build_email_html(
            [self.paper], self.keywords, llm_cfg={}, keyword_weights=None
        )
        self.assertIn(f"相关度 {expected_score}", html_with_weights)
        # 不传权重时展示的分数应该是未加权得分, 与传权重时不同(证明参数确实生效)
        self.assertNotIn(f"相关度 {expected_score}", html_without_weights)

    def test_sort_papers_by_relevance_order_matches_email_score(self):
        low_score_paper = FakePaper(
            title="A generic robotics paper",
            summary="This paper is about generic robotics topics without special keywords.",
        )
        papers = sort_papers_by_relevance(
            [low_score_paper, self.paper], self.keywords, self.keyword_weights
        )
        # 高权重关键词命中的论文应排在前面
        self.assertIs(papers[0], self.paper)


class TestResolveLlmModel(unittest.TestCase):
    """针对 OpenRouter 免费模型自动下线/轮换时, 自动回退到当前可用模型的回归测试"""

    def setUp(self):
        # 每个用例都重置进程内缓存, 避免用例间相互污染
        arxiv_digest._openrouter_model_cache = {
            "checked": False, "available_ids": set(), "free_text_models": []
        }

    def _mock_models_response(self, model_ids_with_free_text):
        """构造一个模拟的 OpenRouter /models 响应: model_ids_with_free_text 为 (id, is_free_text) 列表"""
        data = []
        for model_id, is_free_text in model_ids_with_free_text:
            data.append({
                "id": model_id,
                "context_length": 100000,
                "architecture": {"output_modalities": ["text"] if is_free_text else ["image"]},
            })
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": data}
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    @patch("arxiv_digest.requests.get")
    def test_model_still_available_returns_unchanged(self, mock_get):
        mock_get.return_value = self._mock_models_response([
            ("some-provider/still-here:free", True),
        ])
        result = resolve_llm_model(
            "https://openrouter.ai/api/v1", "some-provider/still-here:free"
        )
        self.assertEqual(result, "some-provider/still-here:free")

    @patch("arxiv_digest.requests.get")
    def test_model_offline_falls_back_to_available_free_model(self, mock_get):
        mock_get.return_value = self._mock_models_response([
            ("some-provider/replacement:free", True),
            ("some-provider/image-model:free", False),  # 非文本模型, 不应被选中
        ])
        result = resolve_llm_model(
            "https://openrouter.ai/api/v1", "nex-agi/nex-n2.5-mini:free"
        )
        self.assertEqual(result, "some-provider/replacement:free")

    @patch("arxiv_digest.requests.get")
    def test_non_openrouter_base_url_skips_check(self, mock_get):
        result = resolve_llm_model(
            "https://api.deepseek.com/v1", "deepseek-chat"
        )
        self.assertEqual(result, "deepseek-chat")
        mock_get.assert_not_called()

    @patch("arxiv_digest.requests.get", side_effect=Exception("network down"))
    def test_fetch_failure_keeps_configured_model(self, mock_get):
        result = resolve_llm_model(
            "https://openrouter.ai/api/v1", "nex-agi/nex-n2.5-mini:free"
        )
        self.assertEqual(result, "nex-agi/nex-n2.5-mini:free")


if __name__ == "__main__":
    unittest.main()
