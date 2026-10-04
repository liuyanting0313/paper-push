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

import requests

import arxiv_digest
from arxiv_digest import (
    compute_relevance_score,
    sort_papers_by_relevance,
    build_email_html,
    resolve_llm_model,
    generate_llm_insights,
    SemanticScholarPaper,
    fetch_all_sources,
    _call_llm_for_insight,
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


class TestGenerateLlmInsights(unittest.TestCase):
    """针对'免费模型 + DeepSeek 对比模型同时调用, 邮件中两段速读并列展示'的回归测试"""

    def setUp(self):
        arxiv_digest._openrouter_model_cache = {
            "checked": False, "available_ids": set(), "free_text_models": []
        }
        self.paper = FakePaper(
            title="A Study on USV Navigation",
            summary="This paper focuses on the unmanned surface vessel control problem.",
        )

    @staticmethod
    def _mock_llm_response(plain_explain, innovation="x", method="y", conclusion="z"):
        content = (
            f'{{"plain_explain": "{plain_explain}", "innovation": "{innovation}", '
            f'"method": "{method}", "conclusion": "{conclusion}"}}'
        )
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
        return mock_resp

    @patch("arxiv_digest.requests.post")
    def test_both_models_enabled_returns_two_results(self, mock_post):
        mock_post.side_effect = [
            self._mock_llm_response("免费模型的大白话"),
            self._mock_llm_response("DeepSeek的大白话"),
        ]
        llm_cfg = {
            "enabled": True, "label": "免费模型", "base_url": "https://openrouter.ai/api/v1",
            "model": "qwen/qwen3.8-27b:free", "api_key": "fake-key",
            "compare": {
                "enabled": True, "label": "DeepSeek", "base_url": "https://api.deepseek.com/v1",
                "model": "deepseek-chat", "api_key": "fake-deepseek-key",
            },
        }
        results = generate_llm_insights(self.paper, llm_cfg)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["label"], "免费模型")
        self.assertEqual(results[0]["insight"]["plain_explain"], "免费模型的大白话")
        self.assertEqual(results[1]["label"], "DeepSeek")
        self.assertEqual(results[1]["insight"]["plain_explain"], "DeepSeek的大白话")

    @patch("arxiv_digest.requests.post")
    def test_compare_disabled_returns_only_primary(self, mock_post):
        mock_post.return_value = self._mock_llm_response("只有免费模型")
        llm_cfg = {
            "enabled": True, "label": "免费模型", "base_url": "https://openrouter.ai/api/v1",
            "model": "qwen/qwen3.8-27b:free", "api_key": "fake-key",
            "compare": {"enabled": False},
        }
        results = generate_llm_insights(self.paper, llm_cfg)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["label"], "免费模型")

    @patch("arxiv_digest.requests.post")
    def test_primary_fails_compare_succeeds(self, mock_post):
        def _side_effect(url, headers, json, timeout):
            if json["model"] == "qwen/qwen3.8-27b:free":
                raise ConnectionError("primary down")
            return self._mock_llm_response("DeepSeek兜底")

        mock_post.side_effect = _side_effect
        llm_cfg = {
            "enabled": True, "label": "免费模型", "base_url": "https://openrouter.ai/api/v1",
            "model": "qwen/qwen3.8-27b:free", "api_key": "fake-key",
            "compare": {
                "enabled": True, "label": "DeepSeek", "base_url": "https://api.deepseek.com/v1",
                "model": "deepseek-chat", "api_key": "fake-deepseek-key",
            },
        }
        results = generate_llm_insights(self.paper, llm_cfg)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["label"], "DeepSeek")

    def test_primary_disabled_returns_empty(self):
        results = generate_llm_insights(self.paper, {"enabled": False})
        self.assertEqual(results, [])


class TestSemanticScholarPaper(unittest.TestCase):
    """Semantic Scholar 数据适配层(统一渲染接口)的回归测试"""

    def test_adapts_fields_correctly(self):
        raw = {
            "title": "A Survey on USV",
            "abstract": "This is an abstract.",
            "authors": [{"name": "Alice Smith"}, {"name": "Bob Lee"}],
            "year": 2026,
            "publicationDate": "2026-09-15",
            "venue": "ICRA",
            "url": "https://www.semanticscholar.org/paper/abc123",
            "externalIds": {"ArXiv": "2609.12345", "DOI": "10.1000/xyz"},
            "openAccessPdf": {"url": "https://example.com/paper.pdf"},
            "fieldsOfStudy": ["Computer Science"],
            "paperId": "abc123",
        }
        paper = SemanticScholarPaper(raw)
        self.assertEqual(paper.title, "A Survey on USV")
        self.assertEqual(paper.summary, "This is an abstract.")
        self.assertEqual([a.name for a in paper.authors], ["Alice Smith", "Bob Lee"])
        self.assertEqual(paper.journal_ref, "ICRA")
        self.assertEqual(paper.pdf_url, "https://example.com/paper.pdf")
        self.assertEqual(paper.arxiv_id, "2609.12345")
        # 带 arXiv 编号时, get_short_id 应复用该编号以便跨源去重
        self.assertEqual(paper.get_short_id(), "2609.12345")

    def test_get_short_id_falls_back_to_s2_id_without_arxiv(self):
        raw = {
            "title": "A Non-arXiv Paper", "abstract": "abstract text",
            "authors": [], "paperId": "xyz789", "externalIds": {},
        }
        paper = SemanticScholarPaper(raw)
        self.assertEqual(paper.get_short_id(), "s2-xyz789")


class TestFetchAllSources(unittest.TestCase):
    """多来源检索与跨源去重的回归测试"""

    @patch("arxiv_digest.fetch_semantic_scholar_papers")
    @patch("arxiv_digest.fetch_recent_papers")
    def test_dedups_semantic_scholar_paper_already_in_arxiv(self, mock_fetch_arxiv, mock_fetch_s2):
        arxiv_paper = FakePaper(
            title="Already on arXiv", summary="...", entry_id="http://arxiv.org/abs/2609.12345v1"
        )
        mock_fetch_arxiv.return_value = [arxiv_paper]

        dup_s2_paper = SemanticScholarPaper({
            "title": "Already on arXiv (S2 copy)", "abstract": "...",
            "authors": [], "paperId": "dup1", "externalIds": {"ArXiv": "2609.12345"},
        })
        new_s2_paper = SemanticScholarPaper({
            "title": "Only on Semantic Scholar", "abstract": "...",
            "authors": [], "paperId": "new1", "externalIds": {},
        })
        mock_fetch_s2.return_value = [dup_s2_paper, new_s2_paper]

        result = fetch_all_sources({"sources": ["arxiv", "semantic_scholar"], "keywords": ["USV"]})
        self.assertEqual(len(result), 2)
        self.assertIn(arxiv_paper, result)
        self.assertIn(new_s2_paper, result)
        self.assertNotIn(dup_s2_paper, result)

    @patch("arxiv_digest.fetch_semantic_scholar_papers")
    @patch("arxiv_digest.fetch_recent_papers")
    def test_only_arxiv_source_skips_semantic_scholar_call(self, mock_fetch_arxiv, mock_fetch_s2):
        mock_fetch_arxiv.return_value = []
        fetch_all_sources({"sources": ["arxiv"], "keywords": ["USV"]})
        mock_fetch_s2.assert_not_called()


class TestModelFallbackChain(unittest.TestCase):
    """
    针对'任意 LLM 来源(免费模型/DeepSeek/未来新增模型), 当前模型改名/下线后自动切换到
    候选链中下一个模型'的回归测试, 覆盖 fallback_models 配置项的通用自愈能力
    """

    def setUp(self):
        arxiv_digest._openrouter_model_cache = {
            "checked": False, "available_ids": set(), "free_text_models": []
        }
        self.paper = FakePaper(
            title="A Study on USV Navigation",
            summary="This paper focuses on the unmanned surface vessel control problem.",
        )

    @staticmethod
    def _ok_response(plain_explain="hi"):
        content = (
            f'{{"plain_explain": "{plain_explain}", "innovation": "i", '
            f'"method": "m", "conclusion": "c"}}'
        )
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
        return mock_resp

    @staticmethod
    def _http_error_response(status_code: int, body: str = ""):
        """构造一个会在 raise_for_status() 时抛出带 .response 的 HTTPError 的 mock 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.text = body
        err = requests.exceptions.HTTPError(f"{status_code} error", response=mock_resp)
        mock_resp.raise_for_status.side_effect = err
        return mock_resp

    @patch("arxiv_digest.requests.post")
    def test_404_on_primary_switches_to_fallback_model(self, mock_post):
        # 第一个模型(deepseek-flash-renamed)连续两次(即 max_retries=2 耗尽)都返回 404,
        # 应自动切换到 fallback_models 里的 deepseek-chat 并成功
        mock_post.side_effect = [
            self._http_error_response(404),
            self._http_error_response(404),
            self._ok_response("用备用模型生成的"),
        ]
        cfg = {
            "enabled": True, "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-flash-renamed", "api_key": "fake-key",
            "fallback_models": ["deepseek-chat"],
        }
        result = _call_llm_for_insight(self.paper, cfg)
        self.assertEqual(result.get("plain_explain"), "用备用模型生成的")
        # 实际发出的最后一次请求应使用 fallback_models 中的模型名
        last_call_payload = mock_post.call_args_list[-1].kwargs["json"]
        self.assertEqual(last_call_payload["model"], "deepseek-chat")

    @patch("arxiv_digest.requests.post")
    def test_auth_error_does_not_trigger_fallback(self, mock_post):
        # 401 鉴权失败不属于"模型不存在", 换模型也解决不了问题, 不应尝试 fallback_models
        mock_post.return_value = self._http_error_response(401, "invalid api key")
        cfg = {
            "enabled": True, "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-flash", "api_key": "wrong-key",
            "fallback_models": ["deepseek-chat"],
        }
        result = _call_llm_for_insight(self.paper, cfg)
        self.assertEqual(result, {})
        called_models = {c.kwargs["json"]["model"] for c in mock_post.call_args_list}
        self.assertEqual(called_models, {"deepseek-flash"})

    @patch("arxiv_digest.requests.post")
    def test_all_candidates_exhausted_returns_empty(self, mock_post):
        mock_post.return_value = self._http_error_response(404)
        cfg = {
            "enabled": True, "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-flash", "api_key": "fake-key",
            "fallback_models": ["deepseek-chat", "deepseek-reasoner"],
        }
        result = _call_llm_for_insight(self.paper, cfg)
        self.assertEqual(result, {})
        called_models = [c.kwargs["json"]["model"] for c in mock_post.call_args_list]
        # 三个候选都应被尝试过(去重后: deepseek-flash, deepseek-chat, deepseek-reasoner)
        self.assertEqual(set(called_models), {"deepseek-flash", "deepseek-chat", "deepseek-reasoner"})

    @patch("arxiv_digest.requests.post")
    def test_no_fallback_configured_behaves_like_before(self, mock_post):
        mock_post.return_value = self._ok_response("照常工作")
        cfg = {
            "enabled": True, "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-flash", "api_key": "fake-key",
        }
        result = _call_llm_for_insight(self.paper, cfg)
        self.assertEqual(result.get("plain_explain"), "照常工作")


class TestLlmResponseEdgeCases(unittest.TestCase):
    """
    针对真实运行中观察到的两类 LLM 响应异常的回归测试:
    1. OpenRouter/免费模型偶尔返回 content=None(被截断或触发内容过滤), 之前会直接抛 AttributeError
    2. 模型在 JSON 前后多输出解释性文字导致 json.loads 直接失败, 需要尝试提取 {...} 边界重新解析
    """

    def setUp(self):
        self.paper = FakePaper(
            title="A Study on USV Navigation",
            summary="This paper focuses on the unmanned surface vessel control problem.",
        )
        self.cfg = {
            "enabled": True, "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-flash", "api_key": "fake-key",
        }

    @patch("arxiv_digest.requests.post")
    def test_none_content_returns_empty_without_crash(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": None}, "finish_reason": "content_filter"}]
        }
        mock_post.return_value = mock_resp
        result = _call_llm_for_insight(self.paper, self.cfg)
        self.assertEqual(result, {})

    @patch("arxiv_digest.requests.post")
    def test_json_with_surrounding_text_is_recovered(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        content = (
            '好的, 这是分析结果:\n'
            '{"plain_explain": "简单说就是这样", "innovation": "i", "method": "m", "conclusion": "c"}\n'
            '希望对你有帮助!'
        )
        mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
        mock_post.return_value = mock_resp
        result = _call_llm_for_insight(self.paper, self.cfg)
        self.assertEqual(result.get("plain_explain"), "简单说就是这样")

    @patch("arxiv_digest.requests.post")
    def test_truly_malformed_json_still_fails_gracefully(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"choices": [{"message": {"content": "not json at all, no braces"}}]}
        mock_post.return_value = mock_resp
        result = _call_llm_for_insight(self.paper, self.cfg)
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
