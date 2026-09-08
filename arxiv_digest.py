"""
arXiv 论文摘要推送脚本(支持每日/每周模式)

功能:
    1. 按配置的 arXiv 分类 + 关键词组合查询最新论文
    2. 过滤出最近 N 天内提交、且尚未推送过的论文(支持排除关键词)
    3. 按关键词命中数计算相关度得分, 按得分优先排序
    4. 对英文摘要提供中文机器翻译, 可选接入 LLM(OpenAI 兼容接口, 如 DeepSeek) 生成
       创新点/方法/结论的中文精读要点
    5. 将论文标题/作者/分类/摘要/BibTeX 等整理成学术日报风格邮件(含纯HTML+纯文本冗余), 通过 SMTP 发送
       (支持多收件人, 发送失败自动指数退避重试)
    6. 可选同时推送到企业微信/飞书/Slack 等 Webhook 机器人
    7. 记录已推送论文 ID, 避免重复推送; 支持忽略论文改版号去重
    8. 若筛选结果为空, 静默退出, 不发送空白邮件

使用:
    python arxiv_digest.py                 # 使用默认 config.yaml 运行一次
    python arxiv_digest.py --config xxx.yaml
    python arxiv_digest.py --dry-run        # 不发送邮件, 只打印/记录日志, 不写入已发送记录

邮箱授权码优先级:
    环境变量 EMAIL_PASSWORD(用于 GitHub Actions/GitHub Secrets) > config.yaml 中 email.password(本地运行)

LLM API Key 优先级:
    环境变量 LLM_API_KEY > config.yaml 中 llm.api_key

Webhook 地址优先级:
    环境变量 PUSH_WEBHOOK_URL > config.yaml 中 push.webhook_url
"""

import argparse
import html
import json
import logging
import os
import random
import re
import smtplib
import sys
import time
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr

import arxiv
import requests
import yaml

try:
    from deep_translator import GoogleTranslator
    _TRANSLATOR_AVAILABLE = True
except ImportError:
    _TRANSLATOR_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

logger = logging.getLogger("arxiv_digest")

# Google 翻译单次请求的字符数上限(deep-translator 底层网页版接口的经验限制), 超长文本需分段翻译
_TRANSLATE_CHUNK_SIZE = 4000


# arXiv 分类代码 -> 人类可读名称, 覆盖机器人/控制/AI 相关常见分类, 未收录的分类会原样显示代码
CATEGORY_NAMES = {
    "cs.RO": "Robotics",
    "cs.SY": "Systems and Control",
    "eess.SY": "Systems and Control",
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CL": "Computation and Language",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.MA": "Multiagent Systems",
    "cs.SE": "Software Engineering",
    "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.NI": "Networking and Internet Architecture",
    "cs.CE": "Computational Engineering, Finance, and Science",
    "cs.HC": "Human-Computer Interaction",
    "math.OC": "Optimization and Control",
    "stat.ML": "Machine Learning (Statistics)",
    "eess.SP": "Signal Processing",
    "eess.IV": "Image and Video Processing",
}


def category_label(code: str) -> str:
    """将分类代码转换为 '代码 (人类可读名称)' 形式的标签"""
    name = CATEGORY_NAMES.get(code)
    return f"{code} ({name})" if name else code


def retry_call(func, max_retries: int = 3, base_delay: float = 1.0,
                exceptions=(Exception,), logger_prefix: str = ""):
    """
    通用指数退避重试包装器: 执行 func(), 失败时按指数退避策略重试, 达到最大次数后抛出最后一次异常
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exc = e
            if attempt >= max_retries:
                break
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(f"{logger_prefix}第 {attempt} 次尝试失败: {e}, {delay:.1f}s 后重试")
            time.sleep(delay)
    raise last_exc


def load_config(config_path: str) -> dict:
    """加载并校验 YAML 配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"未找到配置文件: {config_path}\n"
            f"请复制 config.example.yaml 为 config.yaml 并填写你的邮箱等信息。"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError("配置文件为空或格式错误")

    arxiv_cfg = config.get("arxiv", {})
    if not arxiv_cfg.get("categories") and not arxiv_cfg.get("keywords"):
        raise ValueError("config.yaml 中 arxiv.categories 和 arxiv.keywords 不能同时为空")

    return config


def setup_logging(log_file: str) -> None:
    """配置日志: 同时输出到控制台和文件"""
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)



def build_search_query(categories: list, keywords: list) -> str:
    """
    根据分类和关键词构造 arXiv API 查询字符串
    语义: (分类1 OR 分类2 ...) AND (关键词1 OR 关键词2 ...)
    关键词会同时匹配标题(ti)和摘要(abs)
    """
    parts = []

    if categories:
        cat_query = " OR ".join(f"cat:{c.strip()}" for c in categories)
        parts.append(f"({cat_query})")

    if keywords:
        kw_query = " OR ".join(
            f'(ti:"{k.strip()}" OR abs:"{k.strip()}")' for k in keywords
        )
        parts.append(f"({kw_query})")

    if not parts:
        raise ValueError("categories 和 keywords 不能同时为空")

    return " AND ".join(parts)


def fetch_recent_papers(arxiv_cfg: dict) -> list:
    """
    调用 arXiv API 拉取候选论文, 并按 days_back 过滤提交时间
    返回按提交时间降序排列的 arxiv.Result 列表
    """
    query = build_search_query(
        arxiv_cfg.get("categories") or [], arxiv_cfg.get("keywords") or []
    )
    max_results = arxiv_cfg.get("max_results", 50)
    days_back = arxiv_cfg.get("days_back", 1)

    logger.info(f"arXiv 查询语句: {query}")

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    results = []
    try:
        for result in retry_call(
            lambda: list(client.results(search)),
            max_retries=3, base_delay=5.0, logger_prefix="拉取 arXiv 结果 "
        ):
            if result.published >= cutoff:
                results.append(result)
            else:
                # 结果已按提交时间降序排列, 遇到早于 cutoff 的直接停止
                break
    except Exception as e:
        logger.error(f"从 arXiv 拉取论文时出错: {e}")
        raise

    logger.info(f"从 arXiv 拉取到 {len(results)} 篇 {days_back} 天内的候选论文")
    return results


def base_arxiv_id(short_id: str) -> str:
    """去掉论文短 ID 尾部的版本号(如 v1/v2), 用于忽略改版号的去重"""
    return re.sub(r"v\d+$", "", short_id)


def get_dedup_key(paper, ignore_version: bool = True) -> str:
    """计算论文的去重 key: 默认忽略版本号(同一论文的 v1/v2 视为同一篇, 不重复推送)"""
    short_id = paper.get_short_id()
    return base_arxiv_id(short_id) if ignore_version else short_id


def load_sent_ids(sent_ids_file: str) -> dict:
    """加载已发送论文 ID 记录, 格式: {arxiv_id: 首次发送日期字符串}"""
    if not os.path.exists(sent_ids_file):
        return {}
    try:
        with open(sent_ids_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return {k: datetime.now().strftime("%Y-%m-%d") for k in data if isinstance(k, str)}
            if isinstance(data, dict):
                return data
            return {}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"读取已发送记录失败, 将视为空记录: {e}")
        return {}


def save_sent_ids(sent_ids_file: str, sent_ids: dict, retention_days: int) -> None:
    """保存已发送论文 ID 记录, 并清理超过保留期限的旧记录"""
    cutoff = datetime.now() - timedelta(days=retention_days)
    cleaned = {}
    for arxiv_id, sent_date_str in sent_ids.items():
        try:
            sent_date = datetime.strptime(sent_date_str, "%Y-%m-%d")
            if sent_date >= cutoff:
                cleaned[arxiv_id] = sent_date_str
        except ValueError:
            # 记录格式异常, 保留以免误删
            cleaned[arxiv_id] = sent_date_str

    sent_dir = os.path.dirname(sent_ids_file)
    if sent_dir and not os.path.exists(sent_dir):
        os.makedirs(sent_dir, exist_ok=True)

    with open(sent_ids_file, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)



def filter_unsent_papers(papers: list, sent_ids: dict, max_send: int,
                          ignore_version: bool = True) -> list:
    """过滤掉已经发送过的论文(默认忽略版本号去重), 并限制最大发送数量"""
    unsent = [p for p in papers if get_dedup_key(p, ignore_version) not in sent_ids]
    logger.info(f"过滤已发送记录后, 剩余 {len(unsent)} 篇新论文")
    return unsent[:max_send]


def find_matched_keywords(paper, keywords: list) -> list:
    """在标题+摘要中查找命中的关键词(大小写不敏感), 按配置中的顺序返回原始关键词"""
    if not keywords:
        return []
    haystack = f"{paper.title} {paper.summary or ''}"
    matched = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(kw) + r"(?![A-Za-z0-9])", re.IGNORECASE)
        if pattern.search(haystack):
            matched.append(kw)
    return matched


def find_matched_keywords_count(paper, keywords: list) -> dict:
    """统计每个命中关键词在标题+摘要中出现的次数, 用于相关度打分"""
    if not keywords:
        return {}
    haystack = f"{paper.title} {paper.summary or ''}"
    counts = {}
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(kw) + r"(?![A-Za-z0-9])", re.IGNORECASE)
        n = len(pattern.findall(haystack))
        if n:
            counts[kw] = n
    return counts


def contains_excluded_keyword(paper, exclude_keywords: list) -> str:
    """检查论文标题+摘要是否命中任意排除关键词, 命中则返回该关键词, 否则返回空字符串"""
    if not exclude_keywords:
        return ""
    haystack = f"{paper.title} {paper.summary or ''}"
    for kw in exclude_keywords:
        kw = kw.strip()
        if not kw:
            continue
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(kw) + r"(?![A-Za-z0-9])", re.IGNORECASE)
        if pattern.search(haystack):
            return kw
    return ""


def filter_excluded_papers(papers: list, exclude_keywords: list) -> list:
    """剔除命中排除关键词的论文"""
    if not exclude_keywords:
        return papers
    kept = []
    removed = 0
    for p in papers:
        hit = contains_excluded_keyword(p, exclude_keywords)
        if hit:
            removed += 1
            logger.info(f"排除论文 {p.get_short_id()}(命中排除关键词 '{hit}'): {p.title}")
        else:
            kept.append(p)
    if removed:
        logger.info(f"按排除关键词过滤掉 {removed} 篇论文")
    return kept


def compute_relevance_score(paper, keywords: list, keyword_weights: dict = None) -> int:
    """
    计算论文相关度得分: 各命中关键词次数之和(可按 keyword_weights 加权), 标题命中额外加分
    得分越高代表与研究方向越相关, 用于排序
    """
    keyword_weights = keyword_weights or {}
    counts = find_matched_keywords_count(paper, keywords)
    score = 0
    title_lower = (paper.title or "").lower()
    for kw, n in counts.items():
        weight = keyword_weights.get(kw, 1)
        score += n * weight
        if kw.lower() in title_lower:
            score += 3 * weight  # 标题命中权重更高
    return score


def sort_papers_by_relevance(papers: list, keywords: list, keyword_weights: dict = None) -> list:
    """
    按相关度得分从高到低排序, 得分相同则保持按提交时间降序(原始顺序)的相对次序
    """
    if not keywords:
        return papers
    scored = [(compute_relevance_score(p, keywords, keyword_weights), idx, p)
              for idx, p in enumerate(papers)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [p for _, _, p in scored]


def highlight_keywords(text: str, keywords: list) -> str:
    """对已 HTML 转义后的文本, 用 <mark> 高亮命中的关键词(大小写不敏感, 保留原始大小写显示)"""
    escaped_text = html.escape(text)
    if not keywords:
        return escaped_text

    # 按关键词长度从长到短排序, 避免短词(如 USV)提前替换掉长词(如 unmanned surface vessel)的子串
    sorted_kw = sorted({k.strip() for k in keywords if k.strip()}, key=len, reverse=True)
    for kw in sorted_kw:
        escaped_kw = html.escape(kw)
        pattern = re.compile(
            r"(?<![A-Za-z0-9])(" + re.escape(escaped_kw) + r")(?![A-Za-z0-9])", re.IGNORECASE
        )
        escaped_text = pattern.sub(
            r'<mark style="background:#fff3a3;padding:0 2px;border-radius:2px;">\1</mark>',
            escaped_text,
        )
    return escaped_text



def build_bibtex(paper) -> str:
    """为单篇论文生成 BibTeX 引用条目(misc 类型, 适用于 arXiv 预印本)"""
    short_id = paper.get_short_id()
    first_author_last = ""
    if paper.authors:
        name_parts = paper.authors[0].name.strip().split()
        if name_parts:
            first_author_last = re.sub(r"[^A-Za-z]", "", name_parts[-1])
    year = paper.published.strftime("%Y")
    key_suffix = short_id.split("v")[0].replace(".", "")
    bib_key = f"{first_author_last or 'arxiv'}{year}{key_suffix}"

    authors_bib = " and ".join(a.name for a in paper.authors) if paper.authors else "Unknown"
    primary_category = paper.primary_category or (paper.categories[0] if paper.categories else "")

    bibtex = (
        f"@misc{{{bib_key},\n"
        f"      title={{{paper.title}}},\n"
        f"      author={{{authors_bib}}},\n"
        f"      year={{{year}}},\n"
        f"      eprint={{{short_id}}},\n"
        f"      archivePrefix={{arXiv}},\n"
        f"      primaryClass={{{primary_category}}},\n"
        f"      url={{{paper.entry_id}}}\n"
        f"}}"
    )
    return bibtex


def translate_to_chinese(text: str) -> str:
    """
    将英文摘要翻译为中文, 使用 deep-translator(免费的 Google 翻译网页接口)
    失败时(网络问题/接口限流/库缺失等)返回空字符串, 邮件会自动跳过中文翻译区块, 不影响整体发送
    内置指数退避重试, 缓解偶发的网络抖动/限流问题
    """
    if not text or not text.strip():
        return ""
    if not _TRANSLATOR_AVAILABLE:
        logger.warning("未安装 deep-translator, 跳过中文翻译")
        return ""

    try:
        translator = GoogleTranslator(source="en", target="zh-CN")

        def _translate_one(chunk: str) -> str:
            return retry_call(
                lambda: translator.translate(chunk).strip(),
                max_retries=3, base_delay=2.0, logger_prefix="翻译请求 "
            )

        # 超长摘要按句子边界分段翻译, 避免单次请求过长被接口拒绝或截断
        if len(text) <= _TRANSLATE_CHUNK_SIZE:
            return _translate_one(text)

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 > _TRANSLATE_CHUNK_SIZE:
                if current:
                    chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            chunks.append(current)

        translated_parts = [_translate_one(chunk) for chunk in chunks]
        return " ".join(translated_parts)
    except Exception as e:
        logger.warning(f"中文翻译失败, 将只显示英文摘要: {e}")
        return ""


def generate_llm_insight(paper, llm_cfg: dict) -> dict:
    """
    调用 OpenAI 兼容接口(如 DeepSeek/Moonshot/OpenAI 等)对论文摘要生成中文精读要点:
    创新点/方法/结论三段式总结, 用于替代或增强机器翻译
    未启用或调用失败时返回空字典, 邮件会自动回退到普通机器翻译
    """
    if not llm_cfg or not llm_cfg.get("enabled"):
        return {}

    api_key = os.environ.get("LLM_API_KEY") or llm_cfg.get("api_key")
    if not api_key:
        logger.warning("LLM 功能已启用但未配置 api_key, 跳过精读要点生成")
        return {}

    base_url = llm_cfg.get("base_url", "https://api.deepseek.com/v1").rstrip("/")
    model = llm_cfg.get("model", "deepseek-chat")
    timeout = llm_cfg.get("timeout", 30)

    prompt = (
        "你是一名学术助理, 请阅读以下英文论文标题和摘要, 完成两部分总结:\n\n"
        "【第一部分: 大白话人话速读】\n"
        "假设读者完全没有专业背景, 请用最口语化、最直白的大白话(严禁出现学术行话、数学公式、"
        "专业术语和缩写), 写2~3句话, 必须依次讲清楚:\n"
        "  a) 这篇论文实际上是想解决现实中的什么具体痛点/难题;\n"
        "  b) 它想出了什么办法, 这个办法实际用起来有什么好处。\n\n"
        "【第二部分: 专业精读要点】用简洁的中文分别总结:\n"
        "1. 创新点(该研究提出了什么新方法/新发现, 一句话)\n"
        "2. 方法(简述所用的技术路线, 一句话)\n"
        "3. 结论(实验效果或主要结论, 一句话)\n\n"
        "请严格按以下 JSON 格式输出, 不要输出多余文字, 不要使用 markdown 代码块:\n"
        '{"plain_explain": "...", "innovation": "...", "method": "...", "conclusion": "..."}\n\n'
        f"标题: {paper.title}\n摘要: {paper.summary or ''}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一名严谨、简洁的学术论文速读助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def _call():
        resp = requests.post(
            f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()

    try:
        data = retry_call(_call, max_retries=2, base_delay=3.0, logger_prefix="LLM 请求 ")
        content = data["choices"][0]["message"]["content"].strip()
        # 兼容模型偶尔用 ```json 包裹输出的情况
        content = re.sub(r"^```(json)?|```$", "", content, flags=re.MULTILINE).strip()
        result = json.loads(content)
        return {
            "plain_explain": str(result.get("plain_explain", "")).strip(),
            "innovation": str(result.get("innovation", "")).strip(),
            "method": str(result.get("method", "")).strip(),
            "conclusion": str(result.get("conclusion", "")).strip(),
        }
    except Exception as e:
        logger.warning(f"LLM 精读要点生成失败, 将回退到机器翻译: {e}")
        return {}




def build_email_html(papers: list, keywords: list, llm_cfg: dict = None) -> str:
    """将论文列表渲染为学术日报风格的 HTML 邮件正文"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    items_html = []

    for i, p in enumerate(papers, 1):
        short_id = p.get_short_id()
        authors_list = [a.name for a in p.authors] if p.authors else []
        authors_html = html.escape(", ".join(authors_list)) if authors_list else "未知作者"

        primary_cat = p.primary_category or (p.categories[0] if p.categories else "")
        secondary_cats = [c for c in (p.categories or []) if c != primary_cat]
        primary_cat_html = (
            f'<span style="background:#0b5cab;color:#fff;padding:2px 8px;border-radius:10px;'
            f'font-size:12px;margin-right:6px;">{html.escape(category_label(primary_cat))}</span>'
            if primary_cat else ""
        )
        secondary_cats_html = "".join(
            f'<span style="background:#eef3fa;color:#0b5cab;padding:2px 8px;border-radius:10px;'
            f'font-size:12px;margin-right:6px;">{html.escape(category_label(c))}</span>'
            for c in secondary_cats
        )

        published_str = p.published.strftime("%Y-%m-%d %H:%M UTC")
        updated_str = p.updated.strftime("%Y-%m-%d %H:%M UTC") if p.updated else published_str
        is_updated_version = p.updated and p.updated.date() != p.published.date()
        date_line = f"首发日期: {published_str}"
        if is_updated_version:
            date_line += f" &nbsp;|&nbsp; 最近更新: {updated_str}"

        matched_kw = find_matched_keywords(p, keywords)
        score = compute_relevance_score(p, keywords) if keywords else 0
        matched_kw_html = ""
        if matched_kw:
            chips = "".join(
                f'<span style="background:#ffe8a1;color:#7a4a00;padding:2px 8px;border-radius:10px;'
                f'font-size:12px;margin-right:6px;">key {html.escape(kw)}</span>'
                for kw in matched_kw
            )
            score_chip = (
                f'<span style="background:#e6f4ea;color:#1e7a3c;padding:2px 8px;border-radius:10px;'
                f'font-size:12px;margin-right:6px;">相关度 {score}</span>'
            )
            matched_kw_html = f'<p style="margin:8px 0 0 0;">{score_chip}{chips}</p>'

        abstract_raw = re.sub(r"\s+", " ", (p.summary or "")).strip()
        abstract_html = highlight_keywords(abstract_raw, matched_kw)


        insight = generate_llm_insight(p, llm_cfg or {})
        insight_html = ""
        zh_summary_html = ""
        plain_explain_html = ""
        plain_explain = (insight or {}).get("plain_explain", "").strip()
        if plain_explain:
            plain_explain_html = f"""
            <div style="margin-top:12px;padding:14px 16px;background:#fffbe6;border:1.5px solid #ffd666;border-radius:8px;">
                <p style="margin:0 0 6px 0;font-size:13.5px;color:#ad6800;font-weight:bold;">大白话人话速读</p>
                <p style="margin:0;line-height:1.75;font-size:14.5px;color:#333;">
                    {html.escape(plain_explain)}
                </p>
            </div>
            """
        if insight:
            insight_html = f"""
            <div style="margin-top:12px;padding:12px 14px;background:#fff7ec;border-left:3px solid #d98324;border-radius:4px;">
                <p style="margin:0 0 6px 0;font-size:12.5px;color:#d98324;text-transform:uppercase;letter-spacing:0.5px;">AI 精读要点</p>
                <p style="margin:0 0 4px 0;font-size:14px;color:#222;"><strong>创新点:</strong> {html.escape(insight.get('innovation',''))}</p>
                <p style="margin:0 0 4px 0;font-size:14px;color:#222;"><strong>方法:</strong> {html.escape(insight.get('method',''))}</p>
                <p style="margin:0;font-size:14px;color:#222;"><strong>结论:</strong> {html.escape(insight.get('conclusion',''))}</p>
            </div>
            """
        else:
            zh_summary = translate_to_chinese(abstract_raw)
            if zh_summary:
                zh_summary_html = f"""
                <div style="margin-top:12px;padding:12px 14px;background:#f0f7ff;border-left:3px solid #2f8f4e;border-radius:4px;">
                    <p style="margin:0 0 6px 0;font-size:12.5px;color:#2f8f4e;text-transform:uppercase;letter-spacing:0.5px;">中文摘要翻译</p>
                    <p style="margin:0;line-height:1.75;font-size:14px;color:#222;">
                        {html.escape(zh_summary)}
                    </p>
                </div>
                """

        bibtex = build_bibtex(p)
        bibtex_html = html.escape(bibtex)

        comment_html = (
            f'<p style="margin:4px 0;color:#777;font-size:13px;">备注: {html.escape(p.comment)}</p>'
            if getattr(p, "comment", None) else ""
        )
        journal_ref_html = (
            f'<p style="margin:4px 0;color:#777;font-size:13px;">期刊/会议信息: {html.escape(p.journal_ref)}</p>'
            if getattr(p, "journal_ref", None) else ""
        )


        items_html.append(f"""
        <div style="margin-bottom:30px;padding:18px 20px;border:1px solid #e2e6ea;border-radius:8px;background:#fafbfc;">
            <h3 style="margin:0 0 10px 0;font-size:17px;line-height:1.4;">
                {i}. <a href="{p.entry_id}" style="color:#0b5cab;text-decoration:none;">{html.escape(p.title)}</a>
            </h3>
            <p style="margin:4px 0;color:#333;font-size:13.5px;line-height:1.6;"><strong>作者:</strong> {authors_html}</p>
            <p style="margin:6px 0;color:#555;font-size:13px;">
                {primary_cat_html}{secondary_cats_html}
            </p>
            <p style="margin:4px 0;color:#555;font-size:13px;">
                <strong>arXiv 编号:</strong> {short_id} &nbsp;|&nbsp; {date_line}
            </p>
            <p style="margin:4px 0;color:#555;font-size:13px;">
                <a href="{p.entry_id}" style="color:#0b5cab;">arXiv 详情页</a> &nbsp;|&nbsp;
                <a href="{p.pdf_url}" style="color:#0b5cab;">PDF 直达</a>
            </p>
            {comment_html}
            {journal_ref_html}
            {matched_kw_html}
            {plain_explain_html}
            {insight_html}
            {zh_summary_html}
            <div style="margin-top:12px;padding:12px 14px;background:#ffffff;border-left:3px solid #0b5cab;border-radius:4px;">
                <p style="margin:0 0 6px 0;font-size:12.5px;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Abstract</p>
                <p style="margin:0;line-height:1.75;font-size:14px;color:#222;text-align:justify;font-family:Georgia, 'Times New Roman', serif;">
                    {abstract_html}
                </p>
            </div>
            <details style="margin-top:12px;">
                <summary style="cursor:pointer;color:#0b5cab;font-size:13px;">BibTeX 引用(点击展开/折叠)</summary>
                <pre style="background:#2d2d2d;color:#e6e6e6;padding:12px;border-radius:6px;overflow-x:auto;
                            font-size:12.5px;line-height:1.6;margin-top:8px;white-space:pre-wrap;word-break:break-all;">{bibtex_html}</pre>
            </details>
        </div>
        """)

    body = f"""
    <html>
    <body style="font-family:'Segoe UI', Arial, sans-serif;color:#222;max-width:860px;margin:0 auto;padding:16px;">
        <h2 style="margin:0 0 4px 0;">无人船学术速递 - {today_str}</h2>
        <p style="color:#666;margin:0 0 20px 0;">本次共为你筛选出 <strong>{len(papers)}</strong> 篇新论文, 已按相关度/提交时间排序</p>
        {''.join(items_html)}
        <p style="color:#999;font-size:12px;margin-top:24px;border-top:1px solid #eee;padding-top:12px;">
            本邮件由 arxiv_digest 定时任务自动生成并发送, 数据来源于 arXiv 官方 API
        </p>
    </body>
    </html>
    """
    return body



def build_plain_text_body(papers: list, keywords: list) -> str:
    """生成纯文本版邮件正文, 作为 HTML 邮件的兜底(部分邮件客户端/安全策略会屏蔽 HTML)"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    lines = [f"无人船学术速递 - {today_str}", f"本次共筛选出 {len(papers)} 篇新论文", ""]
    for i, p in enumerate(papers, 1):
        authors = ", ".join(a.name for a in p.authors) if p.authors else "未知作者"
        matched_kw = find_matched_keywords(p, keywords)
        lines.append(f"[{i}] {p.title}")
        lines.append(f"作者: {authors}")
        lines.append(f"分类: {category_label(p.primary_category)}")
        lines.append(f"arXiv: {p.get_short_id()} | {p.entry_id}")
        if matched_kw:
            lines.append(f"命中关键词: {', '.join(matched_kw)}")
        lines.append(f"摘要: {(p.summary or '').strip()}")
        lines.append("-" * 60)
    lines.append("本邮件由 arxiv_digest 定时任务自动生成, 数据来源于 arXiv 官方 API")
    return "\n".join(lines)


def _normalize_receivers(receiver_cfg) -> list:
    """将 email.receiver 归一化为列表: 支持单个字符串或字符串列表两种写法"""
    if not receiver_cfg:
        return []
    if isinstance(receiver_cfg, str):
        return [r.strip() for r in receiver_cfg.split(",") if r.strip()]
    if isinstance(receiver_cfg, list):
        return [str(r).strip() for r in receiver_cfg if str(r).strip()]
    return []



def send_email(email_cfg: dict, papers: list, keywords: list, llm_cfg: dict = None) -> None:
    """通过 SMTP 发送论文摘要邮件(HTML + 纯文本兜底), 支持多收件人, 发送失败自动重试"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    subject_prefix = email_cfg.get("subject_prefix", "[arXiv每日论文摘要]")
    subject = f"{subject_prefix} {today_str} ({len(papers)}篇)"

    receivers = _normalize_receivers(email_cfg.get("receiver"))
    if not receivers:
        raise ValueError("email.receiver 未配置有效的收件人地址")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = Header(subject, "utf-8")
    sender_name = email_cfg.get("sender_name", "arXiv 每日论文摘要")
    msg["From"] = formataddr((str(Header(sender_name, "utf-8")), email_cfg["sender"]))
    msg["To"] = ", ".join(receivers)

    plain_body = build_plain_text_body(papers, keywords)
    html_body = build_email_html(papers, keywords, llm_cfg)
    msg.attach(MIMEText(plain_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    # 授权码优先读取环境变量 EMAIL_PASSWORD(适配 GitHub Actions Secrets), 本地运行时退回 config.yaml 中的配置
    password = os.environ.get("EMAIL_PASSWORD") or email_cfg.get("password")
    if not password:
        raise ValueError(
            "未找到邮箱授权码: 请设置环境变量 EMAIL_PASSWORD, 或在 config.yaml 的 email.password 中填写"
        )

    smtp_server = email_cfg["smtp_server"]
    smtp_port = email_cfg["smtp_port"]
    use_ssl = email_cfg.get("use_ssl", True)

    def _send():
        logger.info(f"正在连接 SMTP 服务器 {smtp_server}:{smtp_port} ...")
        if use_ssl:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
            server.starttls()
        try:
            server.login(email_cfg["sender"], password)
            server.sendmail(email_cfg["sender"], receivers, msg.as_string())
        finally:
            server.quit()

    retry_call(_send, max_retries=3, base_delay=5.0, logger_prefix="邮件发送 ")
    logger.info(f"邮件发送成功, 收件人: {', '.join(receivers)}")



def _build_webhook_text(papers: list, keywords: list, max_items: int = 10) -> str:
    """构造用于企业微信/飞书/Slack 等 Webhook 推送的简要文本摘要(避免消息过长)"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    lines = [f"arXiv 论文速递 {today_str} (共 {len(papers)} 篇, 显示前 {min(max_items, len(papers))} 篇)"]
    for i, p in enumerate(papers[:max_items], 1):
        matched_kw = find_matched_keywords(p, keywords)
        kw_tag = f" [{', '.join(matched_kw)}]" if matched_kw else ""
        lines.append(f"{i}. {p.title}{kw_tag}")
        lines.append(f"   {p.entry_id}")
    return "\n".join(lines)


def push_to_webhook(push_cfg: dict, papers: list, keywords: list) -> None:
    """
    将论文摘要推送到 Webhook 机器人, 支持企业微信(wecom)/飞书(feishu)/Slack(slack)三种格式,
    通过 push.channel 指定渠道, 默认按企业微信 Markdown 消息格式发送
    失败不影响主流程(邮件已发送), 仅记录警告日志
    """
    if not push_cfg or not push_cfg.get("enabled"):
        return

    webhook_url = os.environ.get("PUSH_WEBHOOK_URL") or push_cfg.get("webhook_url")
    if not webhook_url:
        logger.warning("push.enabled=true 但未配置 webhook_url, 跳过 Webhook 推送")
        return

    channel = (push_cfg.get("channel") or "wecom").lower()
    max_items = push_cfg.get("max_items", 10)
    text = _build_webhook_text(papers, keywords, max_items)

    if channel == "slack":
        payload = {"text": text}
    elif channel == "feishu":
        payload = {"msg_type": "text", "content": {"text": text}}
    else:
        # 企业微信机器人 Markdown 消息
        payload = {"msgtype": "markdown", "markdown": {"content": text}}

    def _post():
        resp = requests.post(webhook_url, json=payload, timeout=15)
        resp.raise_for_status()
        return resp

    try:
        retry_call(_post, max_retries=2, base_delay=2.0, logger_prefix="Webhook 推送 ")
        logger.info(f"Webhook({channel}) 推送成功")
    except Exception as e:
        logger.warning(f"Webhook({channel}) 推送失败, 不影响邮件发送结果: {e}")


def print_papers_to_console(papers: list) -> None:
    """dry-run 模式下, 将论文摘要打印到控制台/日志(完整摘要, 不截断)"""
    for i, p in enumerate(papers, 1):
        authors = ", ".join(a.name for a in p.authors)
        logger.info(f"--- [{i}/{len(papers)}] {p.get_short_id()} ---")
        logger.info(f"标题: {p.title}")
        logger.info(f"作者: {authors}")
        logger.info(f"主分类: {category_label(p.primary_category)} | 分类: {', '.join(p.categories)}")
        logger.info(f"提交时间: {p.published}")
        logger.info(f"链接: {p.entry_id} | PDF: {p.pdf_url}")
        logger.info(f"摘要: {(p.summary or '').strip()}")



def run(config_path: str, dry_run: bool = False) -> int:
    """主流程, 返回退出码(0 成功, 非 0 表示失败)"""
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        # 配置未就绪时日志系统可能还没初始化, 直接打印到控制台
        print(f"[ERROR] {e}")
        return 1

    logging_cfg = config.get("logging", {})
    log_file = logging_cfg.get("log_file", os.path.join(BASE_DIR, "logs", "arxiv_digest.log"))
    if not os.path.isabs(log_file):
        log_file = os.path.join(BASE_DIR, log_file)
    setup_logging(log_file)

    logger.info("=" * 60)
    logger.info(f"开始执行 arXiv 每日论文摘要任务 (dry_run={dry_run})")

    arxiv_cfg = config.get("arxiv", {})
    email_cfg = config.get("email", {})
    storage_cfg = config.get("storage", {})
    llm_cfg = config.get("llm", {})
    push_cfg = config.get("push", {})

    sent_ids_file = storage_cfg.get("sent_ids_file", "sent_ids.json")
    if not os.path.isabs(sent_ids_file):
        sent_ids_file = os.path.join(BASE_DIR, sent_ids_file)
    retention_days = storage_cfg.get("sent_ids_retention_days", 14)
    ignore_version = storage_cfg.get("ignore_version_in_dedup", True)

    try:
        papers = fetch_recent_papers(arxiv_cfg)
    except Exception as e:
        logger.error(f"任务失败: 拉取论文出错 - {e}")
        return 1

    exclude_keywords = arxiv_cfg.get("exclude_keywords") or []
    papers = filter_excluded_papers(papers, exclude_keywords)

    keywords = arxiv_cfg.get("keywords") or []
    keyword_weights = arxiv_cfg.get("keyword_weights") or {}
    papers = sort_papers_by_relevance(papers, keywords, keyword_weights)

    sent_ids = load_sent_ids(sent_ids_file)
    max_send = arxiv_cfg.get("max_send", 20)
    new_papers = filter_unsent_papers(papers, sent_ids, max_send, ignore_version)

    if not new_papers:
        logger.info("没有新论文需要推送, 任务结束")
        return 0

    if dry_run:
        logger.info("[dry-run 模式] 不会发送邮件, 也不会写入已发送记录")
        print_papers_to_console(new_papers)
        return 0

    if email_cfg.get("enabled", True):
        try:
            send_email(email_cfg, new_papers, keywords, llm_cfg)
        except Exception as e:
            logger.error(f"任务失败: 邮件发送出错 - {e}")
            return 1
    else:
        logger.info("邮件发送已在配置中关闭(email.enabled=false), 仅打印摘要")
        print_papers_to_console(new_papers)

    push_to_webhook(push_cfg, new_papers, keywords)

    today_str = datetime.now().strftime("%Y-%m-%d")
    for p in new_papers:
        sent_ids[get_dedup_key(p, ignore_version)] = today_str
    save_sent_ids(sent_ids_file, sent_ids, retention_days)

    logger.info(f"任务完成, 本次共推送 {len(new_papers)} 篇论文")
    return 0



def main():
    parser = argparse.ArgumentParser(description="arXiv 每日论文摘要推送脚本")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="配置文件路径, 默认使用同目录下的 config.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印筛选结果, 不发送邮件, 不写入已发送记录(用于调试)",
    )
    args = parser.parse_args()

    exit_code = run(args.config, dry_run=args.dry_run)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

