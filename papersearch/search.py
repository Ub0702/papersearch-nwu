"""检索层：统一 Source 接口，实现 Semantic Scholar 与 arXiv 检索。

两个数据源均为免费公开 API，无需 Key：
- Semantic Scholar: https://api.semanticscholar.org/graph/v1/paper/search
- arXiv:          https://export.arxiv.org/api/query (Atom XML)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Iterable

import requests
from xml.etree import ElementTree as ET

from .models import Paper

#: Semantic Scholar 官方建议的请求间隔（秒），避免被限流
S2_RATE_LIMIT_SECONDS = 1.0
ARXIV_RATE_LIMIT_SECONDS = 3.0

_TIMEOUT = 30
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
#: 429 限流时的重试等待（秒）与次数
_RETRY_WAITS = (3, 5)
_HEADERS = {"User-Agent": "PaperSearch/0.1 (academic paper search; contact: user@example.com)"}


class Source(ABC):
    """数据源抽象接口：新增数据源只需继承并实现 search()。"""

    name = "base"

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[Paper]:
        """按关键词检索，返回论文列表（按相关度降序）。"""


class SemanticScholarSource(Source):
    """Semantic Scholar 数据源（2 亿+ 论文，覆盖大部分期刊/会议）。

    免费注册 https://www.semanticscholar.org/product/api 获取 Key 后可大幅
    提升限流额度；不填 Key 也能用（免费层可能被限流，会由 OpenAlex 兜底）。
    """

    name = "semantic_scholar"
    _search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    _fields = "title,abstract,authors,year,url,externalIds,openAccessPdf"

    def __init__(self, rate_limit: float = S2_RATE_LIMIT_SECONDS, api_key: str | None = None):
        self._rate_limit = rate_limit
        self._api_key = api_key

    def search(self, query: str, limit: int = 10) -> list[Paper]:
        params = {"query": query, "limit": limit, "fields": self._fields}
        headers = dict(_HEADERS)
        if self._api_key:
            headers["x-api-key"] = self._api_key
        resp = _request_with_retry(self._search_url, params, headers=headers)
        data = resp.json()
        papers: list[Paper] = []
        for i, item in enumerate(data.get("data", [])):
            title = (item.get("title") or "").strip()
            if not title:
                continue
            authors = [a.get("name", "") for a in (item.get("authors") or []) if a.get("name")]
            ext = item.get("externalIds") or {}
            open_pdf = item.get("openAccessPdf") or {}
            papers.append(
                Paper(
                    title=title,
                    authors=[n for n in authors if n],
                    year=item.get("year"),
                    abstract=(item.get("abstract") or "").strip(),
                    url=item.get("url") or "",
                    pdf_url=open_pdf.get("url"),
                    source=self.name,
                    doi=ext.get("DOI"),
                    relevance=max(0.0, 1.0 - i / max(limit, 1)),
                )
            )
        time.sleep(self._rate_limit)  # 遵守限流
        return papers


class ArxivSource(Source):
    """arXiv 数据源（预印本主力，计算机/物理/数学最新进展）。"""

    name = "arxiv"
    _query_url = "http://export.arxiv.org/api/query"

    def __init__(self, rate_limit: float = ARXIV_RATE_LIMIT_SECONDS):
        self._rate_limit = rate_limit

    def search(self, query: str, limit: int = 10) -> list[Paper]:
        params = {
            "search_query": f'all:"{query}"',
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        resp = requests.get(self._query_url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        papers: list[Paper] = []
        for i, entry in enumerate(root.findall("atom:entry", _ARXIV_NS)):
            title = " ".join((entry.findtext("atom:title", "", _ARXIV_NS) or "").split())
            if not title:
                continue
            authors = [
                a.findtext("atom:name", "", _ARXIV_NS)
                for a in entry.findall("atom:author", _ARXIV_NS)
            ]
            authors = [a for a in authors if a]
            summary = " ".join((entry.findtext("atom:summary", "", _ARXIV_NS) or "").split())
            link = entry.findtext("atom:id", "", _ARXIV_NS).strip()
            pdf_url = None
            for l in entry.findall("atom:link", _ARXIV_NS):
                if l.get("title") == "pdf":
                    pdf_url = l.get("href")
                    break
            papers.append(
                Paper(
                    title=title,
                    authors=authors,
                    year=_parse_arxiv_year(entry.findtext("atom:published", "", _ARXIV_NS)),
                    abstract=summary,
                    url=link,
                    pdf_url=pdf_url,
                    source=self.name,
                    doi=None,
                    relevance=max(0.0, 1.0 - i / max(limit, 1)),
                )
            )
        time.sleep(self._rate_limit)
        return papers


def _request_with_retry(url: str, params: dict, headers: dict | None = None) -> requests.Response:
    """带 429 重试的 GET 请求。"""
    for attempt, wait in enumerate(_RETRY_WAITS):
        resp = requests.get(url, params=params, timeout=_TIMEOUT, headers=headers or _HEADERS)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        print(f"[warn] {url.split('/')[2]} 限流(429)，{wait}s 后重试...")
        time.sleep(wait)
    resp.raise_for_status()
    return resp


class OpenAlexSource(Source):
    """OpenAlex 数据源（2.5 亿+ 学术作品，免费、无需 Key、限流宽松）。"""

    name = "openalex"
    _works_url = "https://api.openalex.org/works"

    def __init__(self, mailto: str = ""):
        self._mailto = mailto  # 填邮箱可进更高限流池，可选

    def search(self, query: str, limit: int = 10) -> list[Paper]:
        params = {
            "search": query,
            "per-page": limit,
            "sort": "relevance_score:desc",
            "select": "id,display_name,authorships,publication_year,"
                      "abstract_inverted_index,doi,open_access,relevance_score",
        }
        if self._mailto:
            params["mailto"] = self._mailto
        resp = _request_with_retry(self._works_url, params)
        data = resp.json()
        results = data.get("results", [])
        max_score = max(
            (float(r.get("relevance_score") or 0.0) for r in results), default=1.0
        )
        papers: list[Paper] = []
        for item in results:
            title = (item.get("display_name") or "").strip()
            if not title:
                continue
            authors = [
                a.get("author", {}).get("display_name", "")
                for a in (item.get("authorships") or [])
            ]
            oa = item.get("open_access") or {}
            doi = item.get("doi")  # 形如 https://doi.org/10.xxxx
            papers.append(
                Paper(
                    title=title,
                    authors=[n for n in authors if n],
                    year=item.get("publication_year"),
                    abstract=_inverted_to_text(item.get("abstract_inverted_index")),
                    url=doi or "",
                    pdf_url=oa.get("oa_url"),
                    source=self.name,
                    doi=doi.replace("https://doi.org/", "") if doi else None,
                    # 归一化到 0~1，与其他数据源可比
                    relevance=max(0.0, min(1.0, float(item.get("relevance_score") or 0.0) / max_score)),
                )
            )
        return papers


def _s2_key() -> str | None:
    """从环境变量读取 Semantic Scholar API Key（可选）。"""
    import os
    return os.getenv("PAPERSEARCH_S2_API_KEY") or None


def _inverted_to_text(inverted: dict | None) -> str:
    """OpenAlex 的摘要以倒排索引存储，还原为正文。"""
    if not inverted:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inverted.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def _parse_arxiv_year(published: str) -> int | None:
    try:
        return int(published[:4])
    except (ValueError, TypeError):
        return None


def search_all(
    query: str,
    limit_per_source: int = 10,
    sources: Iterable[Source] | None = None,
) -> list[Paper]:
    """聚合多个数据源的检索结果，按相关度降序、去重后返回。"""
    if sources is None:
        sources = [
            SemanticScholarSource(api_key=_s2_key()),
            OpenAlexSource(),
            ArxivSource(),
        ]
    merged: dict[str, Paper] = {}
    for src in sources:
        try:
            results = src.search(query, limit=limit_per_source)
        except requests.RequestException as exc:
            print(f"[warn] {src.name} 检索失败（已跳过）：{exc}")
            continue
        for paper in results:
            key = paper.title.lower().strip()
            if key and key not in merged:
                merged[key] = paper
    return sorted(merged.values(), key=lambda p: p.relevance, reverse=True)
