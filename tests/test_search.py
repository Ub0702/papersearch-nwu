"""search_all 排序与去重逻辑的单元测试（不依赖真实网络）。"""

import pytest

from papersearch.models import Paper
from papersearch.search import Source, search_all


class _FakeSource(Source):
    """固定返回预设论文列表的假数据源。"""

    name = "fake"

    def __init__(self, papers: list[Paper]):
        self._papers = papers

    def search(self, query: str, limit: int = 10) -> list[Paper]:
        return self._papers[:limit]


def _paper(title: str, year: int | None, relevance: float) -> Paper:
    return Paper(title=title, year=year, relevance=relevance, source="fake")


def test_sort_by_relevance_default():
    """默认按相关度降序（即使输入顺序是乱的）。"""
    papers = [
        _paper("low", 2025, 0.3),
        _paper("high", 2020, 0.9),
        _paper("mid", 2023, 0.6),
    ]
    result = search_all("query", sources=[_FakeSource(papers)])
    assert [p.title for p in result] == ["high", "mid", "low"]


def test_sort_by_date():
    """sort='date' 时按年份降序，最新在前。"""
    papers = [
        _paper("old", 2019, 0.9),
        _paper("new", 2025, 0.2),
        _paper("mid", 2023, 0.5),
    ]
    result = search_all("query", sources=[_FakeSource(papers)], sort="date")
    assert [p.title for p in result] == ["new", "mid", "old"]


def test_sort_by_date_missing_year_last():
    """年份缺失的论文排在最后，且年份之间仍按新->旧。"""
    papers = [
        _paper("no-year", None, 0.9),
        _paper("old", 2018, 0.8),
        _paper("new", 2024, 0.1),
    ]
    result = search_all("query", sources=[_FakeSource(papers)], sort="date")
    assert [p.title for p in result] == ["new", "old", "no-year"]


def test_merge_deduplicate_across_sources():
    """跨数据源按标题去重，保留第一个出现的。"""
    dup = _paper("same title", 2024, 0.8)
    result = search_all(
        "query",
        sources=[
            _FakeSource([dup, _paper("a", 2020, 0.5)]),
            _FakeSource([Paper(title="Same Title ", year=2023, relevance=0.7)]),
        ],
    )
    assert len(result) == 2
    assert sum(1 for p in result if p.title.lower().startswith("same title")) == 1


def test_invalid_sort_raises():
    with pytest.raises(ValueError, match="排序方式"):
        search_all("query", sources=[_FakeSource([])], sort="popularity")
