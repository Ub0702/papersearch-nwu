"""论文数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Paper:
    """一篇论文的统一表示（跨数据源）。"""

    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    url: str = ""
    pdf_url: str | None = None
    source: str = "unknown"  # semantic_scholar / arxiv
    doi: str | None = None
    relevance: float = 0.0  # 0~1，越高越相关

    @property
    def authors_text(self) -> str:
        if not self.authors:
            return "N/A"
        return ", ".join(self.authors[:6]) + (" et al." if len(self.authors) > 6 else "")

    @property
    def year_text(self) -> str:
        return str(self.year) if self.year else "N/A"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "source": self.source,
            "doi": self.doi,
            "relevance": round(self.relevance, 3),
        }
