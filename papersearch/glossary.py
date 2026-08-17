"""术语表：保证学术专业名词翻译一致性。

机制：翻译前把术语替换为占位符（保护），翻译完成后还原。
这样无论翻译多少次，术语的中文译名都由术语表统一决定。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

#: 默认术语表路径（项目仓库内 data/glossary.json）
DEFAULT_GLOSSARY_PATH = Path(__file__).resolve().parent.parent / "data" / "glossary.json"


class Glossary:
    """学术术语表，支持保护/还原与双语查询。"""

    def __init__(self, terms: dict[str, str] | None = None):
        #: 小写英文术语 -> 中文译名
        self._en2zh: dict[str, str] = {}
        #: 占位符编号 -> 原始术语（小写）
        self._placeholders: dict[str, str] = {}
        self._seq = 0
        for en, zh in (terms or {}).items():
            self._en2zh[en.strip().lower()] = zh.strip()

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Glossary":
        path = Path(path) if path else DEFAULT_GLOSSARY_PATH
        if not path.exists():
            return cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._en2zh, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 术语操作
    # ------------------------------------------------------------------
    def add(self, en: str, zh: str) -> None:
        self._en2zh[en.strip().lower()] = zh.strip()

    def translate_term(self, en: str) -> str | None:
        """查询单个术语的中文译名（大小写不敏感）。"""
        return self._en2zh.get(en.strip().lower())

    @property
    def size(self) -> int:
        return len(self._en2zh)

    # ------------------------------------------------------------------
    # 保护 / 还原
    # ------------------------------------------------------------------
    def protect(self, text: str) -> str:
        """把文本中出现的术语替换为 [[T{n}]] 占位符，避免翻译引擎改动术语。

        注意：长术语优先匹配，避免短术语吞掉长术语。
        """
        self._placeholders.clear()
        self._seq = 0
        terms = sorted(self._en2zh.keys(), key=len, reverse=True)
        for term in terms:
            pattern = re.compile(r"(?<![A-Za-z])" + re.escape(term) + r"(?![A-Za-z])", re.IGNORECASE)
            text = pattern.sub(lambda m: self._remember(term), text)
        return text

    def restore(self, text: str) -> str:
        """把 [[T{n}]] 占位符还原为术语表对应的中文译名。"""
        for placeholder, term in self._placeholders.items():
            text = text.replace(placeholder, self._en2zh.get(term, term))
        return text

    def _remember(self, term: str) -> str:
        placeholder = f"[[T{self._seq}]]"
        self._seq += 1
        self._placeholders[placeholder] = term
        return placeholder
