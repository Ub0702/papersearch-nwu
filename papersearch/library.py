"""本地文献库管理（M4）。

核心能力：
- 扫描本地文件夹，把 PDF 论文导入 SQLite 文献库
- 用 PyMuPDF 提取元数据（标题 / 作者 / 年份 / arXiv ID / 摘要，启发式尽力而为）
- 提取全文，支持库内全文搜索（LIKE 匹配，中英文通用）

数据默认存在 ~/.papersearch/library.db，可通过 db_path 参数覆盖。
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

try:
    import pymupdf as fitz  # PyMuPDF 1.24+ 新导入名，避免 fitz 弃用警告
except ImportError:
    try:
        import fitz  # type: ignore[no-redef]  旧版本 PyMuPDF
    except ImportError:  # pragma: no cover
        fitz = None


class LibraryError(RuntimeError):
    """文献库可预期错误（缺依赖 / 路径非法 / 数据库损坏等）。"""


def default_db_path() -> Path:
    """默认数据库位置：~/.papersearch/library.db"""
    return Path.home() / ".papersearch" / "library.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path    TEXT    NOT NULL UNIQUE,
    title        TEXT,
    authors      TEXT,              -- JSON 数组字符串
    year         INTEGER,
    abstract     TEXT,
    journal      TEXT,
    arxiv_id     TEXT,
    text_content TEXT,              -- 提取的全文（供搜索）
    size_bytes   INTEGER,
    added_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_papers_year  ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title);
"""


# ---------------------------------------------------------------------------
# 数据库基础
# ---------------------------------------------------------------------------

def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """打开（必要时创建）文献库数据库，返回连接。"""
    path = Path(db_path) if db_path else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)  # schema 含多条语句，需 executescript
    return conn


@dataclass
class LibraryPaper:
    """文献库中的一篇论文（来自数据库行）。"""

    id: int
    file_path: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    journal: str = ""
    arxiv_id: str = ""
    size_bytes: int = 0
    added_at: str = ""

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "LibraryPaper":
        authors_raw = row["authors"]
        try:
            authors = json.loads(authors_raw) if authors_raw else []
        except (json.JSONDecodeError, TypeError):
            authors = [a.strip() for a in str(authors_raw or "").split(",") if a.strip()]
        return cls(
            id=row["id"],
            file_path=row["file_path"],
            title=row["title"] or Path(row["file_path"]).stem,
            authors=authors,
            year=row["year"],
            abstract=row["abstract"] or "",
            journal=row["journal"] or "",
            arxiv_id=row["arxiv_id"] or "",
            size_bytes=row["size_bytes"] or 0,
            added_at=row["added_at"] or "",
        )

    @property
    def authors_text(self) -> str:
        if not self.authors:
            return "N/A"
        return ", ".join(self.authors[:6]) + (" et al." if len(self.authors) > 6 else "")

    @property
    def year_text(self) -> str:
        return str(self.year) if self.year else "N/A"

    @property
    def size_text(self) -> str:
        kb = self.size_bytes / 1024
        return f"{kb / 1024:.1f} MB" if kb > 1024 else f"{kb:.0f} KB"


# ---------------------------------------------------------------------------
# PDF 元数据提取（启发式，尽力而为）
# ---------------------------------------------------------------------------

_ARXIV_RE = re.compile(r"arXiv\s*:\s*(\d{4}\.\d{4,5})", re.IGNORECASE)
# 年份：19xx/20xx。不用尾部词边界，因为 creationDate 形如 D:20230101000000（年份后紧跟数字）
_YEAR_RE = re.compile(r"(?<!\d)(19|20)\d{2}")


def _parse_year(text: str) -> int | None:
    """从文本中提取第一个合理年份。"""
    m = _YEAR_RE.search(text or "")
    return int(m.group(0)) if m else None


def _parse_arxiv_id(text: str) -> str | None:
    """提取 arXiv ID（需要出现 'arXiv' 字样，避免误匹配普通数字）。"""
    m = _ARXIV_RE.search(text or "")
    return m.group(1) if m else None


def _skip_lines(lines: list[str]) -> list[str]:
    """过滤掉明显不是标题/作者的内容行（页码、URL、arXiv 标识等）。"""
    noise = ("arxiv", "http", "https", "www.", "submitted", "received",
             "abstract", "introduction", "corresponding", "@")
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if len(s) < 4:
            continue
        low = s.lower()
        if any(low.startswith(n) for n in noise):
            continue
        if re.fullmatch(r"[\d\s\-—–./:]+", s):  # 纯页码/日期/编号
            continue
        out.append(s)
    return out


def _guess_title(first_page_text: str) -> str | None:
    """标题启发式：首页第一个看起来像标题的非噪音行。"""
    lines = _skip_lines((first_page_text or "").splitlines())
    if not lines:
        return None
    title = lines[0]
    # 标题通常不长；若首行过长，截断到合理长度
    if len(title) > 200:
        title = title[:200].rsplit(" ", 1)[0]
    return title


def _guess_authors(first_page_text: str) -> list[str]:
    """作者启发式：标题之后的第一个非噪音行，含多个姓名分隔符时视为作者行。"""
    raw_lines = (first_page_text or "").splitlines()
    lines = _skip_lines(raw_lines)
    if len(lines) < 2:
        return []
    # 标题是 lines[0]，候选作者行在 lines[1]（最多往后看 3 行）
    for cand in lines[1:4]:
        if _YEAR_RE.search(cand):  # 含年份说明已经到正文/机构区
            break
        # 作者行特征：逗号 / and / & 分隔，多为 2~6 个姓名
        if re.search(r",\s|\band\b|&", cand) and len(cand) < 200:
            parts = re.split(r",|\band\b|&", cand)
            names = [p.strip() for p in parts if p.strip()]
            if 2 <= len(names) <= 8 and all(" " not in n or n.count(" ") <= 3 for n in names):
                return names
    return []


def _guess_abstract(first_page_text: str) -> str:
    """摘要启发式：'Abstract' 段落之后的行。"""
    m = re.search(r"\babstract\b", first_page_text or "", re.IGNORECASE)
    if not m:
        return ""
    tail = first_page_text[m.end():]
    # 截到第一个空白行或 '1 Introduction' 之类的节标题
    tail = re.split(r"\n\s*\n|\b1\s+introduction\b", tail, flags=re.IGNORECASE)[0]
    return " ".join(tail.split())[:2000]


def extract_pdf_info(path: str | Path) -> dict:
    """提取 PDF 的元数据与全文。返回 dict，字段可能为空（尽力而为）。"""
    if fitz is None:
        raise LibraryError("缺少依赖 PyMuPDF，请先安装: pip install pymupdf")
    path = Path(path)
    if not path.exists():
        raise LibraryError(f"文件不存在: {path}")
    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # 损坏 / 加密 / 非 PDF
        raise LibraryError(f"无法打开 PDF: {exc}") from exc
    try:
        meta = doc.metadata or {}
        first_text = doc[0].get_text() if doc.page_count > 0 else ""
        # 全文提取：最多 200 页，防止超大 PDF 拖垮扫描
        page_count = min(doc.page_count, 200)
        text_content = "\n".join(doc[i].get_text() for i in range(page_count))
    finally:
        doc.close()

    title = (meta.get("title") or "").strip() or None
    authors_raw = (meta.get("author") or "").strip() or None
    year = _parse_year(first_text + (meta.get("creationDate") or ""))
    arxiv_id = _parse_arxiv_id(first_text)
    abstract = _guess_abstract(first_text)
    if not title:
        title = _guess_title(first_text)
    authors: list[str] = []
    if authors_raw:
        authors = [a.strip() for a in authors_raw.split(";") if a.strip()]
    else:
        authors = _guess_authors(first_text)

    return {
        "file_path": str(path),
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": abstract,
        "journal": (meta.get("subject") or "").strip(),
        "arxiv_id": arxiv_id,
        "text_content": text_content,
        "size_bytes": path.stat().st_size,
    }


# ---------------------------------------------------------------------------
# 扫描导入
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    """一次扫描导入的统计结果。"""

    total: int = 0
    added: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"扫描 {self.total} 个 PDF：新增 {self.added}，跳过（已存在）{self.skipped}，"
            f"失败 {self.failed}"
        )


def scan_directory(
    directory: str | Path,
    db_path: str | Path | None = None,
    recursive: bool = True,
    progress=None,
) -> ScanResult:
    """扫描文件夹中的所有 PDF 并导入文献库。

    progress: 可选回调 progress(processed: int, total: int, current_path: str)。
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise LibraryError(f"目录不存在: {directory}")
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdfs = sorted(directory.glob(pattern))

    result = ScanResult(total=len(pdfs))
    conn = _connect(db_path)
    try:
        for i, pdf in enumerate(pdfs, 1):
            if progress:
                progress(i, len(pdfs), str(pdf))
            try:
                info = extract_pdf_info(pdf)
            except LibraryError as exc:
                result.failed += 1
                result.errors.append(f"{pdf}: {exc}")
                continue
            try:
                cur = conn.execute(
                    """INSERT OR IGNORE INTO papers
                       (file_path, title, authors, year, abstract, journal,
                        arxiv_id, text_content, size_bytes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        info["file_path"], info["title"],
                        json.dumps(info["authors"], ensure_ascii=False),
                        info["year"], info["abstract"], info["journal"],
                        info["arxiv_id"], info["text_content"], info["size_bytes"],
                    ),
                )
                if cur.rowcount > 0:
                    result.added += 1
                else:  # 文件路径唯一约束冲突：已入库
                    result.skipped += 1
            except sqlite3.Error as exc:
                result.failed += 1
                result.errors.append(f"{pdf}: 写入数据库失败: {exc}")
        conn.commit()
    finally:
        conn.close()
    return result


# ---------------------------------------------------------------------------
# 查询与搜索
# ---------------------------------------------------------------------------

def _escape_like(query: str) -> str:
    """转义 LIKE 通配符，让用户输入按字面匹配。"""
    return query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_library(
    query: str,
    db_path: str | Path | None = None,
    limit: int = 50,
) -> list[LibraryPaper]:
    """库内全文搜索：匹配标题 / 作者 / 摘要 / 全文，按年份倒序。"""
    query = (query or "").strip()
    conn = _connect(db_path)
    try:
        if not query:
            rows = conn.execute(
                "SELECT * FROM papers ORDER BY added_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            like = f"%{_escape_like(query)}%"
            rows = conn.execute(
                """SELECT * FROM papers
                   WHERE title LIKE ? ESCAPE '\\'
                      OR authors LIKE ? ESCAPE '\\'
                      OR abstract LIKE ? ESCAPE '\\'
                      OR text_content LIKE ? ESCAPE '\\'
                   ORDER BY year DESC, id DESC LIMIT ?""",
                (like, like, like, like, limit),
            ).fetchall()
        return [LibraryPaper.from_row(r) for r in rows]
    finally:
        conn.close()


def list_papers(
    db_path: str | Path | None = None,
    limit: int = 100,
) -> list[LibraryPaper]:
    """列出文献库全部论文（按入库时间倒序）。"""
    return search_library("", db_path=db_path, limit=limit)


def get_paper(
    paper_id: int,
    db_path: str | Path | None = None,
) -> LibraryPaper | None:
    """按 id 获取单篇论文。"""
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        return LibraryPaper.from_row(row) if row else None
    finally:
        conn.close()


def remove_paper(
    paper_id: int,
    db_path: str | Path | None = None,
) -> bool:
    """从文献库删除记录（不删除磁盘上的 PDF 文件）。

    同时级联删除该论文的向量分块（chunks 表），避免残留孤儿数据。
    """
    conn = _connect(db_path)
    try:
        cur = conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        if cur.rowcount > 0:
            try:
                # SQLite 外键默认不启用，手动级联删除向量分块（表可能不存在）
                conn.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
            except sqlite3.OperationalError:
                pass  # 从未建立过向量索引，无 chunks 表
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def library_stats(db_path: str | Path | None = None) -> dict:
    """文献库统计信息。"""
    conn = _connect(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        with_year = conn.execute("SELECT COUNT(*) FROM papers WHERE year IS NOT NULL").fetchone()[0]
        with_arxiv = conn.execute("SELECT COUNT(*) FROM papers WHERE arxiv_id IS NOT NULL").fetchone()[0]
        return {
            "total": total,
            "with_year": with_year,
            "with_arxiv": with_arxiv,
            "db_path": str(Path(db_path) if db_path else default_db_path()),
        }
    finally:
        conn.close()
