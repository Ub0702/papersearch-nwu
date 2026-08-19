"""语义检索 + RAG 问答（M5）。

整条链路：
  论文全文 -> 分块 -> embedding -> SQLite 存向量（chunks 表）
  用户提问 -> embedding -> 余弦相似度 Top-K -> 拼 prompt -> LLM 回答 + 引用来源

复用现有模块：
- embeddings.OllamaEmbedder：本地向量化（bge-m3）
- translate.get_translator：对话引擎（ollama/openai，DeepL 不支持）
- library：SQLite 文献库（papers 表 + 元数据）
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .embeddings import EmbeddingError, OllamaEmbedder, cosine_top_k
from .library import LibraryPaper, _connect

#: 分块参数：目标块大小（字符）与重叠（保证跨块语义不丢）
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

#: RAG 系统提示：要求基于给定上下文回答、中文输出、引用标注
RAG_SYSTEM_PROMPT = (
    "你是一个严谨的学术助手。请基于【参考片段】中的论文内容回答用户问题。\n"
    "要求：\n"
    "1. 用简体中文回答；\n"
    "2. 只依据参考片段中的内容，不要编造参考片段里没有的信息；\n"
    "3. 回答中涉及引用时，用 [1][2] 标注对应片段编号；\n"
    "4. 如果参考片段不足以回答问题，明确说明\"参考内容不足以回答该问题\"。"
)


class RagError(RuntimeError):
    """RAG 可预期错误（未索引 / 引擎不支持 / 请求失败）。"""


# ---------------------------------------------------------------------------
# 文本分块
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """把长文本切成有重叠的分块。

    策略：优先在段落边界切；单个段落超长时按固定窗口硬切。
    overlap 用于保留跨块上下文，避免语义被拦腰截断。
    """
    text = (text or "").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            # 段落超长：先落盘已有缓冲，再按窗口硬切
            if buf:
                chunks.append(buf)
                buf = ""
            for start in range(0, len(para), chunk_size - overlap):
                piece = para[start : start + chunk_size]
                if piece:
                    chunks.append(piece)
            continue
        if len(buf) + len(para) + 2 <= chunk_size:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    return chunks


# ---------------------------------------------------------------------------
# 向量索引（SQLite chunks 表）
# ---------------------------------------------------------------------------

_CHUNK_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id  INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    chunk_idx INTEGER NOT NULL,
    text      TEXT    NOT NULL,
    embedding TEXT    NOT NULL,          -- JSON 数组（向量）
    UNIQUE(paper_id, chunk_idx)
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
"""


def _connect_indexed(db_path) -> sqlite3.Connection:
    """打开文献库连接并确保 chunks 表存在。"""
    conn = _connect(db_path)
    conn.executescript(_CHUNK_SCHEMA)
    return conn


@dataclass
class IndexResult:
    """一次索引任务的统计。"""

    total: int = 0          # 已入库论文总数
    indexed: int = 0        # 本次新增索引的论文数
    chunks: int = 0         # 本次写入的分块总数
    skipped: int = 0        # 已有索引的论文数
    failed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"论文 {self.total} 篇：本次索引 {self.indexed} 篇（{self.chunks} 个分块），"
            f"已有索引跳过 {self.skipped}，失败 {len(self.failed)}"
        )


def index_papers(
    db_path: str | Path | None = None,
    embedder: OllamaEmbedder | None = None,
    reindex: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> IndexResult:
    """为文献库中尚未索引的论文建立向量索引（增量）。

    reindex=True 时清空全部 chunks 重建（用于更换 embedding 模型后）。
    """
    embedder = embedder or OllamaEmbedder()
    if not embedder.available():
        raise RagError(
            f"embedding 模型不可用：请确认 Ollama 已启动且已执行 ollama pull {embedder.model}"
        )

    conn = _connect_indexed(db_path)
    result = IndexResult()
    try:
        if reindex:
            conn.execute("DELETE FROM chunks")
            conn.commit()

        rows = conn.execute(
            """SELECT p.id, p.title, p.text_content
               FROM papers p
               LEFT JOIN (SELECT paper_id, COUNT(*) AS cnt FROM chunks GROUP BY paper_id) c
                 ON p.id = c.paper_id
               WHERE p.text_content IS NOT NULL AND p.text_content != ''
                 AND (c.cnt IS NULL OR ?)
               ORDER BY p.id""",
            (1 if reindex else 0,),
        ).fetchall()
        result.total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

        for i, row in enumerate(rows, 1):
            if progress:
                progress(i, len(rows), str(row["title"]))
            paper_id, title, text = row["id"], row["title"], row["text_content"]
            chunks = chunk_text(text)
            if not chunks:
                result.skipped += 1
                continue
            try:
                vectors = embedder.embed(chunks)
            except EmbeddingError as exc:
                result.failed.append(f"{title}: {exc}")
                continue
            if len(vectors) != len(chunks):
                result.failed.append(f"{title}: 向量数量与分块数量不一致")
                continue
            cur = conn.executemany(
                """INSERT OR IGNORE INTO chunks (paper_id, chunk_idx, text, embedding)
                   VALUES (?, ?, ?, ?)""",
                [
                    (paper_id, idx, text, json.dumps(vec, ensure_ascii=False))
                    for idx, (text, vec) in enumerate(zip(chunks, vectors))
                ],
            )
            result.chunks += cur.rowcount
            result.indexed += 1
        conn.commit()
    finally:
        conn.close()
    return result


def _load_chunk_matrix(conn: sqlite3.Connection) -> tuple[np.ndarray, list[int]]:
    """加载全部 chunk 向量为矩阵。返回 (矩阵, chunk_id 列表)。"""
    rows = conn.execute("SELECT id, embedding FROM chunks").fetchall()
    if not rows:
        return np.empty((0, 0), dtype=np.float32), []
    ids = [r["id"] for r in rows]
    matrix = np.array(
        [json.loads(r["embedding"]) for r in rows], dtype=np.float32
    )
    return matrix, ids


# ---------------------------------------------------------------------------
# 语义检索
# ---------------------------------------------------------------------------

@dataclass
class ChunkHit:
    """一条语义检索命中。"""

    paper: LibraryPaper
    chunk_idx: int
    text: str
    score: float

    @property
    def snippet(self) -> str:
        """用于展示的文本片段（截断到 300 字符）。"""
        s = " ".join(self.text.split())
        return s[:300] + ("..." if len(s) > 300 else "")


def semantic_search(
    query: str,
    db_path: str | Path | None = None,
    embedder: OllamaEmbedder | None = None,
    top_k: int = 5,
) -> list[ChunkHit]:
    """语义搜索：问题 -> 向量 -> 与文献库全部分块算相似度 -> Top-K。"""
    query = (query or "").strip()
    if not query:
        return []
    embedder = embedder or OllamaEmbedder()
    conn = _connect_indexed(db_path)
    try:
        matrix, ids = _load_chunk_matrix(conn)
        if not ids:
            raise RagError(
                "文献库还没有向量索引，请先运行: papersearch library index"
            )
        qvec = embedder.embed([query])[0]
        hits = cosine_top_k(qvec, matrix, top_k)
        out: list[ChunkHit] = []
        for row_idx, score in hits:
            # cosine_top_k 返回矩阵行索引，需映射回 chunks 表 id
            chunk_id = ids[row_idx]
            row = conn.execute(
                """SELECT c.*, p.* FROM chunks c
                   JOIN papers p ON p.id = c.paper_id
                   WHERE c.id = ?""",
                (chunk_id,),
            ).fetchone()
            if row is None:
                continue
            paper = LibraryPaper.from_row(row)
            out.append(ChunkHit(paper=paper, chunk_idx=row["chunk_idx"],
                                text=row["text"], score=score))
        return out
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# RAG 问答
# ---------------------------------------------------------------------------

@dataclass
class Source:
    """RAG 回答的一条引用来源。"""

    paper: LibraryPaper
    chunk_idx: int
    text: str
    score: float

    @property
    def citation(self) -> str:
        """格式化引用：作者(年份) 标题。"""
        return f"{self.paper.authors_text} ({self.paper.year_text}) {self.paper.title}"


@dataclass
class RagAnswer:
    """RAG 问答结果。"""

    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"问: {self.question}", "", self.answer, ""]
        if self.sources:
            lines.append("引用来源:")
            for i, src in enumerate(self.sources, 1):
                lines.append(f"[{i}] {src.citation}（相似度 {src.score:.3f}）")
        return "\n".join(lines)


def ask(
    question: str,
    db_path: str | Path | None = None,
    embedder: OllamaEmbedder | None = None,
    llm=None,
    top_k: int = 5,
) -> RagAnswer:
    """RAG 问答：检索相关片段 -> 组装 prompt -> LLM 回答 -> 附引用来源。

    llm: translate.Translator 实例（需支持 chat()）。默认用 Ollama qwen2.5:7b。
    """
    question = (question or "").strip()
    if not question:
        raise RagError("问题不能为空")
    hits = semantic_search(question, db_path=db_path, embedder=embedder, top_k=top_k)
    if not hits:
        raise RagError(
            "文献库还没有向量索引，请先运行: papersearch library index"
        )

    if llm is None:
        from .translate import DEFAULT_OLLAMA_MODEL, OllamaTranslator

        llm = OllamaTranslator(model=DEFAULT_OLLAMA_MODEL)

    # 组装参考上下文（限制总量，避免超长）
    context_blocks = []
    for i, hit in enumerate(hits, 1):
        snippet = " ".join(hit.text.split())
        context_blocks.append(f"[{i}] {snippet[:1200]}")
    context = "\n\n".join(context_blocks)
    user_prompt = (
        f"【参考片段】\n{context}\n\n"
        f"【问题】{question}\n\n"
        "请基于参考片段回答，并标注引用编号 [1][2]..."
    )

    try:
        answer = llm.chat(RAG_SYSTEM_PROMPT, user_prompt)
    except NotImplementedError as exc:
        raise RagError(str(exc)) from exc

    return RagAnswer(
        question=question,
        answer=answer,
        sources=[Source(paper=h.paper, chunk_idx=h.chunk_idx,
                        text=h.text, score=h.score) for h in hits],
    )
