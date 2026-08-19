"""M5 语义检索 + RAG 单测。

用 FakeEmbedder（字符袋向量）和 FakeLLM（固定回答）隔离外部依赖，
覆盖：分块、向量索引、增量跳过、语义检索排序、RAG 回答、引用来源、
未索引报错、级联删除。
"""

from __future__ import annotations

import numpy as np
import pytest

from papersearch.rag import (
    RagError,
    ask,
    chunk_text,
    index_papers,
    semantic_search,
)
from papersearch.library import remove_paper, scan_directory

try:
    import pymupdf as fitz
except ImportError:
    import fitz


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """确定性 embedding：字符袋向量（同词文本相似度高，便于断言排序）。"""

    name = "fake"

    def __init__(self, model="fake"):
        self.model = model

    def available(self) -> bool:
        return True

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = np.zeros(128, dtype=float)
            for ch in (t or "").lower():
                v[ord(ch) % 128] += 1.0
            norm = np.linalg.norm(v)
            out.append((v / norm if norm > 0 else v).tolist())
        return out


class FakeLLM:
    """固定回答的假 LLM（记录收到的 prompt 供断言）。"""

    name = "fake"

    def __init__(self):
        self.last_system = ""
        self.last_user = ""

    def available(self) -> bool:
        return True

    def chat(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return "根据参考片段，图神经网络在推荐系统中主要用于学习用户和物品的表示。"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _make_pdf(path, lines, title=None, author=None):
    """用 PyMuPDF 生成一个文本 PDF（insert_textbox 自动换行）。"""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(72, 72, 520, 720), "\n".join(lines), fontsize=11)
    meta = {}
    if title:
        meta["title"] = title
    if author:
        meta["author"] = author
    if meta:
        doc.set_metadata(meta)
    doc.save(str(path))
    doc.close()


def _make_library(tmp_path, papers) -> str:
    """构造一个文献库并返回 db 路径。papers: [(文件名, 行列表, 标题)]"""
    src = tmp_path / "src"
    src.mkdir()
    for fname, lines, title in papers:
        _make_pdf(src / fname, lines, title=title)
    db = tmp_path / "lib.db"
    result = scan_directory(src, db_path=db)
    assert result.added == len(papers)
    return str(db)


# ---------------------------------------------------------------------------
# 分块
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("short text")
        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_long_text_split_with_overlap(self):
        para = "word " * 200  # ~1000 字符 > 800
        chunks = chunk_text(para, chunk_size=800, overlap=150)
        assert len(chunks) >= 2
        # 相邻块应有重叠（overlap 保留跨块上下文）
        assert chunks[1].startswith("word " * 37)  # 800 - 150 = 650 字符 ≈ 37 个 word

    def test_paragraph_boundary(self):
        text = ("a" * 100 + "\n\n" + "b" * 100 + "\n\n" + "c" * 100)
        chunks = chunk_text(text, chunk_size=250, overlap=0)
        # 三个短段落应合并进一块（总长 300 > 250？—— 300>250，会切两块）
        assert 1 <= len(chunks) <= 2


# ---------------------------------------------------------------------------
# 索引
# ---------------------------------------------------------------------------

class TestIndex:
    def test_index_then_incremental_skip(self, tmp_path):
        db = _make_library(tmp_path, [
            ("gnn.pdf", ["Graph Neural Networks Survey", "We survey GNN methods."], "GNN Survey"),
        ])
        embedder = FakeEmbedder()
        r1 = index_papers(db_path=db, embedder=embedder)
        assert r1.indexed == 1
        assert r1.chunks >= 1
        # 再次索引：全部跳过
        r2 = index_papers(db_path=db, embedder=embedder)
        assert r2.indexed == 0
        assert r2.chunks == 0

    def test_reindex(self, tmp_path):
        db = _make_library(tmp_path, [
            ("gnn.pdf", ["Graph Neural Networks Survey", "We survey GNN methods."], "GNN Survey"),
        ])
        embedder = FakeEmbedder()
        index_papers(db_path=db, embedder=embedder)
        r = index_papers(db_path=db, embedder=embedder, reindex=True)
        assert r.indexed == 1  # 清空后重建

    def test_remove_paper_cascades_chunks(self, tmp_path):
        db = _make_library(tmp_path, [
            ("a.pdf", ["Alpha content about graph neural networks."], "Alpha"),
            ("b.pdf", ["Beta content about medical imaging."], "Beta"),
        ])
        index_papers(db_path=db, embedder=FakeEmbedder())
        assert remove_paper(1, db_path=db)
        # 删除论文 1 后，剩论文 2 的分块
        hits = semantic_search("medical imaging", db_path=db, embedder=FakeEmbedder(), top_k=5)
        assert len(hits) == 1
        assert hits[0].paper.id == 2

    def test_unavailable_embedder_raises(self, tmp_path):
        class Offline(FakeEmbedder):
            def available(self):
                return False

        db = _make_library(tmp_path, [("a.pdf", ["some text"], "A")])
        with pytest.raises(RagError):
            index_papers(db_path=db, embedder=Offline())


# ---------------------------------------------------------------------------
# 语义检索
# ---------------------------------------------------------------------------

class TestSemanticSearch:
    def test_relevant_chunk_ranked_first(self, tmp_path):
        db = _make_library(tmp_path, [
            ("gnn.pdf", [
                "Graph Neural Networks: A Survey",
                "We survey graph neural networks and their applications in recommender systems.",
            ], "GNN Survey"),
            ("medical.pdf", [
                "Deep Learning for Medical Image Segmentation",
                "This paper presents a novel segmentation method.",
            ], "Medical Segmentation"),
        ])
        index_papers(db_path=db, embedder=FakeEmbedder())
        hits = semantic_search("graph neural network", db_path=db, embedder=FakeEmbedder(), top_k=5)
        assert len(hits) >= 1
        assert hits[0].paper.title == "GNN Survey"  # 语义最相关排第一
        assert hits[0].score > 0

    def test_unindexed_raises(self, tmp_path):
        db = _make_library(tmp_path, [("a.pdf", ["some text"], "A")])
        with pytest.raises(RagError, match="index"):
            semantic_search("anything", db_path=db, embedder=FakeEmbedder())

    def test_empty_query(self, tmp_path):
        db = _make_library(tmp_path, [("a.pdf", ["some text"], "A")])
        index_papers(db_path=db, embedder=FakeEmbedder())
        assert semantic_search("", db_path=db, embedder=FakeEmbedder()) == []


# ---------------------------------------------------------------------------
# RAG 问答
# ---------------------------------------------------------------------------

class TestAsk:
    def test_answer_with_sources(self, tmp_path):
        db = _make_library(tmp_path, [
            ("gnn.pdf", [
                "Graph Neural Networks: A Survey",
                "We survey graph neural networks and their applications in recommender systems.",
            ], "GNN Survey"),
        ])
        index_papers(db_path=db, embedder=FakeEmbedder())
        llm = FakeLLM()
        answer = ask("图神经网络有什么应用？", db_path=db, embedder=FakeEmbedder(), llm=llm)
        assert "参考片段" in answer.answer or answer.answer
        assert len(answer.sources) >= 1
        assert answer.sources[0].paper.title == "GNN Survey"
        # LLM 应收到包含问题与片段的 prompt
        assert "图神经网络有什么应用" in llm.last_user
        assert "Graph Neural Networks" in llm.last_user

    def test_summary_format(self, tmp_path):
        db = _make_library(tmp_path, [
            ("gnn.pdf", ["Graph Neural Networks Survey text"], "GNN Survey"),
        ])
        index_papers(db_path=db, embedder=FakeEmbedder())
        answer = ask("GNN", db_path=db, embedder=FakeEmbedder(), llm=FakeLLM())
        s = answer.summary()
        assert "引用来源" in s
        assert "GNN Survey" in s

    def test_empty_question_raises(self, tmp_path):
        db = _make_library(tmp_path, [("a.pdf", ["text"], "A")])
        index_papers(db_path=db, embedder=FakeEmbedder())
        with pytest.raises(RagError, match="不能为空"):
            ask("  ", db_path=db, embedder=FakeEmbedder(), llm=FakeLLM())

    def test_unindexed_raises(self, tmp_path):
        db = _make_library(tmp_path, [("a.pdf", ["text"], "A")])
        with pytest.raises(RagError):
            ask("question", db_path=db, embedder=FakeEmbedder(), llm=FakeLLM())
