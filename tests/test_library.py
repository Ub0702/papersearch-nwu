"""本地文献库（M4）单元测试。

用 PyMuPDF 现场构造测试 PDF，覆盖：
- 扫描导入（含损坏 PDF 容错）
- 元数据提取（内嵌元数据优先 / 首页启发式兜底）
- 全文搜索（标题 / 正文 / 无命中 / LIKE 通配符转义）
- 重复扫描跳过、删除记录、非法路径
"""

import pytest

pytest.importorskip("fitz")  # 未装 PyMuPDF 则跳过本模块
import fitz

from papersearch.library import (
    LibraryError,
    get_paper,
    list_papers,
    remove_paper,
    scan_directory,
    search_library,
)


def _make_pdf(path, lines, title=None, author=None, creation=None):
    """用 PyMuPDF 生成一个简单的文本 PDF（insert_textbox 自动换行，避免长行截断）。"""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(72, 72, 520, 720), "\n".join(lines), fontsize=11)
    meta = {}
    if title:
        meta["title"] = title
    if author:
        meta["author"] = author
    if creation:
        meta["creationDate"] = creation
    if meta:
        doc.set_metadata(meta)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def pdf_dir(tmp_path):
    d = tmp_path / "papers"
    d.mkdir()
    # A：内嵌元数据齐全 + arXiv 标识
    _make_pdf(
        d / "a_survey.pdf",
        lines=[
            "Graph Neural Networks: A Survey",
            "Alice Smith, Bob Johnson",
            "Department of Computer Science, Some University",
            "arXiv:2301.06243",
            "Abstract",
            "This paper surveys graph neural networks and attention mechanisms.",
            "Experiments show 90% accuracy.",
        ],
        title="Graph Neural Networks: A Survey",
        author="Alice Smith; Bob Johnson",
        creation="D:20230101000000",
    )
    # B：无任何元数据，靠首页启发式提取
    _make_pdf(
        d / "b_medical.pdf",
        lines=[
            "Deep Learning for Medical Image Segmentation",
            "Carol White, David Brown",
            "Abstract This paper presents a novel deep learning method for "
            "medical image segmentation, published in 2022.",
        ],
    )
    # C：普通文本论文（全文搜索目标）
    _make_pdf(
        d / "c_transformers.pdf",
        lines=[
            "Transformers for Natural Language Processing",
            "Eve Green, Frank Hill",
            "Abstract",
            "Transformers are powerful. The model achieves 100% accuracy on the test set.",
        ],
    )
    return d


def test_scan_imports_all_pdfs(pdf_dir, tmp_path):
    db = tmp_path / "lib.db"
    result = scan_directory(pdf_dir, db_path=db)
    assert result.total == 3
    assert result.added == 3
    assert result.skipped == 0
    assert result.failed == 0
    assert len(list_papers(db_path=db)) == 3


def test_metadata_from_pdf_metadata(pdf_dir, tmp_path):
    db = tmp_path / "lib.db"
    scan_directory(pdf_dir, db_path=db)
    papers = {p.title: p for p in list_papers(db_path=db)}
    a = papers.get("Graph Neural Networks: A Survey")
    assert a is not None
    assert a.authors == ["Alice Smith", "Bob Johnson"]
    assert a.year == 2023
    assert a.arxiv_id == "2301.06243"


def test_metadata_heuristic_fallback(pdf_dir, tmp_path):
    db = tmp_path / "lib.db"
    scan_directory(pdf_dir, db_path=db)
    papers = {p.title: p for p in list_papers(db_path=db)}
    b = papers.get("Deep Learning for Medical Image Segmentation")
    assert b is not None
    assert "Carol White" in b.authors
    assert b.year == 2022


def test_search_hits_title_and_fulltext(pdf_dir, tmp_path):
    db = tmp_path / "lib.db"
    scan_directory(pdf_dir, db_path=db)
    by_title = search_library("medical image segmentation", db_path=db)
    assert len(by_title) == 1
    assert by_title[0].title.startswith("Deep Learning")
    by_text = search_library("attention mechanisms", db_path=db)
    assert len(by_text) == 1
    assert by_text[0].title.startswith("Graph Neural")


def test_search_no_hit(pdf_dir, tmp_path):
    db = tmp_path / "lib.db"
    scan_directory(pdf_dir, db_path=db)
    assert search_library("totally nonexistent phrase", db_path=db) == []


def test_search_literal_percent_escaped(pdf_dir, tmp_path):
    """% 必须按字面匹配，不能当 LIKE 通配符用。"""
    db = tmp_path / "lib.db"
    scan_directory(pdf_dir, db_path=db)
    assert len(search_library("100%", db_path=db)) == 1
    assert len(search_library("20%", db_path=db)) == 0


def test_rescan_skips_existing(pdf_dir, tmp_path):
    db = tmp_path / "lib.db"
    first = scan_directory(pdf_dir, db_path=db)
    second = scan_directory(pdf_dir, db_path=db)
    assert first.added == 3
    assert second.added == 0
    assert second.skipped == 3


def test_scan_nonexistent_dir(tmp_path):
    with pytest.raises(LibraryError):
        scan_directory(tmp_path / "nope", db_path=tmp_path / "lib.db")


def test_scan_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = scan_directory(empty, db_path=tmp_path / "lib.db")
    assert result.total == 0
    assert result.added == 0


def test_scan_handles_corrupt_pdf(pdf_dir, tmp_path):
    (pdf_dir / "broken.pdf").write_bytes(b"this is not a real pdf")
    db = tmp_path / "lib.db"
    result = scan_directory(pdf_dir, db_path=db)
    assert result.total == 4
    assert result.added == 3
    assert result.failed == 1
    assert len(result.errors) == 1


def test_remove_paper(pdf_dir, tmp_path):
    db = tmp_path / "lib.db"
    scan_directory(pdf_dir, db_path=db)
    papers = list_papers(db_path=db)
    pid = papers[0].id
    assert remove_paper(pid, db_path=db) is True
    assert get_paper(pid, db_path=db) is None
    assert remove_paper(99999, db_path=db) is False
    assert len(list_papers(db_path=db)) == 2
