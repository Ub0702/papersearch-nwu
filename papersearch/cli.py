"""PaperSearch 命令行入口。

用法示例：
    python -m papersearch "graph neural network"            # 检索 Top 论文（列表）
    python -m papersearch "medical image segmentation" -t 5 # 检索并翻译摘要
    python -m papersearch "diffusion model" --engine ollama --model qwen2.5:7b
    python -m papersearch "llm" --engine openai --api-key sk-xxx
    python -m papersearch pdf paper.pdf                     # 整篇 PDF 翻译（保留排版）
    python -m papersearch pdf paper.pdf -o out --pages 1-3  # 只翻译前 3 页
    python -m papersearch library scan ~/papers             # 扫描文件夹，导入本地 PDF 文献库
    python -m papersearch library search "graph neural"     # 库内全文搜索
    python -m papersearch library index                     # 建立向量索引（语义检索/RAG 前置步骤）
    python -m papersearch library search "什么是图神经网络" --semantic  # 语义搜索
    python -m papersearch library ask "这篇文章用了什么方法" # 基于论文内容问答（RAG）

环境变量：
    PAPERSEARCH_API_KEY    openai/deepl 引擎的 API Key（亦可 --api-key 传入）
    PAPERSEARCH_BASE_URL   自定义 OpenAI 兼容服务地址
"""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .glossary import Glossary
from .models import Paper
from .output import papers_to_markdown, write_output
from .search import SemanticScholarSource, ArxivSource, search_all
from .translate import get_translator, translate_with_glossary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="papersearch",
        description="学术论文检索 + 学术级翻译（关键词 -> 双语论文摘要）",
    )
    parser.add_argument("query", help="检索关键词，例如 'graph neural network'")
    parser.add_argument("-n", "--top", type=int, default=5, help="返回论文数量（默认 5）")
    parser.add_argument(
        "--sort", default="relevance", choices=["relevance", "date"],
        help="结果排序：relevance 按相关度（默认），date 按出版年份最新在前",
    )
    parser.add_argument(
        "-t", "--translate", action="store_true",
        help="翻译论文摘要并输出双语对照",
    )
    parser.add_argument(
        "--engine", default="ollama", choices=["ollama", "openai", "deepl"],
        help="翻译引擎（默认 ollama，本地免费）",
    )
    parser.add_argument("--model", help="引擎模型名（ollama: qwen2.5:7b；openai: gpt-4o-mini）")
    parser.add_argument("--api-key", default=os.getenv("PAPERSEARCH_API_KEY"), help="API Key（openai/deepl）")
    parser.add_argument("--base-url", default=os.getenv("PAPERSEARCH_BASE_URL"), help="OpenAI 兼容服务地址")
    parser.add_argument(
        "--glossary", default=None,
        help="自定义术语表 JSON 路径（默认使用内置 data/glossary.json）",
    )
    parser.add_argument("--out-dir", default="output", help="输出目录（默认 output/）")
    parser.add_argument("--json", action="store_true", help="同时输出机器可读的 JSON 结果")
    parser.add_argument("--version", action="version", version=f"PaperSearch {__version__}")
    return parser


def _print_paper(idx: int, paper: Paper) -> None:
    print(f"\n[{idx}] {paper.title}")
    print(f"    {paper.authors_text} | {paper.year_text} | {paper.source} | relevance={paper.relevance:.2f}")
    print(f"    {paper.url}")
    if paper.pdf_url:
        print(f"    PDF: {paper.pdf_url}")
    abstract = paper.abstract[:220] + ("..." if len(paper.abstract) > 220 else "")
    print(f"    Abstract: {abstract}")


def build_pdf_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="papersearch pdf",
        description="整篇 PDF 翻译（保留排版与公式，基于 pdf2zh/PDFMathTranslate）",
    )
    parser.add_argument("pdf", help="输入 PDF 文件路径")
    parser.add_argument("-o", "--out-dir", default=None, help="输出目录（默认与输入 PDF 同目录）")
    parser.add_argument(
        "--engine", default="ollama",
        help="翻译后端（默认 ollama，本地免费；也可 openai/deepl/google）",
    )
    parser.add_argument("--model", default="qwen2.5:7b", help="模型名（ollama: qwen2.5:7b；openai: gpt-4o-mini）")
    parser.add_argument("--lang-in", default="en", help="源语言代码（默认 en）")
    parser.add_argument("--lang-out", default="zh", help="目标语言代码（默认 zh）")
    parser.add_argument("--pages", default=None, help="只翻译指定页，如 1、1-3、1,3（默认全部）")
    parser.add_argument("--thread", type=int, default=4, help="并行翻译线程数（默认 4）")
    parser.add_argument("--timeout", type=int, default=3600, help="超时秒数（默认 3600，整篇 PDF 较慢）")
    return parser


def pdf_main(argv: list[str] | None = None) -> int:
    from .pdf_translate import PdfTranslateError, translate_pdf

    args = build_pdf_parser().parse_args(argv)
    if not os.path.exists(args.pdf):
        print(f"[error] 输入文件不存在: {args.pdf}")
        return 1
    print(
        f"[info] 翻译 PDF: {args.pdf}\n"
        f"[info] 引擎: {args.engine}:{args.model} | {args.lang_in} -> {args.lang_out}"
        + (f" | 页: {args.pages}" if args.pages else "")
    )
    print("[info] 首次运行会下载版面分析模型，之后将缓存（国内网络建议开启代理）...")
    try:
        result = translate_pdf(
            args.pdf,
            args.out_dir,
            engine=args.engine,
            model=args.model,
            lang_in=args.lang_in,
            lang_out=args.lang_out,
            pages=args.pages,
            thread=args.thread,
            timeout=args.timeout,
        )
    except PdfTranslateError as exc:
        print(f"[error] {exc}")
        return 1
    print(f"[info] 翻译完成 ✅")
    print(result.summary())
    return 0


def build_library_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="papersearch library",
        description="本地文献库管理（扫描导入 PDF + 元数据提取 + 全文搜索 + 语义检索 + RAG 问答）",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="命令")

    scan = sub.add_parser("scan", help="扫描文件夹，导入其中的 PDF 论文")
    scan.add_argument("directory", help="要扫描的文件夹路径")
    scan.add_argument("--no-recursive", action="store_true", help="只扫描顶层目录（默认递归子目录）")
    scan.add_argument("--db", default=None, help="数据库路径（默认 ~/.papersearch/library.db）")

    lst = sub.add_parser("list", help="列出文献库全部论文（按入库时间倒序）")
    lst.add_argument("--limit", type=int, default=100, help="最大条数（默认 100）")
    lst.add_argument("--db", default=None, help="数据库路径（默认 ~/.papersearch/library.db）")

    search = sub.add_parser("search", help="库内搜索：默认全文搜索，--semantic 切换语义搜索")
    search.add_argument("query", help="搜索关键词或问题")
    search.add_argument("--limit", type=int, default=50, help="最大条数（默认 50，仅全文搜索有效）")
    search.add_argument("--semantic", action="store_true", help="语义搜索（按语义相关度，需先 library index）")
    search.add_argument("--top-k", type=int, default=5, help="语义搜索返回的分块数（默认 5）")
    search.add_argument("--embed-model", default=None, help="embedding 模型（默认 bge-m3）")
    search.add_argument("--db", default=None, help="数据库路径（默认 ~/.papersearch/library.db）")

    index = sub.add_parser("index", help="为文献库建立向量索引（语义检索与 RAG 的前置步骤）")
    index.add_argument("--reindex", action="store_true", help="清空旧索引后重建（换 embedding 模型后需要）")
    index.add_argument("--embed-model", default=None, help="embedding 模型（默认 bge-m3）")
    index.add_argument("--db", default=None, help="数据库路径（默认 ~/.papersearch/library.db）")

    ask = sub.add_parser("ask", help="基于文献库内容问答（RAG，需先 library index）")
    ask.add_argument("question", help="要问的问题")
    ask.add_argument("--top-k", type=int, default=5, help="参考片段数（默认 5）")
    ask.add_argument("--embed-model", default=None, help="embedding 模型（默认 bge-m3）")
    ask.add_argument("--engine", default="ollama", choices=["ollama", "openai", "deepl"], help="回答引擎（默认 ollama）")
    ask.add_argument("--model", default=None, help="回答模型（ollama: qwen2.5:7b；openai: gpt-4o-mini）")
    ask.add_argument("--api-key", default=os.getenv("PAPERSEARCH_API_KEY"), help="API Key（openai/deepl）")
    ask.add_argument("--base-url", default=os.getenv("PAPERSEARCH_BASE_URL"), help="OpenAI 兼容服务地址")
    ask.add_argument("--db", default=None, help="数据库路径（默认 ~/.papersearch/library.db）")

    info = sub.add_parser("info", help="查看单篇论文详情")
    info.add_argument("id", type=int, help="论文 id（list 里第一列）")
    info.add_argument("--db", default=None, help="数据库路径（默认 ~/.papersearch/library.db）")

    remove = sub.add_parser("remove", help="从文献库删除记录（不删除磁盘文件）")
    remove.add_argument("id", type=int, help="论文 id")
    remove.add_argument("--db", default=None, help="数据库路径（默认 ~/.papersearch/library.db）")

    stats = sub.add_parser("stats", help="文献库统计信息")
    stats.add_argument("--db", default=None, help="数据库路径（默认 ~/.papersearch/library.db）")
    return parser


def _print_library_papers(papers) -> None:
    if not papers:
        print("（空）")
        return
    for p in papers:
        print(f"[{p.id}] {p.title}")
        print(f"    {p.authors_text} | {p.year_text} | {p.size_text}")
        print(f"    {p.file_path}")


def library_main(argv: list[str] | None = None) -> int:
    from .library import (
        LibraryError,
        get_paper,
        library_stats,
        list_papers,
        remove_paper,
        scan_directory,
        search_library,
    )

    args = build_library_parser().parse_args(argv)
    try:
        if args.command == "scan":
            result = scan_directory(
                args.directory, db_path=args.db, recursive=not args.no_recursive
            )
            print(f"[info] {result.summary()}")
            for err in result.errors:
                print(f"[warn] {err}")
        elif args.command == "list":
            papers = list_papers(db_path=args.db, limit=args.limit)
            print(f"[info] 共 {len(papers)} 篇：")
            _print_library_papers(papers)
        elif args.command == "search":
            if getattr(args, "semantic", False):
                return _library_semantic_search(args)
            papers = search_library(args.query, db_path=args.db, limit=args.limit)
            print(f"[info] 搜索「{args.query}」命中 {len(papers)} 篇：")
            _print_library_papers(papers)
        elif args.command == "index":
            return _library_index(args)
        elif args.command == "ask":
            return _library_ask(args)
        elif args.command == "info":
            paper = get_paper(args.id, db_path=args.db)
            if not paper:
                print(f"[error] 未找到 id={args.id} 的论文")
                return 1
            print(f"标题: {paper.title}")
            print(f"作者: {paper.authors_text}")
            print(f"年份: {paper.year_text}")
            print(f"arXiv: {paper.arxiv_id or 'N/A'}")
            print(f"文件: {paper.file_path} ({paper.size_text})")
            print(f"入库: {paper.added_at}")
            if paper.abstract:
                print(f"摘要: {paper.abstract[:300]}")
        elif args.command == "remove":
            if remove_paper(args.id, db_path=args.db):
                print(f"[info] 已从文献库删除 id={args.id}（磁盘文件未动）")
            else:
                print(f"[error] 未找到 id={args.id} 的论文")
                return 1
        elif args.command == "stats":
            s = library_stats(db_path=args.db)
            print(f"[info] 数据库: {s['db_path']}")
            print(f"[info] 共 {s['total']} 篇 | 有年份 {s['with_year']} | 含 arXiv ID {s['with_arxiv']}")
        return 0
    except LibraryError as exc:
        print(f"[error] {exc}")
        return 1


def _library_index(args) -> int:
    """papersearch library index：建立向量索引。"""
    from .embeddings import OllamaEmbedder
    from .rag import RagError, index_papers

    embedder = OllamaEmbedder(model=args.embed_model) if args.embed_model else OllamaEmbedder()
    print(
        f"[info] 建立向量索引 | embedding 模型: {embedder.model}"
        + (" | 重建模式" if args.reindex else "")
    )
    if not embedder.available():
        print(
            f"[error] embedding 模型 {embedder.model} 不可用：请确认 Ollama 已启动，"
            f"并执行 ollama pull {embedder.model}"
        )
        return 1
    result = index_papers(db_path=args.db, embedder=embedder, reindex=args.reindex)
    print(f"[info] {result.summary()}")
    for err in result.failed:
        print(f"[warn] {err}")
    return 0


def _library_semantic_search(args) -> int:
    """papersearch library search --semantic：语义搜索。"""
    from .embeddings import OllamaEmbedder
    from .rag import RagError, semantic_search

    embedder = OllamaEmbedder(model=args.embed_model) if args.embed_model else OllamaEmbedder()
    try:
        hits = semantic_search(
            args.query, db_path=args.db, embedder=embedder, top_k=args.top_k
        )
    except RagError as exc:
        print(f"[error] {exc}")
        return 1
    if not hits:
        print("（无命中）")
        return 0
    print(f"[info] 语义搜索「{args.query}」Top {len(hits)} 命中（相关度从高到低）：")
    for i, hit in enumerate(hits, 1):
        print(f"[{i}] {hit.paper.title}（相似度 {hit.score:.3f}）")
        print(f"    {hit.paper.authors_text} | {hit.paper.year_text}")
        print(f"    片段: {hit.snippet}")
    return 0


def _library_ask(args) -> int:
    """papersearch library ask：RAG 问答。"""
    from .embeddings import OllamaEmbedder
    from .rag import RagError, ask
    from .translate import DEFAULT_OLLAMA_MODEL, get_translator

    embedder = OllamaEmbedder(model=args.embed_model) if args.embed_model else OllamaEmbedder()
    try:
        llm = get_translator(
            engine=args.engine, api_key=args.api_key,
            model=args.model, base_url=args.base_url,
        )
    except ValueError as exc:
        print(f"[error] {exc}")
        return 1
    if not llm.available():
        print(f"[error] 回答引擎 {args.engine} 当前不可用（Ollama 未启动？模型未安装？）")
        return 1
    print(f"[info] RAG 问答 | embedding: {embedder.model} | 回答: {args.engine}"
          + (f":{args.model}" if args.model else f":{DEFAULT_OLLAMA_MODEL}"))
    try:
        answer = ask(args.question, db_path=args.db, embedder=embedder, llm=llm, top_k=args.top_k)
    except RagError as exc:
        print(f"[error] {exc}")
        return 1
    print()
    print(answer.summary())
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "pdf":
        return pdf_main(argv[1:])
    if argv and argv[0] == "library":
        return library_main(argv[1:])
    args = build_parser().parse_args(argv)
    glossary = Glossary.load(args.glossary)
    print(f"[info] PaperSearch v{__version__} | 关键词: {args.query} | 术语表: {glossary.size} 条")
    sort_desc = "按年份最新在前" if args.sort == "date" else "按相关度"
    print(f"[info] 检索中（OpenAlex + Semantic Scholar + arXiv），排序: {sort_desc}...")
    papers = search_all(args.query, limit_per_source=max(args.top * 2, 10), sort=args.sort)
    papers = papers[: args.top]
    if not papers:
        print("[error] 未检索到任何论文，请更换关键词或稍后重试")
        return 1

    print(f"[info] 共找到 {len(papers)} 篇，Top {args.top}：")
    for i, p in enumerate(papers, 1):
        _print_paper(i, p)

    translated: dict[str, str] = {}
    if args.translate:
        try:
            translator = get_translator(
                engine=args.engine, api_key=args.api_key,
                model=args.model, base_url=args.base_url,
            )
        except ValueError as exc:
            print(f"[error] {exc}")
            return 1
        if not translator.available():
            print(
                f"[warn] 翻译引擎 {args.engine} 当前不可用（Ollama 未启动？模型未安装？）。\n"
                f"       请先运行: ollama serve 以及 ollama pull {args.model or 'qwen2.5:7b'}\n"
                f"       或改用: --engine openai --api-key <你的key>"
            )
            if args.engine == "ollama":
                print("       （本次仅输出检索结果，翻译已跳过）")
                papers_to_markdown(papers)
                _write_outputs(papers, translated, args)
                return 0
        print(f"[info] 使用引擎 {args.engine} 翻译摘要（{len(papers)} 篇）...")
        for i, p in enumerate(papers, 1):
            print(f"  [{i}/{len(papers)}] 翻译中: {p.title[:60]}...")
            translated[p.title] = translate_with_glossary(translator, p.abstract, glossary)

    _write_outputs(papers, translated, args)
    if args.translate:
        print(f"\n[info] 双语对照输出完成。")
    return 0


def _write_outputs(papers, translated, args) -> None:
    md = papers_to_markdown(papers, translated)
    path = write_output(md, args.out_dir, name=args.query, as_html=True)
    print(f"[info] 已输出: {path}")
    if args.json:
        import json
        json_path = os.path.join(args.out_dir, "papers.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([p.to_dict() for p in papers], f, ensure_ascii=False, indent=2)
        print(f"[info] 已输出: {json_path}")


if __name__ == "__main__":
    sys.exit(main())
