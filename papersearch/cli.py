"""PaperSearch 命令行入口。

用法示例：
    python -m papersearch "graph neural network"            # 检索 Top 论文（列表）
    python -m papersearch "medical image segmentation" -t 5 # 检索并翻译摘要
    python -m papersearch "diffusion model" --engine ollama --model qwen2.5:7b
    python -m papersearch "llm" --engine openai --api-key sk-xxx
    python -m papersearch pdf paper.pdf                     # 整篇 PDF 翻译（保留排版）
    python -m papersearch pdf paper.pdf -o out --pages 1-3  # 只翻译前 3 页

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


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "pdf":
        return pdf_main(argv[1:])
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
