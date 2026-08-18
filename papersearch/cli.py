"""PaperSearch 命令行入口。

用法示例：
    python -m papersearch "graph neural network"            # 检索 Top 论文（列表）
    python -m papersearch "medical image segmentation" -t 5 # 检索并翻译摘要
    python -m papersearch "diffusion model" --engine ollama --model qwen2.5:7b
    python -m papersearch "llm" --engine openai --api-key sk-xxx

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


def main(argv: list[str] | None = None) -> int:
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
