"""PaperSearch Web UI（Streamlit）。

运行：
    streamlit run app.py

功能：
- 论文检索：输入关键词 -> 聚合检索 -> 逐篇查看中英双语摘要，可导出 Markdown。
- PDF 翻译：上传 PDF -> 整篇翻译（保留排版与公式，需本地安装 pdf2zh）。
- 本地文献库：扫描本地 PDF 建立 SQLite 文献库，元数据提取 + 全文搜索 + 语义搜索。
- 论文问答：基于文献库内容 RAG 问答（需要先建立向量索引，Ollama 拉取 bge-m3）。
"""

from __future__ import annotations

import os

import streamlit as st

from papersearch.glossary import Glossary
from papersearch.models import Paper
from papersearch.search import search_all
from papersearch.translate import get_translator, translate_with_glossary

st.set_page_config(page_title="PaperSearch", page_icon="📄", layout="wide")

# ----------------------------------------------------------------------
# 状态
# ----------------------------------------------------------------------
if "papers" not in st.session_state:
    st.session_state.papers = []
if "translated" not in st.session_state:
    st.session_state.translated = {}
if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# ----------------------------------------------------------------------
# 侧边栏：翻译引擎设置
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## PaperSearch")
    st.caption("学术论文检索 + 学术级翻译")
    st.divider()
    st.markdown("### 翻译引擎")
    # 云端部署支持：Streamlit Secrets 注入的 PAPERSEARCH_API_KEY / BASE_URL 会自动生效
    default_api_key = os.environ.get("PAPERSEARCH_API_KEY", "")
    default_base_url = os.environ.get("PAPERSEARCH_BASE_URL", "https://api.openai.com/v1")
    engine = st.selectbox("引擎", ["ollama", "openai", "deepl"], index=1 if default_api_key else 0)
    if engine == "ollama":
        model = st.text_input("模型", value="qwen2.5:7b")
        base_url = st.text_input("Ollama 地址", value="http://localhost:11434")
        api_key = None
    elif engine == "openai":
        model = st.text_input("模型", value=os.environ.get("PAPERSEARCH_MODEL", "gpt-4o-mini"))
        base_url = st.text_input("Base URL", value=default_base_url)
        api_key = st.text_input("API Key", type="password", value=default_api_key)
    else:
        model, base_url = None, "https://api-free.deepl.com/v2"
        api_key = st.text_input("DeepL Auth Key", type="password", value=default_api_key)
    translate_on = st.toggle("翻译摘要", value=False)
    st.divider()
    st.caption(f"内置术语表 {Glossary.load().size} 条，保证专业名词译名一致。")


# ----------------------------------------------------------------------
# 检索 Tab
# ----------------------------------------------------------------------
def _render_search_tab() -> None:
    st.title("学术论文检索与翻译")
    query = st.text_input("输入研究关键词（英文效果最佳）", placeholder="e.g. graph neural network")
    col1, col2, col3, _ = st.columns([1, 1, 1, 3])
    sort_choice = col1.radio("排序", ["相关度", "最新"], horizontal=True, index=0)
    top_n = col2.number_input("Top N", min_value=1, max_value=20, value=5, step=1)
    search_clicked = col3.button("搜索", type="primary", use_container_width=True)

    if search_clicked or (query and st.session_state.last_query == query and st.session_state.papers):
        if search_clicked:
            with st.spinner("检索中（OpenAlex + Semantic Scholar + arXiv）..."):
                sort = "date" if sort_choice == "最新" else "relevance"
                st.session_state.papers = search_all(
                    query, limit_per_source=max(top_n * 2, 10), sort=sort
                )[:top_n]
                st.session_state.translated = {}
                st.session_state.last_query = query
        papers: list[Paper] = st.session_state.papers

        if not papers:
            st.warning("未检索到论文，换个关键词试试。")
        else:
            st.success(f"找到 {len(papers)} 篇相关论文")
            st.divider()

            # 翻译开关在检索后仍可触发
            if translate_on and not st.session_state.translated and papers:
                try:
                    translator = get_translator(engine=engine, api_key=api_key, model=model, base_url=base_url)
                except ValueError as exc:
                    st.error(str(exc))
                    translator = None
                if translator:
                    if not translator.available():
                        st.warning(
                            f"引擎 {engine} 当前不可用。Ollama 请先运行 `ollama serve` 并 `ollama pull {model}`；"
                            "或改用 openai 引擎填入 API Key。"
                        )
                    else:
                        progress = st.progress(0.0, text="翻译中...")
                        glossary = Glossary.load()
                        for i, p in enumerate(papers):
                            st.session_state.translated[p.title] = translate_with_glossary(
                                translator, p.abstract, glossary
                            )
                            progress.progress((i + 1) / len(papers), text=f"翻译中: {p.title[:40]}...")
                        progress.empty()

            translated = st.session_state.translated
            # 导出按钮（云端部署时下载到本地，不写服务器文件系统）
            if st.button("生成 Markdown 下载", use_container_width=True):
                from papersearch.output import papers_to_markdown
                md = papers_to_markdown(papers, translated)
                st.download_button(
                    "⬇️ 下载双语 Markdown",
                    data=md,
                    file_name=f"{query or 'result'}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            # 论文列表：可展开查看，可选中翻译详情
            for i, p in enumerate(papers, 1):
                with st.expander(
                    f"{i}. {p.title}  ({p.year_text} · {p.source} · 相关度 {p.relevance:.2f})",
                    expanded=(i == 1),
                ):
                    st.markdown(f"**作者**: {p.authors_text}")
                    st.markdown(f"**链接**: [{p.url}]({p.url})" + (f" · [PDF]({p.pdf_url})" if p.pdf_url else ""))
                    zh = translated.get(p.title)
                    if zh:
                        st.markdown("#### 中文摘要")
                        st.markdown(zh)
                        st.markdown("#### English Abstract")
                    st.markdown(p.abstract if not zh else f"<details><summary>原文</summary>{p.abstract}</details>", unsafe_allow_html=True)
    else:
        st.caption("输入关键词开始检索。覆盖 OpenAlex（2.5亿+ 论文）、Semantic Scholar 与 arXiv 预印本。")
        st.markdown(
            """
            **使用提示**
            - 关键词用英文效果最好，例如 `medical image segmentation`
            - 翻译默认走本地 Ollama（免费离线），也可在左侧切换到 OpenAI / DeepL
            - 术语表内置 100+ 条计算机/AI 学术术语，可自行扩展 `data/glossary.json`
            """
        )


# ----------------------------------------------------------------------
# PDF 翻译 Tab
# ----------------------------------------------------------------------
def _render_pdf_tab() -> None:
    from papersearch.pdf_translate import PdfTranslateError, pdf2zh_available, translate_pdf

    st.title("📄 PDF 整篇翻译")
    st.caption("上传外文论文 PDF，保留排版与公式输出中文版（基于 PDFMathTranslate）。")

    if not pdf2zh_available():
        st.warning(
            "当前环境未安装 pdf2zh（PDF 翻译引擎）。\n\n"
            "**本地运行请先安装**: `pip install pdf2zh`（依赖较大，约需 1-2 分钟）\n\n"
            "⚠️ 线上 Demo 默认不含此功能（云端无本地 Ollama 与 pdf2zh），请在本地使用。"
        )
        return

    pdf_file = st.file_uploader("选择 PDF 文件", type=["pdf"])
    if pdf_file is None:
        st.info("支持论文 PDF，翻译后输出「纯译文」与「双语对照」两个文件。")
        return

    c1, c2, c3 = st.columns([1, 1, 1])
    engine = c1.selectbox("翻译后端", ["ollama", "openai"], index=0)
    model = c2.text_input("模型", value="qwen2.5:7b")
    pages = c3.text_input("页数（留空=全部）", placeholder="如 1-3")

    if st.button("🚀 开始翻译", type="primary"):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            src_path = Path(tmp) / pdf_file.name
            src_path.write_bytes(pdf_file.getbuffer())
            with st.spinner("翻译中（整篇 PDF 较慢，请耐心等待）..."):
                try:
                    result = translate_pdf(
                        src_path,
                        tmp,
                        engine=engine,
                        model=model,
                        pages=pages or None,
                    )
                except PdfTranslateError as exc:
                    st.error(f"翻译失败: {exc}")
                    return
            st.success("翻译完成 ✅")
            c_left, c_right = st.columns(2)
            if result.mono:
                with c_left:
                    st.download_button(
                        "⬇️ 下载纯译文 PDF",
                        data=result.mono.read_bytes(),
                        file_name=result.mono.name,
                        mime="application/pdf",
                        use_container_width=True,
                    )
            if result.dual:
                with c_right:
                    st.download_button(
                        "⬇️ 下载双语对照 PDF",
                        data=result.dual.read_bytes(),
                        file_name=result.dual.name,
                        mime="application/pdf",
                        use_container_width=True,
                    )

# ----------------------------------------------------------------------
# 本地文献库 Tab
# ----------------------------------------------------------------------
def _render_library_tab() -> None:
    from papersearch.library import (
        LibraryError,
        default_db_path,
        list_papers,
        remove_paper,
        scan_directory,
        search_library,
    )

    st.title("📚 本地文献库")
    st.caption("扫描本地 PDF 论文建立文献库（SQLite），自动提取标题/作者/年份，支持全文搜索与语义搜索。")

    db_path = st.text_input("数据库路径", value=str(default_db_path()), key="lib_db_path")

    # --- 扫描导入 ---
    with st.expander("📥 扫描导入 PDF", expanded=True):
        col1, col2 = st.columns([3, 1])
        folder = col1.text_input("要扫描的文件夹路径", placeholder="e.g. C:/Users/you/Downloads/papers")
        recursive = col2.checkbox("递归子目录", value=True)
        if st.button("开始扫描", type="primary"):
            folder = folder.strip().strip('"')
            if not folder:
                st.error("请先填写文件夹路径")
            elif not os.path.isdir(folder):
                st.error(f"目录不存在: {folder}")
            else:
                progress_bar = st.progress(0.0, text="准备扫描...")

                def _on_progress(done: int, total: int, cur: str) -> None:
                    name = os.path.basename(cur)
                    progress_bar.progress(done / total, text=f"扫描中 ({done}/{total}): {name}")

                try:
                    result = scan_directory(
                        folder, db_path=db_path, recursive=recursive, progress=_on_progress
                    )
                except LibraryError as exc:
                    st.error(str(exc))
                else:
                    progress_bar.empty()
                    st.success(result.summary())
                    for err in result.errors[:10]:
                        st.caption(f"⚠️ {err}")

    # --- 向量索引（语义检索 / RAG 前置） ---
    with st.expander("🧠 向量索引（语义检索与问答的前置步骤）"):
        st.caption(
            "把论文全文分块并向量化（本地 Ollama + bge-m3）。"
            "第一次使用需要先 `ollama pull bge-m3`。索引后即可用语义搜索和论文问答。"
        )
        c1, c2 = st.columns([1, 1])
        embed_model = c1.text_input("embedding 模型", value="bge-m3", key="lib_embed_model")
        reindex = c2.checkbox("重建索引（清空旧的）", key="lib_reindex")
        if st.button("⚡ 建立向量索引", type="primary"):
            from papersearch.embeddings import EmbeddingError, OllamaEmbedder
            from papersearch.rag import RagError, index_papers

            embedder = OllamaEmbedder(model=embed_model.strip() or "bge-m3")
            if not embedder.available():
                st.error(f"embedding 模型 {embedder.model} 不可用，请先运行: ollama pull {embedder.model}")
            else:
                idx_progress = st.progress(0.0, text="准备索引...")

                def _on_idx(done: int, total: int, title: str) -> None:
                    idx_progress.progress(done / max(total, 1), text=f"索引中 ({done}/{total}): {title[:40]}")

                try:
                    result = index_papers(
                        db_path=db_path, embedder=embedder,
                        reindex=reindex, progress=_on_idx,
                    )
                except RagError as exc:
                    st.error(str(exc))
                else:
                    idx_progress.empty()
                    st.success(result.summary())
                    for err in result.failed[:10]:
                        st.caption(f"⚠️ {err}")

    # --- 搜索 ---
    c_sem, _ = st.columns([1, 5])
    semantic = c_sem.checkbox("🧠 语义搜索", value=False, key="lib_semantic",
                              help="按语义相关度检索（需要先建立向量索引）")
    query = st.text_input(
        "🔎 库内搜索",
        placeholder="输入关键词或问题；语义搜索时用自然语言描述你想找的内容",
        key="lib_query",
    )

    # --- 列表 ---
    try:
        if query.strip():
            if semantic:
                from papersearch.embeddings import OllamaEmbedder
                from papersearch.rag import RagError, semantic_search

                embedder = OllamaEmbedder(model=st.session_state.get("lib_embed_model", "bge-m3").strip() or "bge-m3")
                hits = semantic_search(query.strip(), db_path=db_path, embedder=embedder, top_k=5)
                st.caption(f"语义搜索「{query.strip()}」Top {len(hits)} 命中（相似度从高到低）")
                for i, hit in enumerate(hits, 1):
                    with st.expander(
                        f"[{i}] {hit.paper.title}  ({hit.paper.year_text} · 相似度 {hit.score:.3f})",
                        expanded=(i == 1),
                    ):
                        st.markdown(f"**作者**: {hit.paper.authors_text}")
                        st.markdown(f"**文件**: `{hit.paper.file_path}`")
                        st.markdown(f"**命中片段**: {hit.snippet}")
                papers = []
            else:
                papers = search_library(query.strip(), db_path=db_path, limit=100)
                st.caption(f"搜索「{query.strip()}」命中 {len(papers)} 篇（按年份倒序）")
        else:
            papers = list_papers(db_path=db_path, limit=200)
            st.caption(f"共 {len(papers)} 篇（按入库时间倒序，仅显示前 200 篇）")
    except LibraryError as exc:
        st.error(str(exc))
        papers = []
    except RagError as exc:
        st.error(str(exc))
        papers = []

    if papers:
        st.dataframe(
            [
                {
                    "id": p.id,
                    "标题": p.title,
                    "作者": p.authors_text,
                    "年份": p.year_text,
                    "arXiv": p.arxiv_id or "",
                    "大小": p.size_text,
                    "路径": p.file_path,
                }
                for p in papers
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption("下方可展开查看单篇详情（摘要等）。")
        for p in papers[:20]:
            with st.expander(f"[{p.id}] {p.title}  ({p.year_text})"):
                st.markdown(f"**作者**: {p.authors_text}")
                st.markdown(f"**文件**: `{p.file_path}`")
                st.markdown(f"**arXiv**: {p.arxiv_id or 'N/A'} | **入库时间**: {p.added_at}")
                if p.abstract:
                    st.markdown(f"**摘要**: {p.abstract[:400]}")
                if st.button("🗑️ 从文献库删除记录（不删文件）", key=f"lib_del_{p.id}"):
                    if remove_paper(p.id, db_path=db_path):
                        st.success(f"已删除记录 [{p.id}]（磁盘文件未动）")
                        try:
                            st.rerun()
                        except AttributeError:  # 旧版 Streamlit
                            st.experimental_rerun()
    elif not query.strip():
        st.info("文献库还是空的。填写上方文件夹路径并点击「开始扫描」导入 PDF。")


# ----------------------------------------------------------------------
# 论文问答 Tab（RAG）
# ----------------------------------------------------------------------
def _render_rag_tab() -> None:
    from papersearch.library import LibraryError, default_db_path
    from papersearch.rag import RagError, ask

    st.title("💬 论文问答（RAG）")
    st.caption("基于本地文献库的内容回答你的问题，并附引用来源（需要先建立向量索引）。")

    db_path = st.text_input("数据库路径", value=str(default_db_path()), key="rag_db_path")

    # 问答引擎配置（默认复用侧边栏翻译引擎；DeepL 不支持对话）
    with st.expander("⚙️ 问答引擎设置", expanded=False):
        c1, c2, c3 = st.columns([1, 1, 2])
        rag_engine = c1.selectbox("引擎", ["ollama", "openai"], index=0)
        rag_model = c2.text_input("模型", value="qwen2.5:7b" if rag_engine == "ollama" else "gpt-4o-mini")
        rag_api_key = c3.text_input("API Key（openai）", type="password",
                                    value=os.environ.get("PAPERSEARCH_API_KEY", "")) if rag_engine == "openai" else None
    rag_embed_model = st.text_input("embedding 模型", value="bge-m3", key="rag_embed_model")

    question = st.text_input("你的问题", placeholder="e.g. 这些论文里关于图神经网络的核心方法是什么？")
    top_k = st.slider("参考片段数", min_value=1, max_value=10, value=5)

    if st.button("💡 提问", type="primary") and question.strip():
        from papersearch.embeddings import OllamaEmbedder
        from papersearch.translate import get_translator

        embedder = OllamaEmbedder(model=rag_embed_model.strip() or "bge-m3")
        try:
            llm = get_translator(engine=rag_engine, api_key=rag_api_key, model=rag_model)
        except ValueError as exc:
            st.error(str(exc))
            return
        if not embedder.available():
            st.error(f"embedding 模型 {embedder.model} 不可用，请先运行: ollama pull {embedder.model}")
            return
        if not llm.available():
            st.error("问答引擎不可用（Ollama 未启动？模型未安装？）")
            return
        with st.spinner("检索相关论文片段并生成回答（本地模型可能需要一点时间）..."):
            try:
                answer = ask(
                    question.strip(), db_path=db_path,
                    embedder=embedder, llm=llm, top_k=top_k,
                )
            except RagError as exc:
                st.error(str(exc))
                return
        st.markdown("---")
        st.markdown("### 📝 回答")
        st.markdown(answer.answer)
        if answer.sources:
            with st.expander(f"📎 引用来源（{len(answer.sources)} 个片段）", expanded=True):
                for i, src in enumerate(answer.sources, 1):
                    st.markdown(
                        f"**[{i}] {src.paper.title}**  "
                        f"({src.paper.authors_text} · {src.paper.year_text} · 相似度 {src.score:.3f})"
                    )
                    st.markdown(f"`{src.paper.file_path}`")
                    st.markdown(f"> {src.text[:400]}")
                    st.markdown("---")
    elif not question.strip():
        st.info("输入问题开始问答。示例：「综述里提到的图神经网络有哪些应用？」")


# ----------------------------------------------------------------------
# 主区域：Tab 切换（论文检索 / PDF 翻译 / 本地文献库 / 论文问答）
# ----------------------------------------------------------------------
tab_search, tab_pdf, tab_library, tab_rag = st.tabs(
    ["🔍 论文检索", "📄 PDF 整篇翻译", "📚 本地文献库", "💬 论文问答"]
)

with tab_search:
    _render_search_tab()

with tab_pdf:
    _render_pdf_tab()

with tab_library:
    _render_library_tab()

with tab_rag:
    _render_rag_tab()
