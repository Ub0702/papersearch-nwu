"""PaperSearch Web UI（Streamlit）。

运行：
    streamlit run app.py

功能：输入关键词 -> 聚合检索 -> 逐篇查看中英双语摘要，可导出 Markdown。
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
# 主区域：检索
# ----------------------------------------------------------------------
st.title("学术论文检索与翻译")
query = st.text_input("输入研究关键词（英文效果最佳）", placeholder="e.g. graph neural network")
col1, col2, _ = st.columns([1, 1, 4])
top_n = col1.number_input("Top N", min_value=1, max_value=20, value=5, step=1)
search_clicked = col2.button("搜索", type="primary", use_container_width=True)

if search_clicked or (query and st.session_state.last_query == query and st.session_state.papers):
    if search_clicked:
        with st.spinner("检索中（OpenAlex + Semantic Scholar + arXiv）..."):
            st.session_state.papers = search_all(query, limit_per_source=max(top_n * 2, 10))[:top_n]
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
        c1, c2 = st.columns([1, 1])
        if c1.button("生成 Markdown 下载", use_container_width=True):
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
