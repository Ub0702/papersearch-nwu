"""PaperSearch - 学术论文检索与学术级翻译工具。

输入关键词 -> 聚合检索国外论文 (OpenAlex / Semantic Scholar / arXiv)
           -> 术语表感知的 LLM 翻译 -> 中英双语对照输出
           -> 整篇 PDF 翻译（保留排版与公式）
           -> 本地文献库管理（扫描导入 + 元数据提取 + 全文搜索）
           -> 语义检索 + RAG 问答（bge-m3 向量索引，引用来源）
"""

__version__ = "0.3.0"
