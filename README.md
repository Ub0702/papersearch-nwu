# PaperSearch

[![CI](https://github.com/Ub0702/papersearch-nwu/actions/workflows/ci.yml/badge.svg)](https://github.com/Ub0702/papersearch-nwu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Ub0702/papersearch-nwu/blob/main/LICENSE)

> 学术论文检索 + 学术级翻译 —— 输入关键词，聚合检索国外论文，输出**术语一致的中英双语对照**。

PaperSearch 解决研究生/本科生的文献阅读痛点：查文献要跨多个数据库，读外文文献被专业术语卡住。
本项目把「检索」与「翻译」串成一条流水线，并用**学术术语表**保证专业名词的译名全程一致。

## 特性

- **多源聚合检索**：OpenAlex（2.5亿+ 论文）+ Semantic Scholar（2亿+）+ arXiv（预印本），免费 API、无需 Key；单源失败自动兜底
- **术语表感知翻译**：内置 100+ 条计算机/AI 学术术语，翻译前保护、翻译后还原，专业名词译名统一
- **可插拔翻译引擎**：默认本地 Ollama（免费离线），支持任意 OpenAI 兼容 API（DeepSeek/通义/vLLM）与 DeepL
- **双语对照输出**：Markdown / HTML，逐段原文 + 译文对照，公式与引用原样保留
- **两种入口**：命令行 CLI + Streamlit Web UI

## 效果演示

真实运行（本地 Ollama + qwen2.5:7b，RTX 4060）：

```bash
$ python -m papersearch "graph neural network" -n 3 --translate
[info] 检索中（OpenAlex + Semantic Scholar + arXiv）...
[info] OpenAlex      : 3 篇
[info] SemanticSchola: 0 篇（限流跳过，已自动兜底）
[info] arXiv         : 3 篇
[info] 翻译引擎: ollama (qwen2.5:7b)
[info] 已输出: output/graph_neural_network.html / .md / .json
```

翻译效果（论文摘要，中文为 qwen2.5:7b 生成）：

> **英文**：We propose a novel framework that systematically converts past model design experience into structured, fine-grained knowledge priors well-suited for meta-learning with LLMs. By constructing a solid meta-knowledge between unseen graph understanding and known effective architecture patterns, DesiGNN can deliver top-5.77% initial model proposals for unseen datasets within seconds.
>
> **中文**：我们提出了 DesiGNN，一种以知识为中心的框架，系统地将过去的设计经验转换为结构化、细致的知识先验，以适应大规模语言模型（LLMs）的元学习。通过在未知图理解与已知有效架构模式之间构建坚实的元知识，DesiGNN 可在几秒钟内为未知数据集提供排名前 5.77% 的初始模型提案。

术语一致性由术语表保证：`GNN → 图神经网络`、`LLM → 大规模语言模型`、`MADRL → 多智能体深度强化学习`、`FL → 联邦学习`、`AoI → 信息年龄`、`RSU → 路侧单元`——全文译名统一，不随模型措辞漂移。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt          # 仅 CLI
pip install streamlit                    # 如需 Web UI

# 2. 检索论文（不翻译）
python -m papersearch "graph neural network"

# 3. 检索 + 翻译摘要（默认 Ollama，需先启动本地服务）
ollama pull qwen2.5:7b                   # 一次性：下载模型
ollama serve                             # 启动服务
python -m papersearch "medical image segmentation" --translate

# 3.1 国内网络拉取模型超时的解决办法
#     registry.ollama.ai 在国内需代理。先启动代理，再设置环境变量后重启 Ollama：
setx HTTPS_PROXY http://127.0.0.1:7897   # Windows，端口换成你的代理端口
setx HTTP_PROXY  http://127.0.0.1:7897
# 然后完全退出 Ollama（含托盘图标）再重新打开，最后 ollama pull qwen2.5:7b

# 4. 使用 OpenAI 兼容 API
export PAPERSEARCH_API_KEY=sk-xxx        # Windows: set PAPERSEARCH_API_KEY=sk-xxx
python -m papersearch "diffusion model" --engine openai --model gpt-4o-mini

# 5. Web UI
streamlit run app.py
```

输出文件位于 `output/` 目录（Markdown + HTML 双语对照）。

## CLI 参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `query` | 检索关键词（英文效果最佳） | 必填 |
| `-n / --top` | 返回论文数量 | 5 |
| `-t / --translate` | 翻译摘要并输出双语对照 | 关 |
| `--engine` | `ollama` / `openai` / `deepl` | `ollama` |
| `--model` | 引擎模型名 | `qwen2.5:7b` |
| `--api-key` | API Key（openai/deepl） | 环境变量 |
| `--glossary` | 自定义术语表 JSON 路径 | 内置表 |
| `--out-dir` | 输出目录 | `output/` |

## 架构

```
检索层  OpenAlexSource / SemanticScholarSource / ArxivSource
       （search.py，统一 Source 接口，单源失败自动兜底）
   ↓
翻译层  OllamaTranslator / OpenAITranslator / DeepLTranslator
   ↓   + Glossary 术语表：protect() -> translate -> restore()
排版层  Markdown / HTML 双语对照                     （output.py）
   ↓
入口    CLI (cli.py) / Streamlit UI (app.py)
```

**新增数据源**：继承 `Source` 实现 `search()`，在 `search_all()` 中注册即可。
**新增翻译引擎**：继承 `Translator` 实现 `translate()`，在 `get_translator()` 中注册即可。

### 环境变量

| 变量 | 用途 |
|---|---|
| `PAPERSEARCH_API_KEY` | openai/deepl 引擎的 API Key |
| `PAPERSEARCH_BASE_URL` | 自定义 OpenAI 兼容服务地址 |
| `PAPERSEARCH_S2_API_KEY` | Semantic Scholar API Key（可选，免费注册后限流额度大幅提升） |

## 术语表

`data/glossary.json` 以 `{"英文术语": "中文译名"}` 格式维护。翻译时：

1. `protect()`：把正文中的术语替换为 `[[T0]]` 占位符（长术语优先匹配）
2. 调用翻译引擎
3. `restore()`：占位符还原为术语表中文译名

这样无论引擎怎么改措辞，术语译名始终由术语表决定。你可以按研究方向扩充，例如：

```bash
python -m papersearch "your query" --translate --glossary my_glossary.json
```

## 路线图

- [x] M0：项目骨架 + 多源检索（Semantic Scholar + arXiv）
- [x] M1：可插拔翻译引擎 + 术语表机制
- [x] M2：CLI + Web UI 双语对照
- [ ] M3：整篇 PDF 翻译（集成 PDFMathTranslate，保留公式排版）
- [ ] M4：中文论文检索（百度学术）与本地文献库管理
- [ ] M5：语义检索（向量索引）与 RAG 问答

## 测试

```bash
pip install pytest
python -m pytest        # 或直接 pytest（根目录 conftest.py 保证包可导入）
```

## License

[MIT](LICENSE)
