"""语义检索：本地 embedding + 向量相似度计算（M5）。

- OllamaEmbedder：调用本地 Ollama 的 /api/embed 接口，把文本变成向量
  （默认 bge-m3，多语言模型，中文提问检索英文论文效果好）
- cosine_top_k：numpy 实现的余弦相似度 Top-K 检索（零额外依赖）

设计原则：复用现有 Ollama 基础设施（和翻译层同一套服务），
向量检索用 numpy 暴力计算——文献库规模（几千个分块）毫秒级完成，
不引入 chromadb/faiss 等重依赖，实现透明可讲。
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import requests

from .translate import OLLAMA_URL

#: 默认 embedding 模型（bge-m3：多语言，1024 维，中英混合检索效果好）
DEFAULT_EMBED_MODEL = "bge-m3"


class EmbeddingError(RuntimeError):
    """embedding 可预期错误（Ollama 未运行 / 模型未拉取 / 请求失败）。"""


class OllamaEmbedder:
    """本地 Ollama embedding 客户端。

    用法:
        embedder = OllamaEmbedder()
        vecs = embedder.embed(["hello", "world"])   # -> list[list[float]]
    """

    name = "ollama"

    def __init__(self, model: str = DEFAULT_EMBED_MODEL, base_url: str = OLLAMA_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def available(self) -> bool:
        """检查 Ollama 服务是否运行且模型是否已拉取。"""
        try:
            req = urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2)
            data = json.loads(req.read().decode("utf-8"))
            models = {m.get("name", "") for m in data.get("models", [])}
            return any(self.model in name for name in models)
        except Exception:
            return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把一批文本转为向量。texts 为空时返回空列表。"""
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return []
        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=120,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise EmbeddingError(
                f"embedding 请求失败: {exc}（请确认 Ollama 已启动且已执行 ollama pull {self.model}）"
            ) from exc
        data = resp.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise EmbeddingError(f"Ollama 返回异常: {data.get('error', '无 embeddings 字段')}")
        return embeddings


# ---------------------------------------------------------------------------
# 余弦相似度 Top-K（numpy 暴力检索）
# ---------------------------------------------------------------------------

def cosine_top_k(
    query_vec: list[float],
    matrix: np.ndarray,
    k: int = 5,
) -> list[tuple[int, float]]:
    """在向量矩阵中找与 query 最相似的 k 行。

    参数:
        query_vec: 查询向量（一维）
        matrix:   候选向量矩阵，shape (n, dim)
        k:        返回数量

    返回:
        [(行索引, 余弦相似度)]，按相似度从高到低。k > n 时返回全部。
    """
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return []
    q = np.asarray(query_vec, dtype=np.float32)
    # 归一化 + 矩阵乘法 = 批量余弦相似度
    q_norm = q / (np.linalg.norm(q) + 1e-9)
    m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    scores = m_norm @ q_norm
    k = min(k, len(scores))
    # 用 argpartition 取 top-k 索引（比全排序快，返回顺序乱，再按分数排）
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]
