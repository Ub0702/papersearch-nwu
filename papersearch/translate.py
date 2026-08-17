"""翻译层：可插拔的翻译引擎 + 术语表感知的翻译流水线。

内置三种引擎（通过 --engine 切换）：
- ollama : 本地 Ollama 服务（默认，免费，无需 API Key）
- openai : 任意 OpenAI 兼容 API（可配置 base_url，如 DeepSeek/通义/自建 vLLM）
- deepl  : DeepL API（需 Key）

翻译流程：术语保护 -> 调用引擎 -> 术语还原，确保专业名词译名一致。
"""

from __future__ import annotations

import json
import urllib.request
from abc import ABC, abstractmethod

import requests

from .glossary import Glossary

#: 默认 Ollama 模型（学术翻译够用；qwen2.5 中文质量好）
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434"

SYSTEM_PROMPT = (
    "You are an academic translator. Translate the given text from English to "
    "Simplified Chinese (中文简体). Requirements:\n"
    "1. Keep academic rigor: translate technical terms precisely and consistently;\n"
    "2. Preserve LaTeX formulas, numbers, citations, and code verbatim;\n"
    "3. Output ONLY the translation, no explanations, no notes;\n"
    "4. Use formal academic Chinese (书面语), suitable for a research paper."
)


class Translator(ABC):
    """翻译引擎抽象接口。"""

    name = "base"

    @abstractmethod
    def translate(self, text: str) -> str:
        """把英文文本翻译为简体中文。"""

    def available(self) -> bool:
        return True


class OllamaTranslator(Translator):
    """本地 Ollama 引擎，完全免费、离线可用。"""

    name = "ollama"

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, base_url: str = OLLAMA_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def available(self) -> bool:
        try:
            req = urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2)
            data = json.loads(req.read().decode("utf-8"))
            models = {m.get("name", "") for m in data.get("models", [])}
            return any(self.model in name for name in models)
        except Exception:
            return False

    def translate(self, text: str) -> str:
        payload = {
            "model": self.model,
            "prompt": f"{SYSTEM_PROMPT}\n\nText to translate:\n{text}",
            "stream": False,
        }
        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


class OpenAITranslator(Translator):
    """OpenAI 兼容引擎：支持 OpenAI / DeepSeek / 通义 / 任何兼容服务。"""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def translate(self, text: str) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.2,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


class DeepLTranslator(Translator):
    """DeepL 引擎（需官方 API Key，翻译质量高）。"""

    name = "deepl"

    def __init__(self, api_key: str, base_url: str = "https://api-free.deepl.com/v2"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def translate(self, text: str) -> str:
        resp = requests.post(
            f"{self.base_url}/translate",
            headers={"Authorization": f"DeepL-Auth-Key {self.api_key}"},
            data={"text": text, "target_lang": "ZH"},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["translations"][0]["text"]


# ----------------------------------------------------------------------
# 工厂与流水线
# ----------------------------------------------------------------------

def get_translator(
    engine: str = "ollama",
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> Translator:
    """按名称创建翻译引擎实例。"""
    engine = (engine or "ollama").lower()
    if engine == "ollama":
        return OllamaTranslator(model=model or DEFAULT_OLLAMA_MODEL, base_url=base_url or OLLAMA_URL)
    if engine == "openai":
        if not api_key:
            raise ValueError("openai 引擎需要 API Key，请通过 --api-key 或 PAPERSEARCH_API_KEY 环境变量提供")
        return OpenAITranslator(api_key=api_key, model=model or "gpt-4o-mini", base_url=base_url or "https://api.openai.com/v1")
    if engine == "deepl":
        if not api_key:
            raise ValueError("deepl 引擎需要 API Key")
        return DeepLTranslator(api_key=api_key, base_url=base_url or "https://api-free.deepl.com/v2")
    raise ValueError(f"未知翻译引擎: {engine}（可选: ollama / openai / deepl）")


def translate_with_glossary(translator: Translator, text: str, glossary: Glossary) -> str:
    """术语表感知的翻译：保护术语 -> 翻译 -> 还原术语。"""
    if not text.strip():
        return ""
    protected = glossary.protect(text)
    translated = translator.translate(protected)
    return glossary.restore(translated)
