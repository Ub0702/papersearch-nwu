"""整篇 PDF 翻译（基于 PDFMathTranslate / pdf2zh）。

设计说明：
- 通过子进程调用 pdf2zh 的 CLI 入口（python -c 包装器），而不是直接
  import 其 Python API。原因：pdf2zh 各版本 API 差异大（1.7.x 无
  translate 函数，只能走 CLI），子进程方式对版本变化最鲁棒，且崩溃隔离。
- 子进程启动时注入 numpy 兼容补丁：pdf2zh 1.7.x 用了 numpy 2.x 移除的
  np.fromstring 二进制模式，补丁使其在任意 numpy 版本下可用。
- pdf2zh 是可选依赖（体积大：torch/opencv 等）。未安装时抛出
  PdfTranslateError，调用方（CLI / Web UI）负责给出友好提示。
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


class PdfTranslateError(RuntimeError):
    """PDF 翻译相关的可预期错误（缺依赖、超时、翻译失败等）。"""


@dataclass
class PdfTranslateResult:
    """一次 PDF 翻译的输出。"""

    input_path: Path
    output_dir: Path
    mono: Path | None = None  # 纯译文版 {name}-zh.pdf
    dual: Path | None = None  # 双语对照版 {name}-dual.pdf
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.mono is not None and self.mono.exists()

    def summary(self) -> str:
        lines = [f"输入: {self.input_path.name}"]
        if self.mono:
            lines.append(f"纯译文: {self.mono}")
        if self.dual:
            lines.append(f"双语对照: {self.dual}")
        return "\n".join(lines)


def pdf2zh_available() -> bool:
    """pdf2zh 是否已安装（作为可选依赖）。"""
    try:
        import pdf2zh  # noqa: F401

        return True
    except ImportError:
        return False


#: pdf2zh 1.7.x 与 numpy>=2 的兼容补丁。
#: 1.7.x 内部使用 np.fromstring 读取图像字节流，numpy 2.x 移除了该 API 的
#: 二进制模式。此补丁在子进程启动时注入：bytes/bytearray 输入走
#: np.frombuffer（语义等价），其余走原 fromstring。
#: 注意：必须用多行字符串（python -c 支持换行），if 单行 suite 内不能定义 def。
_NUMPY_COMPAT_BOOT = (
    "import numpy as np\n"
    "if not hasattr(np.fromstring, '_ps_compat'):\n"
    "    _orig = np.fromstring\n"
    "    def _compat(s, dtype=float, count=-1, sep=''):\n"
    "        if isinstance(s, (bytes, bytearray, memoryview)):\n"
    "            return np.frombuffer(s, dtype=dtype, count=count)\n"
    "        return _orig(s, dtype=dtype, count=count, sep=sep)\n"
    "    _compat._ps_compat = True\n"
    "    np.fromstring = _compat\n"
    "from pdf2zh.pdf2zh import main\n"
    "import sys\n"
    "sys.exit(main())\n"
)


def translate_pdf(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    engine: str = "ollama",
    model: str = "qwen2.5:7b",
    lang_in: str = "en",
    lang_out: str = "zh",
    pages: str | None = None,
    thread: int = 4,
    timeout: int = 3600,
) -> PdfTranslateResult:
    """翻译整篇 PDF，保留排版与公式，输出纯译文 + 双语对照两个 PDF。

    Args:
        input_path: 输入 PDF 路径。
        output_dir: 输出目录（默认输入文件同目录）。
        engine: 翻译后端，透传给 pdf2zh 的 -s 参数。可选
            ollama / openai / deepl / google 等；本地默认 ollama。
        model: 模型名（engine=ollama 时如 qwen2.5:7b；openai 时如 gpt-4o-mini）。
        lang_in / lang_out: 源语言 / 目标语言代码（如 en / zh）。
        pages: 只翻译指定页，如 "1"、"1-3"、"1,3"；None 表示全部。
        thread: 并行翻译线程数。
        timeout: 子进程超时秒数（整篇 PDF 翻译可能较慢，默认 1 小时）。

    Returns:
        PdfTranslateResult，含 mono（纯译文）与 dual（双语对照）路径。

    Raises:
        PdfTranslateError: pdf2zh 未安装、输入不存在或翻译失败。
    """
    src = Path(input_path).resolve()
    if not src.exists():
        raise PdfTranslateError(f"输入文件不存在: {src}")

    if not pdf2zh_available():
        raise PdfTranslateError(
            "未检测到 pdf2zh（PDF 翻译引擎）。请先安装: pip install pdf2zh\n"
            "注意: pdf2zh 依赖较大（torch/opencv），且要求 Python 3.10-3.13。"
        )

    out_dir = Path(output_dir).resolve() if output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # service 参数: ollama 用 "ollama:模型名"，其余直接透传
    service = f"{engine}:{model}" if engine == "ollama" else engine

    # 通过 python -c 启动：先注入 numpy 兼容补丁，再调用 pdf2zh CLI。
    # 参数从 sys.argv[1:] 传入（pdf2zh.pdf2zh.main() 无参时如此解析）。
    cmd = [
        sys.executable,
        "-c",
        _NUMPY_COMPAT_BOOT,
        str(src),
        "--lang-in",
        lang_in,
        "--lang-out",
        lang_out,
        "--service",
        service,
        "--thread",
        str(thread),
    ]
    if pages:
        cmd += ["--pages", pages]

    try:
        proc = subprocess.run(
            cmd,
            cwd=out_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        raise PdfTranslateError(
            f"翻译超时（>{timeout}s）。整篇 PDF 翻译较慢，可增大 timeout 或减少 pages。"
        ) from None
    except OSError as exc:
        raise PdfTranslateError(f"无法启动 pdf2zh: {exc}") from None

    # pdf2zh 输出到当前工作目录（即 out_dir），命名: {stem}-zh.pdf / {stem}-dual.pdf
    mono = out_dir / f"{src.stem}-zh.pdf"
    dual = out_dir / f"{src.stem}-dual.pdf"

    if proc.returncode != 0:
        raise PdfTranslateError(
            f"pdf2zh 翻译失败（exit={proc.returncode}）\n"
            f"stderr: {proc.stderr[-2000:]}"
        )

    return PdfTranslateResult(
        input_path=src,
        output_dir=out_dir,
        mono=mono if mono.exists() else None,
        dual=dual if dual.exists() else None,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
