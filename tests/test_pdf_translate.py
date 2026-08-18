"""pdf_translate 模块的单元测试。

不真正调用 pdf2zh（避免下载模型/依赖 Ollama），通过 mock 子进程验证：
- 参数构造（service=ollama:模型名）
- 成功路径：mono/dual 文件存在性判断
- 失败路径：非零退出码 -> PdfTranslateError
- 缺依赖路径：pdf2zh 未安装 -> PdfTranslateError
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from papersearch.pdf_translate import (
    PdfTranslateError,
    PdfTranslateResult,
    translate_pdf,
)


@pytest.fixture
def fake_pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")
    return pdf


@pytest.fixture
def mock_pdf2zh_installed(monkeypatch):
    """让 pdf2zh_available() 返回 True。"""
    from papersearch import pdf_translate

    monkeypatch.setattr(pdf_translate, "pdf2zh_available", lambda: True)


def _run_success(monkeypatch, fake_pdf: Path, tmp_path: Path) -> PdfTranslateResult:
    """模拟 pdf2zh 成功输出两个 PDF 文件。"""
    mono = tmp_path / f"{fake_pdf.stem}-zh.pdf"
    dual = tmp_path / f"{fake_pdf.stem}-dual.pdf"
    mono.write_bytes(b"%PDF mono")
    dual.write_bytes(b"%PDF dual")

    def fake_run(cmd, **kwargs):
        # 走 python -c 包装器：第一个参数必须是 -c，随后是补丁代码
        assert cmd[0].endswith("python") or cmd[0].endswith("python.exe")
        assert cmd[1] == "-c"
        # 参数列表最后是 pdf2zh CLI 参数，校验 service
        assert "--service" in cmd
        service_idx = cmd.index("--service")
        assert cmd[service_idx + 1] == "ollama:qwen2.5:7b"
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return translate_pdf(fake_pdf, tmp_path, engine="ollama", model="qwen2.5:7b")


def test_translate_success(monkeypatch, fake_pdf: Path, tmp_path: Path, mock_pdf2zh_installed):
    result = _run_success(monkeypatch, fake_pdf, tmp_path)
    assert result.ok
    assert result.mono is not None and result.mono.name == "paper-zh.pdf"
    assert result.dual is not None and result.dual.name == "paper-dual.pdf"


def test_service_passthrough(monkeypatch, fake_pdf: Path, tmp_path: Path, mock_pdf2zh_installed):
    """openai 引擎时 service 直接传引擎名。"""
    mono = tmp_path / f"{fake_pdf.stem}-zh.pdf"
    mono.write_bytes(b"%PDF mono")

    def fake_run(cmd, **kwargs):
        service_idx = cmd.index("--service")
        assert cmd[service_idx + 1] == "openai"
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = translate_pdf(fake_pdf, tmp_path, engine="openai", model="gpt-4o-mini")
    assert result.ok


def test_failure_nonzero_exit(monkeypatch, fake_pdf: Path, tmp_path: Path, mock_pdf2zh_installed):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom error")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(PdfTranslateError, match="boom error"):
        translate_pdf(fake_pdf, tmp_path)


def test_missing_input(tmp_path: Path, mock_pdf2zh_installed):
    with pytest.raises(PdfTranslateError, match="输入文件不存在"):
        translate_pdf(tmp_path / "nope.pdf", tmp_path)


def test_pdf2zh_not_installed(fake_pdf: Path, tmp_path: Path, monkeypatch):
    """没有 pdf2zh 时给出可读错误。"""
    from papersearch import pdf_translate

    monkeypatch.setattr(pdf_translate, "pdf2zh_available", lambda: False)
    with pytest.raises(PdfTranslateError, match="pdf2zh"):
        translate_pdf(fake_pdf, tmp_path)
