"""翻译流水线测试：术语保护 -> 翻译 -> 还原。"""

from papersearch.glossary import Glossary
from papersearch.translate import Translator, translate_with_glossary


class FakeTranslator(Translator):
    """模拟翻译引擎：简单把句子前后翻转，保留占位符。"""

    name = "fake"

    def translate(self, text: str) -> str:
        # 模拟"翻译"：把每个词后面加个“译”字并调换顺序，占位符原样保留
        words = text.split()
        return " ".join(w + "译" for w in reversed(words))


def test_glossary_protects_terms_through_pipeline():
    glossary = Glossary({"deep learning": "深度学习", "neural network": "神经网络"})
    text = "Deep learning and neural network are popular."
    result = translate_with_glossary(FakeTranslator(), text, glossary)
    # 术语的占位符被 FakeTranslator 保留，restore 后还原为术语表中文
    assert "深度学习" in result
    assert "神经网络" in result
    # 非术语词经过了引擎的"翻译"处理
    assert "译" in result


def test_translate_empty_text():
    glossary = Glossary.load()
    assert translate_with_glossary(FakeTranslator(), "   ", glossary) == ""
