"""术语表机制测试：保护/还原与长术语优先。"""

from papersearch.glossary import Glossary


def test_basic_protect_restore():
    g = Glossary({"deep learning": "深度学习", "neural network": "神经网络"})
    text = "Deep learning with neural network is popular."
    protected = g.protect(text)
    assert "deep learning" not in protected.lower()
    restored = g.restore(protected)
    assert restored == "深度学习 with 神经网络 is popular."


def test_long_term_priority():
    g = Glossary({"learning": "学习", "deep learning": "深度学习"})
    text = "Deep learning is powerful."
    assert g.protect(text) == "[[T0]] is powerful."
    assert g.restore(g.protect(text)) == "深度学习 is powerful."


def test_case_insensitive():
    g = Glossary({"Transformer": "Transformer 模型"})
    assert g.restore(g.protect("the transformer architecture")) == "the Transformer 模型 architecture"


def test_load_default():
    g = Glossary.load()
    assert g.size > 0
