import preprocess


def test_preprocess_lowercases_and_strips_noise() -> None:
    preprocess.ensure_nltk_data()
    out = preprocess.preprocess_text("HELLO friend!!!", remove_stopwords=False)
    assert "hello" in out
    assert "friend" in out


def test_preprocess_empty() -> None:
    preprocess.ensure_nltk_data()
    assert preprocess.preprocess_text("") == ""
    assert preprocess.preprocess_text("   ") == ""
