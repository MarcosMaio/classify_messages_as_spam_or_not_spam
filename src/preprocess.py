"""Text cleaning for SMS: NLTK tokenization, stopwords, lemmatization."""

from __future__ import annotations

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_nltk_ready = False
_lemmatizer: WordNetLemmatizer | None = None
_stop_words: set[str] | None = None


def _ensure_nltk_resource(package: str, find_path: str) -> None:
    try:
        nltk.data.find(find_path)
    except LookupError:
        nltk.download(package, quiet=True)


def ensure_nltk_data() -> None:
    """Download required NLTK corpora/tokenizers once (idempotent)."""
    global _nltk_ready, _lemmatizer, _stop_words
    if _nltk_ready:
        return

    _ensure_nltk_resource("punkt", "tokenizers/punkt")
    # NLTK 3.8+ uses punkt_tab for word_tokenize in some builds
    try:
        _ensure_nltk_resource("punkt_tab", "tokenizers/punkt_tab")
    except Exception:
        pass
    _ensure_nltk_resource("stopwords", "corpora/stopwords")
    _ensure_nltk_resource("wordnet", "corpora/wordnet")
    _ensure_nltk_resource("omw-1.4", "corpora/omw-1.4")

    _lemmatizer = WordNetLemmatizer()
    _stop_words = set(stopwords.words("english"))
    _nltk_ready = True


def preprocess_text(text: str, *, remove_stopwords: bool = True) -> str:
    """
    Normalize SMS text for bag-of-words / TF-IDF.

    - Lowercase
    - Keep tokens that are alphanumeric or contain digits (useful for spam URLs/numbers)
    - Remove punctuation-only tokens
    - Optional English stopword removal
    - Lemmatize (WordNet)
    """
    ensure_nltk_data()
    assert _lemmatizer is not None and _stop_words is not None

    if not isinstance(text, str) or not text.strip():
        return ""

    lowered = text.lower()
    tokens = word_tokenize(lowered)

    cleaned: list[str] = []
    for tok in tokens:
        # Preserve digit-containing tokens (e.g. call 0800, win1000)
        if any(ch.isdigit() for ch in tok):
            alnum = re.sub(r"[^\w]", "", tok)
            if alnum:
                cleaned.append(alnum)
            continue
        # Drop pure punctuation
        if all(ch in string.punctuation for ch in tok):
            continue
        word = re.sub(r"[^\w]", "", tok)
        if not word:
            continue
        if remove_stopwords and word in _stop_words:
            continue
        cleaned.append(_lemmatizer.lemmatize(word))

    return " ".join(cleaned)


def preprocess_series(texts, *, remove_stopwords: bool = True):
    """Apply :func:`preprocess_text` to a pandas Series or iterable of strings."""
    return [preprocess_text(t, remove_stopwords=remove_stopwords) for t in texts]
