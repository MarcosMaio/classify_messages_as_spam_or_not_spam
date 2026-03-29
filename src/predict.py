"""Run inference with saved TF-IDF vectorizer + classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from preprocess import preprocess_text  # noqa: E402


def models_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


def load_artifacts(models_path: Path | None = None):
    root = models_path or models_dir()
    vectorizer = joblib.load(root / "tfidf_vectorizer.joblib")
    nb = joblib.load(root / "multinomial_nb.joblib")
    lr = joblib.load(root / "logistic_regression.joblib")
    preprocess_config = joblib.load(root / "preprocess_config.joblib")
    return vectorizer, nb, lr, preprocess_config


def predict_one(
    text: str,
    *,
    model_name: str = "lr",
    models_path: Path | None = None,
) -> tuple[str, float]:
    """
    Returns (label_name, proba_spam) using ``lr`` or ``nb`` model.
    proba_spam is P(spam) from predict_proba when available.
    """
    vectorizer, nb, lr, cfg = load_artifacts(models_path)
    remove_sw = cfg.get("remove_stopwords", True)
    clean = preprocess_text(text, remove_stopwords=remove_sw)
    X = vectorizer.transform([clean])

    if model_name == "nb":
        clf = nb
    elif model_name == "lr":
        clf = lr
    else:
        raise ValueError("model_name must be 'nb' or 'lr'")

    pred = int(clf.predict(X)[0])
    if hasattr(clf, "predict_proba"):
        proba = float(clf.predict_proba(X)[0, 1])
    else:
        proba = float("nan")

    label = "spam" if pred == 1 else "ham"
    return label, proba


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a message as spam or ham.")
    parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="Single message string. If omitted, reads one line from stdin.",
    )
    parser.add_argument(
        "--model",
        choices=("lr", "nb"),
        default="lr",
        help="Classifier to use (default: logistic regression).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Override directory containing .joblib artifacts.",
    )
    args = parser.parse_args()

    if args.message is None:
        text = sys.stdin.readline().strip()
    else:
        text = args.message

    if not text:
        print("Error: empty message.", file=sys.stderr)
        sys.exit(1)

    label, proba = predict_one(
        text, model_name=args.model, models_path=args.models_dir
    )
    if proba == proba:  # not NaN
        print(f"{label} (P(spam)={proba:.4f})")
    else:
        print(label)


if __name__ == "__main__":
    main()
