"""
Extension: sentence embeddings (BERT-family) + Logistic Regression.

Compares dense embeddings (sentence-transformers) with the TF-IDF sparse baseline
on the same train/test split. Heavier dependencies (torch); CPU is fine for this dataset size.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from load_data import (  # noqa: E402
    default_raw_path,
    download_sms_spam_collection,
    load_sms_spam,
)
from preprocess import preprocess_series  # noqa: E402
from train import train_and_evaluate  # noqa: E402


def run_bert_extension(
    *,
    model_name: str = "all-MiniLM-L6-v2",
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 64,
    use_preprocessed_text: bool = False,
    remove_stopwords: bool = True,
) -> dict:
    """Train LR on sentence embeddings; return metrics dict."""
    raw_path = default_raw_path()
    if not raw_path.is_file():
        download_sms_spam_collection()

    df = load_sms_spam(raw_path)
    texts = df["text"].tolist()
    y = df["label"].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if use_preprocessed_text:
        X_train = preprocess_series(X_train_raw, remove_stopwords=remove_stopwords)
        X_test = preprocess_series(X_test_raw, remove_stopwords=remove_stopwords)
    else:
        X_train = X_train_raw
        X_test = X_test_raw

    encoder = SentenceTransformer(model_name)
    X_train_emb = encoder.encode(
        X_train,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    X_test_emb = encoder.encode(
        X_test,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    lr = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
    )
    lr.fit(X_train_emb, y_train)
    y_pred = lr.predict(X_test_emb)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, target_names=["ham", "spam"]),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "model_name": model_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LR on sentence-BERT embeddings and compare to TF-IDF baselines."
    )
    parser.add_argument(
        "--sentence-model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model id (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--use-preprocessed",
        action="store_true",
        help="Apply same NLTK preprocessing as TF-IDF before encoding (usually worse for BERT).",
    )
    parser.add_argument(
        "--skip-tfidf-baseline",
        action="store_true",
        help="Only run embedding model (faster if you already have TF-IDF numbers).",
    )
    args = parser.parse_args()

    if not args.skip_tfidf_baseline:
        print("=== TF-IDF baselines (same settings as src/train.py defaults) ===\n")
        tfidf_bundle = train_and_evaluate(
            test_size=args.test_size,
            random_state=args.random_state,
            remove_stopwords=True,
        )
        m = tfidf_bundle["metrics"]
        print(f"Naive Bayes accuracy: {m['naive_bayes']['accuracy']:.4f}")
        print(m["naive_bayes"]["report"])
        print(f"Logistic Regression accuracy: {m['logistic_regression']['accuracy']:.4f}")
        print(m["logistic_regression"]["report"])
        print()

    print(
        f"=== Sentence embeddings + Logistic Regression ({args.sentence_model}) ===\n"
    )
    emb = run_bert_extension(
        model_name=args.sentence_model,
        test_size=args.test_size,
        random_state=args.random_state,
        use_preprocessed_text=args.use_preprocessed,
    )
    print(f"Accuracy: {emb['accuracy']:.4f}")
    print(emb["report"])
    print("Confusion matrix [[TN FP],[FN TP]] for labels ham=0, spam=1:")
    print(emb["confusion_matrix"])


if __name__ == "__main__":
    main()
