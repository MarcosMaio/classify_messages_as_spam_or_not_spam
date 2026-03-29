"""
Train TF-IDF + Multinomial Naive Bayes and Logistic Regression on SMS Spam.

Spam recall matters: missing spam is often worse than a false alarm for filters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from load_data import (  # noqa: E402
    default_raw_path,
    download_sms_spam_collection,
    load_sms_spam,
)
from preprocess import ensure_nltk_data, preprocess_series  # noqa: E402


def models_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


def train_and_evaluate(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 10_000,
    remove_stopwords: bool = True,
) -> dict:
    ensure_nltk_data()
    raw_path = default_raw_path()
    if not raw_path.is_file():
        download_sms_spam_collection()

    df = load_sms_spam(raw_path)
    X_raw = df["text"].tolist()
    y = df["label"].values

    X_clean = preprocess_series(X_raw, remove_stopwords=remove_stopwords)

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_vec, y_train)
    y_pred_nb = nb.predict(X_test_vec)

    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    lr.fit(X_train_vec, y_train)
    y_pred_lr = lr.predict(X_test_vec)

    results = {
        "naive_bayes": {
            "accuracy": float(accuracy_score(y_test, y_pred_nb)),
            "report": classification_report(
                y_test, y_pred_nb, target_names=["ham", "spam"]
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred_nb).tolist(),
        },
        "logistic_regression": {
            "accuracy": float(accuracy_score(y_test, y_pred_lr)),
            "report": classification_report(
                y_test, y_pred_lr, target_names=["ham", "spam"]
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred_lr).tolist(),
        },
    }
    return {
        "vectorizer": vectorizer,
        "nb": nb,
        "lr": lr,
        "metrics": results,
        "preprocess_config": {"remove_stopwords": remove_stopwords},
    }


def save_artifacts(bundle: dict, out_dir: Path | None = None) -> None:
    out_dir = out_dir or models_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle["vectorizer"], out_dir / "tfidf_vectorizer.joblib")
    joblib.dump(bundle["nb"], out_dir / "multinomial_nb.joblib")
    joblib.dump(bundle["lr"], out_dir / "logistic_regression.joblib")
    joblib.dump(bundle["preprocess_config"], out_dir / "preprocess_config.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SMS spam classifiers (TF-IDF).")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=10_000)
    parser.add_argument(
        "--keep-stopwords",
        action="store_true",
        help="Do not remove English stopwords during preprocessing.",
    )
    args = parser.parse_args()

    bundle = train_and_evaluate(
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
        remove_stopwords=not args.keep_stopwords,
    )

    save_artifacts(bundle)

    print("=== Multinomial Naive Bayes ===")
    print(f"Accuracy: {bundle['metrics']['naive_bayes']['accuracy']:.4f}")
    print(bundle["metrics"]["naive_bayes"]["report"])
    print("Confusion matrix [ [TN FP], [FN TP] ] row=true col=pred (0=ham,1=spam):")
    print(bundle["metrics"]["naive_bayes"]["confusion_matrix"])

    print("\n=== Logistic Regression ===")
    print(f"Accuracy: {bundle['metrics']['logistic_regression']['accuracy']:.4f}")
    print(bundle["metrics"]["logistic_regression"]["report"])
    print("Confusion matrix:")
    print(bundle["metrics"]["logistic_regression"]["confusion_matrix"])

    print(f"\nArtifacts saved under: {models_dir()}")


if __name__ == "__main__":
    main()
