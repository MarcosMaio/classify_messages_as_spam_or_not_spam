"""
Train TF-IDF + Multinomial Naive Bayes and Logistic Regression on SMS Spam.

Spam recall matters: missing spam is often worse than a false alarm for filters.

CLI flags:
  --no-save          Only print metrics; do not write artifacts.
  --version v03      Save under models/versions/v03/ (default: auto-increment v01, v02, ...).
  --set-current      Write models/current to this version after save.
  --data-path PATH   TSV ham/spam file (same format as UCI); default: data/raw/SMSSpamCollection.
  --parent-version   Stored in metadata only (full retrain each run; no incremental partial_fit).
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
from model_registry import (  # noqa: E402
    build_metadata,
    file_sha256,
    models_dir,
    next_version_name,
    save_version_bundle,
    sklearn_params,
    validate_version,
)
from preprocess import ensure_nltk_data, preprocess_series  # noqa: E402


def train_and_evaluate(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 10_000,
    remove_stopwords: bool = True,
    data_path: Path | None = None,
) -> dict:
    ensure_nltk_data()
    raw_path = Path(data_path) if data_path is not None else default_raw_path()
    if data_path is None and not raw_path.is_file():
        download_sms_spam_collection()
    if not raw_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {raw_path}")

    df = load_sms_spam(raw_path)
    sha = file_sha256(raw_path)
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
        "dataset_path": str(raw_path.resolve()),
        "n_rows": int(len(df)),
        "dataset_sha256": sha,
    }


def save_artifacts_flat(bundle: dict, out_dir: Path | None = None) -> None:
    """Write joblibs directly under ``models/`` (legacy layout)."""
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
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Train and print metrics only; do not write model files.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version folder name (e.g. v03). Default: next available v01, v02, ...",
    )
    parser.add_argument(
        "--set-current",
        action="store_true",
        help="After save, write models/current to this version.",
    )
    parser.add_argument(
        "--parent-version",
        type=str,
        default=None,
        help="Recorded in metadata only (documentation of lineage).",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Free-form note stored in metadata.json.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to tab-separated ham/spam file (label<TAB>text per line).",
    )
    parser.add_argument(
        "--legacy-flat",
        action="store_true",
        help="If saving, write to models/*.joblib instead of models/versions/vNN/.",
    )
    args = parser.parse_args()

    bundle = train_and_evaluate(
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
        remove_stopwords=not args.keep_stopwords,
        data_path=args.data_path,
    )

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

    if args.no_save:
        print("\n(--no-save: artifacts not written.)")
        return

    if args.legacy_flat:
        save_artifacts_flat(bundle)
        print(f"\nArtifacts saved (legacy flat layout) under: {models_dir()}")
        return

    version = args.version or next_version_name()
    validate_version(version)

    hyperparameters = {
        "train_test_split": {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "stratify": True,
        },
        "max_features": args.max_features,
        "remove_stopwords": not args.keep_stopwords,
        "tfidf_vectorizer": sklearn_params(bundle["vectorizer"]),
        "multinomial_nb": sklearn_params(bundle["nb"]),
        "logistic_regression": sklearn_params(bundle["lr"]),
    }

    metadata = build_metadata(
        version=version,
        dataset_path=bundle["dataset_path"],
        dataset_sha256=bundle["dataset_sha256"],
        n_rows=bundle["n_rows"],
        hyperparameters=hyperparameters,
        metrics=bundle["metrics"],
        parent_version=args.parent_version,
        notes=args.note,
    )

    dest = save_version_bundle(
        bundle,
        version=version,
        set_current=args.set_current,
        metadata=metadata,
    )
    print(f"\nArtifacts saved under: {dest}")
    if args.set_current:
        print(f"models/current -> {version}")


if __name__ == "__main__":
    main()
