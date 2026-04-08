"""Evaluate saved artifacts on a small labeled TSV (regression / smoke test)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from load_data import load_sms_spam, project_root  # noqa: E402
from model_registry import resolve_active_models_dir  # noqa: E402
from preprocess import preprocess_text  # noqa: E402


def default_golden_path() -> Path:
    return project_root() / "data" / "golden_messages.tsv"


def evaluate_on_golden(
    models_path: Path,
    golden_path: Path,
    *,
    model_name: str = "lr",
) -> dict:
    df = load_sms_spam(golden_path)
    vectorizer = joblib.load(models_path / "tfidf_vectorizer.joblib")
    nb = joblib.load(models_path / "multinomial_nb.joblib")
    lr = joblib.load(models_path / "logistic_regression.joblib")
    cfg = joblib.load(models_path / "preprocess_config.joblib")
    remove_sw = cfg.get("remove_stopwords", True)

    clf = nb if model_name == "nb" else lr

    y_true = df["label"].values
    y_pred: list[int] = []
    for text in df["text"]:
        clean = preprocess_text(str(text), remove_stopwords=remove_sw)
        X = vectorizer.transform([clean])
        y_pred.append(int(clf.predict(X)[0]))

    y_pred_arr = np.array(y_pred)
    report = classification_report(y_true, y_pred_arr, target_names=["ham", "spam"])
    acc = float(accuracy_score(y_true, y_pred_arr))
    spam_f1 = float(f1_score(y_true, y_pred_arr, pos_label=1, zero_division=0))
    rep_dict = classification_report(
        y_true, y_pred_arr, target_names=["ham", "spam"], output_dict=True
    )
    spam_recall = float(rep_dict["spam"]["recall"])
    return {
        "accuracy": acc,
        "spam_f1": spam_f1,
        "spam_recall": spam_recall,
        "report": report,
        "n_samples": len(df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model artifacts on golden_messages.tsv (or custom TSV)."
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=None,
        help=f"Path to labeled TSV (default: {default_golden_path()})",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Explicit artifact directory.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Artifact version directory under models/versions/.",
    )
    parser.add_argument(
        "--model",
        choices=("lr", "nb"),
        default="lr",
        help="Classifier to evaluate.",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="Exit with code 1 if accuracy is below this value.",
    )
    parser.add_argument(
        "--min-spam-recall",
        type=float,
        default=None,
        help="Exit with code 1 if spam recall is below this value.",
    )
    args = parser.parse_args()

    golden = args.golden or default_golden_path()
    if not golden.is_file():
        print(f"Golden file not found: {golden}", file=sys.stderr)
        sys.exit(2)

    root = resolve_active_models_dir(explicit=args.models_dir, version=args.version)

    m = evaluate_on_golden(root, golden, model_name=args.model)
    print(f"Artifacts: {root}")
    print(f"Golden:    {golden}")
    print(f"Model:     {args.model}")
    print(f"Samples:   {m['n_samples']}")
    print(f"Accuracy:       {m['accuracy']:.4f}")
    print(f"Spam recall:    {m['spam_recall']:.4f}")
    print(f"Spam F1:        {m['spam_f1']:.4f}")
    print(m["report"])

    fail = False
    if args.min_accuracy is not None and m["accuracy"] < args.min_accuracy:
        print(
            f"FAIL: accuracy {m['accuracy']:.4f} < {args.min_accuracy}",
            file=sys.stderr,
        )
        fail = True
    if args.min_spam_recall is not None and m["spam_recall"] < args.min_spam_recall:
        print(
            f"FAIL: spam_recall {m['spam_recall']:.4f} < {args.min_spam_recall}",
            file=sys.stderr,
        )
        fail = True
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
