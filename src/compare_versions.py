"""Compare two versioned artifact folders on the same golden TSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evaluate_golden import default_golden_path, evaluate_on_golden  # noqa: E402
from model_registry import validate_version, version_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two trained versions on the same golden set."
    )
    parser.add_argument("version_a", type=str, help="First version (e.g. v01).")
    parser.add_argument("version_b", type=str, help="Second version (e.g. v02).")
    parser.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="Labeled TSV path (default: data/golden_messages.tsv).",
    )
    parser.add_argument(
        "--model",
        choices=("lr", "nb"),
        default="lr",
        help="Classifier to compare.",
    )
    args = parser.parse_args()

    for v in (args.version_a, args.version_b):
        validate_version(v)
        d = version_dir(v)
        if not d.is_dir():
            print(f"Missing version directory: {d}", file=sys.stderr)
            sys.exit(2)

    golden = args.golden or default_golden_path()
    if not golden.is_file():
        print(f"Golden file not found: {golden}", file=sys.stderr)
        sys.exit(2)

    m_a = evaluate_on_golden(
        version_dir(args.version_a), golden, model_name=args.model
    )
    m_b = evaluate_on_golden(
        version_dir(args.version_b), golden, model_name=args.model
    )

    print(f"Golden: {golden}  |  classifier: {args.model}\n")
    print(f"{'metric':<16} {args.version_a:<12} {args.version_b:<12}")
    print("-" * 42)
    for key in ("accuracy", "spam_recall", "spam_f1"):
        print(f"{key:<16} {m_a[key]:<12.4f} {m_b[key]:<12.4f}")


if __name__ == "__main__":
    main()
