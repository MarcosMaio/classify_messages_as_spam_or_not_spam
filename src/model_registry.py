"""
Versioned storage for trained spam-classifier artifacts under models/versions/vNN/.

Also resolves which directory predict should load: explicit path, --version,
models/current, or legacy flat models/*.joblib.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

VERSION_RE = re.compile(r"^v\d+$")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def models_dir() -> Path:
    return project_root() / "models"


def versions_root() -> Path:
    return models_dir() / "versions"


def current_pointer_path() -> Path:
    return models_dir() / "current"


def validate_version(name: str) -> None:
    if not VERSION_RE.fullmatch(name):
        raise ValueError(f"Invalid version name {name!r}; expected pattern vNN (e.g. v01).")


def version_dir(name: str) -> Path:
    validate_version(name)
    return versions_root() / name


def list_versions() -> list[str]:
    root = versions_root()
    if not root.is_dir():
        return []
    names = []
    for p in root.iterdir():
        if p.is_dir() and VERSION_RE.fullmatch(p.name):
            names.append(p.name)
    return sorted(names, key=lambda s: int(s[1:]))


def next_version_name() -> str:
    versions = list_versions()
    if not versions:
        return "v01"
    nums = [int(v[1:]) for v in versions]
    n = max(nums) + 1
    width = max(2, len(str(max(nums))), len(str(n)))
    return f"v{n:0{width}d}"


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def sklearn_params(obj) -> dict[str, Any]:
    """Subset of get_params() suitable for JSON metadata."""
    raw = obj.get_params(deep=False)
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if callable(v):
            continue
        out[k] = _json_safe(v)
    return out


def build_metadata(
    *,
    version: str,
    dataset_path: str,
    dataset_sha256: str | None,
    n_rows: int,
    hyperparameters: dict[str, Any],
    metrics: dict[str, Any],
    parent_version: str | None,
    notes: str | None,
) -> dict[str, Any]:
    validate_version(version)
    env: dict[str, str] = {"python": _python_version_string()}
    try:
        import importlib.metadata as im

        for name in ("scikit-learn", "numpy", "pandas", "nltk"):
            try:
                env[name] = im.version(name)
            except im.PackageNotFoundError:
                env[name] = "unknown"
    except Exception:
        pass

    return {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": dataset_path,
            "sha256": dataset_sha256,
            "n_rows": n_rows,
        },
        "hyperparameters": _json_safe(hyperparameters),
        "environment": env,
        "metrics": _json_safe(metrics),
        "parent_version": parent_version,
        "notes": notes or "",
    }


def _python_version_string() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def save_version_bundle(
    bundle: dict,
    *,
    version: str,
    set_current: bool = False,
    metadata: dict[str, Any],
) -> Path:
    """
    Write vectorizer, classifiers, preprocess_config, and metadata.json under
    models/versions/<version>/.
    """
    validate_version(version)
    dest = version_dir(version)
    dest.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle["vectorizer"], dest / "tfidf_vectorizer.joblib")
    joblib.dump(bundle["nb"], dest / "multinomial_nb.joblib")
    joblib.dump(bundle["lr"], dest / "logistic_regression.joblib")
    joblib.dump(bundle["preprocess_config"], dest / "preprocess_config.joblib")

    meta_path = dest / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if set_current:
        set_current_version(version)

    return dest


def set_current_version(version: str) -> None:
    validate_version(version)
    d = version_dir(version)
    if not d.is_dir():
        raise FileNotFoundError(f"Version directory does not exist: {d}")
    current_pointer_path().write_text(version + "\n", encoding="utf-8")


def resolve_active_models_dir(
    *,
    explicit: Path | None = None,
    version: str | None = None,
) -> Path:
    """
    Resolution order:
    1. explicit path
    2. version -> models/versions/<version>
    3. models/current -> models/versions/<name>
    4. legacy models/*.joblib at project models/
    """
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"Not a directory: {p}")
        req = (
            "tfidf_vectorizer.joblib",
            "multinomial_nb.joblib",
            "logistic_regression.joblib",
            "preprocess_config.joblib",
        )
        missing = [f for f in req if not (p / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Directory {p} missing artifact(s): {', '.join(missing)}"
            )
        return p

    if version is not None:
        validate_version(version)
        d = version_dir(version)
        if not d.is_dir():
            raise FileNotFoundError(f"No trained version at {d}")
        return d

    cur = current_pointer_path()
    if cur.is_file():
        name = cur.read_text(encoding="utf-8").strip()
        if name and VERSION_RE.fullmatch(name):
            d = version_dir(name)
            if d.is_dir():
                return d

    leg = models_dir()
    if (leg / "tfidf_vectorizer.joblib").is_file():
        return leg

    raise FileNotFoundError(
        "No model artifacts found. Train with src/train.py --save, or set models/current."
    )
