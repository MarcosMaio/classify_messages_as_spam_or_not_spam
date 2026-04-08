from __future__ import annotations

import json
from pathlib import Path

import pytest

import model_registry as mr


def test_validate_version_ok() -> None:
    mr.validate_version("v01")


def test_validate_version_bad() -> None:
    with pytest.raises(ValueError):
        mr.validate_version("01")
    with pytest.raises(ValueError):
        mr.validate_version("V01")


def test_next_version_name_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mr, "versions_root", lambda: tmp_path / "versions")
    assert mr.next_version_name() == "v01"


def test_next_version_name_increment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "versions"
    root.mkdir()
    (root / "v01").mkdir()
    (root / "v02").mkdir()
    monkeypatch.setattr(mr, "versions_root", lambda: root)
    assert mr.next_version_name() == "v03"


def test_save_version_bundle_writes_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    versions = tmp_path / "versions"
    monkeypatch.setattr(mr, "versions_root", lambda: versions)
    monkeypatch.setattr(mr, "project_root", lambda: tmp_path)

    bundle = {
        "vectorizer": {"dummy": 1},
        "nb": {"dummy": 2},
        "lr": {"dummy": 3},
        "preprocess_config": {"remove_stopwords": True},
    }
    meta = {
        "version": "v01",
        "created_at": "2020-01-01T00:00:00+00:00",
        "dataset": {"path": "/x", "sha256": "abc", "n_rows": 10},
        "hyperparameters": {},
        "environment": {},
        "metrics": {},
        "parent_version": None,
        "notes": "",
    }
    dest = mr.save_version_bundle(bundle, version="v01", set_current=False, metadata=meta)
    assert dest == versions / "v01"
    assert (dest / "metadata.json").is_file()
    loaded = json.loads((dest / "metadata.json").read_text(encoding="utf-8"))
    assert loaded["version"] == "v01"


def test_resolve_legacy_flat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mdir = tmp_path / "models"
    mdir.mkdir()
    for name in (
        "tfidf_vectorizer.joblib",
        "multinomial_nb.joblib",
        "logistic_regression.joblib",
        "preprocess_config.joblib",
    ):
        (mdir / name).write_bytes(b"0")

    monkeypatch.setattr(mr, "models_dir", lambda: mdir)
    monkeypatch.setattr(mr, "current_pointer_path", lambda: mdir / "current")
    assert mr.resolve_active_models_dir().resolve() == mdir.resolve()
