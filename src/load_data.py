"""Load and optionally download the UCI SMS Spam Collection dataset."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

# UCI ML Repository — SMS Spam Collection (zip contains SMSSpamCollection)
UCI_SMS_SPAM_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
)
DEFAULT_DATA_FILENAME = "SMSSpamCollection"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_raw_path() -> Path:
    return project_root() / "data" / "raw" / DEFAULT_DATA_FILENAME


def download_sms_spam_collection(dest_dir: Path | None = None) -> Path:
    """Download the official zip from UCI and extract ``SMSSpamCollection``."""
    dest_dir = dest_dir or (project_root() / "data" / "raw")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_file = dest_dir / DEFAULT_DATA_FILENAME

    with urlopen(UCI_SMS_SPAM_ZIP_URL, timeout=120) as resp:
        raw = resp.read()

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        member = next(
            (n for n in zf.namelist() if n.endswith(DEFAULT_DATA_FILENAME)), None
        )
        if member is None:
            raise FileNotFoundError(
                f"No {DEFAULT_DATA_FILENAME!r} found in UCI zip archive."
            )
        out_file.write_bytes(zf.read(member))

    return out_file


def load_sms_spam(path: Path | None = None) -> pd.DataFrame:
    """
    Load tab-separated file: ``<label>\\t<message>`` per line.
    Returns columns ``label`` (0=ham, 1=spam) and ``text``.
    """
    path = path or default_raw_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run download or place {DEFAULT_DATA_FILENAME} "
            f"under data/raw/. You can call download_sms_spam_collection()."
        )

    # File may use utf-8 or latin-1 in the wild; utf-8 works for official copy.
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label_raw", "text"],
        encoding="utf-8",
        on_bad_lines="skip",
    )
    df["label_raw"] = df["label_raw"].str.strip().str.lower()
    df = df[df["label_raw"].isin(["ham", "spam"])].copy()
    df["label"] = (df["label_raw"] == "spam").astype(int)
    df = df[["label", "text"]].reset_index(drop=True)
    return df
