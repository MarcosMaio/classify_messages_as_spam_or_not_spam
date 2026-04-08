"""Lightweight text batch profiling (length / token counts) for sanity checks."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def profile_texts(texts: Iterable[str]) -> pd.DataFrame:
    """
    Return a one-row summary DataFrame: mean/std/min/max of char length and word count.
    """
    series = pd.Series([t if isinstance(t, str) else "" for t in texts])
    lengths = series.str.len()
    tokens = series.str.split().str.len()
    summary = pd.DataFrame(
        {
            "char_len_mean": [lengths.mean()],
            "char_len_std": [lengths.std()],
            "char_len_min": [lengths.min()],
            "char_len_max": [lengths.max()],
            "token_count_mean": [tokens.mean()],
            "token_count_std": [tokens.std()],
            "token_count_min": [tokens.min()],
            "token_count_max": [tokens.max()],
            "n_messages": [len(series)],
        }
    )
    return summary
