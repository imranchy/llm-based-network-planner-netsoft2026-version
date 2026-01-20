"""io_utils.py

Minimal I/O helpers.
"""

from __future__ import annotations

import os

import pandas as pd


def save_dataframe(df: pd.DataFrame, path: str, *, index: bool = False, encoding: str = "utf-8") -> str:
    """Save a DataFrame to a CSV file and return the absolute path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=index, encoding=encoding)
    return os.path.abspath(path)
