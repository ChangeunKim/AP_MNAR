from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def normalize_to_month_end(values: pd.Series) -> pd.Series:
    """Normalize mixed monthly timestamps to month-end."""
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.to_period("M").dt.to_timestamp("M")


def load_csv_panel(
    path: str | Path,
    usecols: Iterable[str],
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load a CSV panel with only the requested columns."""
    frame = pd.read_csv(path, usecols=list(usecols), nrows=nrows)
    if "date" in frame.columns:
        frame["date"] = normalize_to_month_end(frame["date"])
    if "permno" in frame.columns:
        frame["permno"] = pd.to_numeric(frame["permno"], errors="coerce").astype("Int64")
    return frame


def load_firm_panel(
    path: str | Path,
    nrows: int | None = None,
    extra_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    columns = ["permno", "date", "ret", "mve", "exchcd"]
    if extra_columns:
        columns.extend(extra_columns)
    frame = load_csv_panel(path, usecols=columns, nrows=nrows)
    for column in [col for col in columns if col not in {"permno", "date"}]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_signal_panel(
    path: str | Path,
    signals: Iterable[str],
    nrows: int | None = None,
) -> pd.DataFrame:
    columns = ["permno", "date", *signals]
    frame = load_csv_panel(path, usecols=columns, nrows=nrows)
    for signal in signals:
        frame[signal] = pd.to_numeric(frame[signal], errors="coerce")
    return frame

