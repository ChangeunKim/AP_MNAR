from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

from ap_mnar.data.load_raw import load_csv_panel


@dataclass(frozen=True)
class XObsSpec:
    columns: tuple[str, ...]


def load_x_obs_spec(path: str | Path) -> XObsSpec:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    columns = tuple(payload.get("columns", []))
    return XObsSpec(columns=columns)


def merge_x_obs_columns(
    panel: pd.DataFrame,
    firm_panel_path: str | Path,
    x_obs_columns: Sequence[str],
) -> pd.DataFrame:
    missing_columns = [column for column in x_obs_columns if column not in panel.columns]
    if not missing_columns:
        return panel.copy()

    supplement = load_csv_panel(firm_panel_path, usecols=["permno", "date", *missing_columns])
    for column in missing_columns:
        supplement[column] = pd.to_numeric(supplement[column], errors="coerce")

    merged = panel.merge(supplement, on=["permno", "date"], how="left")
    return merged


def build_x_obs_availability_table(
    panel: pd.DataFrame,
    x_obs_columns: Sequence[str],
) -> pd.DataFrame:
    total_count = len(panel)
    rows = []
    for column in x_obs_columns:
        non_missing_count = int(panel[column].notna().sum())
        rows.append(
            {
                "column": column,
                "non_missing_count": non_missing_count,
                "total_count": total_count,
                "availability_rate": non_missing_count / total_count if total_count else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("column", ignore_index=True)
