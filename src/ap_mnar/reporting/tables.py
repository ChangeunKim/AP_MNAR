from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


def build_signal_coverage_table(
    panel: pd.DataFrame,
    signals: Sequence[str],
) -> pd.DataFrame:
    group_specs = {
        "overall": [],
        "year": ["year"],
        "exchange": ["exchcd"],
        "size_bucket": ["size_bucket"],
    }
    rows: list[dict[str, object]] = []

    for signal in signals:
        for basis, column in [("raw", signal), ("tier1", f"{signal}_tier1")]:
            for dimension, group_cols in group_specs.items():
                if group_cols:
                    grouped = panel.groupby(group_cols, dropna=False)
                    for group_key, subset in grouped:
                        if not isinstance(group_key, tuple):
                            group_key = (group_key,)
                        group_label = "|".join("NA" if pd.isna(value) else str(value) for value in group_key)
                        rows.append(_coverage_row(signal, basis, dimension, group_label, subset[column]))
                else:
                    rows.append(_coverage_row(signal, basis, dimension, "all", panel[column]))

    return pd.DataFrame(rows).sort_values(
        ["signal", "coverage_basis", "dimension", "group_value"],
        ignore_index=True,
    )


def _coverage_row(
    signal: str,
    basis: str,
    dimension: str,
    group_value: str,
    values: pd.Series,
) -> dict[str, object]:
    total_count = int(len(values))
    non_missing_count = int(values.notna().sum())
    coverage_rate = non_missing_count / total_count if total_count else 0.0
    return {
        "signal": signal,
        "coverage_basis": basis,
        "dimension": dimension,
        "group_value": group_value,
        "total_count": total_count,
        "non_missing_count": non_missing_count,
        "coverage_rate": coverage_rate,
    }


def build_missingness_before_after_table(
    panel: pd.DataFrame,
    signals: Iterable[str],
) -> pd.DataFrame:
    rows = []
    total_count = len(panel)
    for signal in signals:
        raw_missing = int(panel[signal].isna().sum())
        tier1_missing = int(panel[f"{signal}_tier1"].isna().sum())
        rows.append(
            {
                "signal": signal,
                "total_count": total_count,
                "raw_missing_count": raw_missing,
                "tier1_missing_count": tier1_missing,
                "missing_removed_by_tier1": raw_missing - tier1_missing,
                "raw_missing_rate": raw_missing / total_count if total_count else 0.0,
                "tier1_missing_rate": tier1_missing / total_count if total_count else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("signal", ignore_index=True)


def build_eligibility_filter_impact_table(
    panel: pd.DataFrame,
    signals: Iterable[str],
) -> pd.DataFrame:
    rows = []
    total_count = len(panel)
    for signal in signals:
        raw_available = int(panel[signal].notna().sum())
        tier1_available = int(panel[f"{signal}_tier1"].notna().sum())
        rows.append(
            {
                "signal": signal,
                "total_count": total_count,
                "raw_available_count": raw_available,
                "tier1_available_count": tier1_available,
                "availability_shift_count": tier1_available - raw_available,
                "raw_available_rate": raw_available / total_count if total_count else 0.0,
                "tier1_available_rate": tier1_available / total_count if total_count else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("signal", ignore_index=True)


def build_x_obs_availability_audit(
    panel: pd.DataFrame,
    x_obs_columns: Sequence[str],
) -> pd.DataFrame:
    if not x_obs_columns:
        return pd.DataFrame(
            [
                {
                    "column": "NOT_CONFIGURED",
                    "non_missing_count": 0,
                    "total_count": len(panel),
                    "availability_rate": 0.0,
                    "note": "Phase 0 scaffold created before the fixed Freyberger-style X_obs list was mirrored into code.",
                }
            ]
        )

    rows = []
    total_count = len(panel)
    for column in x_obs_columns:
        rows.append(
            {
                "column": column,
                "non_missing_count": int(panel[column].notna().sum()),
                "total_count": total_count,
                "availability_rate": int(panel[column].notna().sum()) / total_count if total_count else 0.0,
                "note": "",
            }
        )
    return pd.DataFrame(rows).sort_values("column", ignore_index=True)


def build_summary_statistics_table(
    panel: pd.DataFrame,
    signals: Sequence[str],
) -> pd.DataFrame:
    """Build a summary statistics table containing mean, std, min, percentiles, max."""
    cols_to_summarize = []
    
    # Target return
    if "ret_fwd_1m" in panel:
        cols_to_summarize.append("ret_fwd_1m")
        
    # Raw and Tier 1 signals
    for sig in signals:
        if sig in panel:
            cols_to_summarize.append(sig)
        tier1_col = f"{sig}_tier1"
        if tier1_col in panel:
            cols_to_summarize.append(tier1_col)
            
    stats = panel[cols_to_summarize].describe().T
    stats.reset_index(inplace=True)
    stats.rename(columns={"index": "variable", "count": "non_missing_count"}, inplace=True)
    return stats
