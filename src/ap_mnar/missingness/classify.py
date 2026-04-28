from __future__ import annotations

from typing import Sequence

import pandas as pd


def build_missingness_panel(
    panel: pd.DataFrame,
    eligibility_matrix: pd.DataFrame,
    signals: Sequence[str],
) -> pd.DataFrame:
    result = panel.copy()

    for column in eligibility_matrix.columns:
        result[column] = eligibility_matrix[column].values

    residual_columns: list[str] = []
    eligible_columns: list[str] = []
    observed_columns: list[str] = []

    for signal in signals:
        raw_col = signal
        tier1_col = f"{signal}_tier1"
        eligible_col = f"{signal}_eligible"
        observed_col = f"{signal}_observed_eligible"
        residual_col = f"{signal}_residual_missing"

        result[f"{signal}_raw_missing"] = result[raw_col].isna()
        result[f"{signal}_tier1_missing"] = result[tier1_col].isna()
        result[observed_col] = result[eligible_col] & result[tier1_col].notna()
        result[residual_col] = result[eligible_col] & result[f"{signal}_tier1_missing"]

        eligible_columns.append(eligible_col)
        observed_columns.append(observed_col)
        residual_columns.append(residual_col)

    result["eligible_signal_count"] = result[eligible_columns].sum(axis=1).astype(int)
    result["observed_signal_count"] = result[observed_columns].sum(axis=1).astype(int)
    result["residual_missing_count"] = result[residual_columns].sum(axis=1).astype(int)
    result["has_any_eligible_signal"] = result["eligible_signal_count"] > 0
    result["complete_case_flag"] = result["has_any_eligible_signal"] & result["residual_missing_count"].eq(0)
    result["missing_pattern_code"] = result[residual_columns].astype(int).astype(str).agg("".join, axis=1)
    result["missing_pattern_label"] = result.apply(lambda row: _pattern_label(row, signals), axis=1)

    return result


def _pattern_label(row: pd.Series, signals: Sequence[str]) -> str:
    if int(row["eligible_signal_count"]) == 0:
        return "NO_ELIGIBLE_SIGNAL"

    missing_signals = [signal for signal in signals if bool(row[f"{signal}_residual_missing"])]
    if not missing_signals:
        return "COMPLETE"
    return "|".join(missing_signals)
