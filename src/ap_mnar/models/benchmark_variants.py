from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalBenchmarkSpec:
    benchmark_type: str
    augment_signal_history: bool


SIGNAL_BENCHMARK_SPECS = (
    SignalBenchmarkSpec("fixed_x_obs", False),
    SignalBenchmarkSpec("augmented_signal_history", True),
)


def get_signal_benchmark_specs(
    include_augmented_signal_history: bool = True,
) -> list[SignalBenchmarkSpec]:
    if include_augmented_signal_history:
        return list(SIGNAL_BENCHMARK_SPECS)
    return [spec for spec in SIGNAL_BENCHMARK_SPECS if not spec.augment_signal_history]


def augment_signal_history_features(
    panel: pd.DataFrame,
    signal: str,
    base_x_obs_columns: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    working = panel.sort_values(["permno", "date"]).copy()
    signal_col = f"{signal}_tier1"
    lagged_signal = working.groupby("permno", sort=False)[signal_col].shift(1)
    lagged_observed = working.groupby("permno", sort=False)[signal_col].shift(1).notna().astype(int)

    history_group = working.groupby("permno", sort=False)
    ever_observed_before = history_group[signal_col].transform(
        lambda s: s.notna().astype(int).shift(1, fill_value=0).cummax()
    ).astype(int)
    missing_before = history_group[signal_col].transform(
        lambda s: s.isna().astype(int).shift(1, fill_value=0).cumsum()
    ).astype(int)

    lagged_fill_value = float(lagged_signal.mean()) if lagged_signal.notna().any() else 0.0
    added_columns = {
        f"{signal}_phase2_lag1_value": lagged_signal.fillna(lagged_fill_value).astype(float),
        f"{signal}_phase2_lag1_observed": lagged_observed.astype(float),
        f"{signal}_phase2_ever_observed_before": ever_observed_before.astype(float),
        f"{signal}_phase2_prior_missing_count": missing_before.astype(float),
        f"{signal}_phase2_before_support": working.get(
            f"{signal}_before_support",
            pd.Series(False, index=working.index),
        ).astype(float),
        f"{signal}_phase2_after_support": working.get(
            f"{signal}_after_support",
            pd.Series(False, index=working.index),
        ).astype(float),
        f"{signal}_phase2_no_support": working.get(
            f"{signal}_no_support",
            pd.Series(False, index=working.index),
        ).astype(float),
        f"{signal}_phase2_prev_observed_flag": history_group[signal_col].transform(
            lambda s: s.notna().astype(int).shift(1, fill_value=0)
        ).astype(float),
    }
    for column, values in added_columns.items():
        working[column] = values.to_numpy(dtype=float)

    augmented_columns = list(base_x_obs_columns) + list(added_columns.keys())
    augmented_columns = list(dict.fromkeys(augmented_columns))
    return working, augmented_columns


def classify_signal_pattern_slice(
    sample: pd.DataFrame,
    observed_indicator_col: str = "observed_indicator",
    missing_indicator_col: str = "missing_indicator",
) -> pd.Series:
    output = pd.Series(index=sample.index, dtype="object")
    ordered = sample.sort_values(["permno", "date"])

    for _, group in ordered.groupby("permno", sort=False):
        observed_mask = group[observed_indicator_col].eq(1)
        if not observed_mask.any():
            labels = np.where(group[missing_indicator_col].eq(1), "always_missing", "observed")
            output.loc[group.index] = labels
            continue

        observed_dates = group.loc[observed_mask, "date"]
        first_observed = observed_dates.min()
        last_observed = observed_dates.max()
        labels = np.full(len(group), "observed", dtype=object)
        missing_mask = group[missing_indicator_col].eq(1).to_numpy()
        date_values = group["date"].to_numpy()
        labels[missing_mask & (date_values < first_observed)] = "start_missing"
        labels[missing_mask & (date_values > last_observed)] = "end_missing"
        labels[missing_mask & (date_values >= first_observed) & (date_values <= last_observed)] = "middle_missing"
        output.loc[group.index] = labels

    return output.reindex(sample.index)
