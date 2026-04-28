from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ap_mnar.models.mar_benchmark import fit_linear_projection, predict_linear_projection


@dataclass(frozen=True)
class PricingSpecification:
    name: str
    feature_columns: tuple[str, ...]
    focus_columns: tuple[str, ...]


def build_pricing_specifications(x_obs_columns: Sequence[str]) -> list[PricingSpecification]:
    x_obs_tuple = tuple(x_obs_columns)
    return [
        PricingSpecification(
            name="baseline_missing_only",
            feature_columns=(*x_obs_tuple, "missing_indicator"),
            focus_columns=("missing_indicator",),
        ),
        PricingSpecification(
            name="mar_signal_only",
            feature_columns=(*x_obs_tuple, "x_mar_filled"),
            focus_columns=("x_mar_filled",),
        ),
        PricingSpecification(
            name="missing_plus_mar_signal",
            feature_columns=(*x_obs_tuple, "missing_indicator", "x_mar_filled"),
            focus_columns=("missing_indicator", "x_mar_filled"),
        ),
    ]


def build_signal_pricing_panel(
    panel: pd.DataFrame,
    signal: str,
    x_obs_columns: Sequence[str],
    min_train_months: int = 60,
    return_col: str = "ret_fwd_1m",
) -> pd.DataFrame:
    target_signal_col = f"{signal}_tier1"
    eligible_col = f"{signal}_eligible"
    residual_col = f"{signal}_residual_missing"

    required_columns = [
        "permno",
        "date",
        return_col,
        target_signal_col,
        eligible_col,
        residual_col,
        *x_obs_columns,
    ]
    sample = panel[required_columns].copy()
    sample = sample.loc[sample[eligible_col] & sample[return_col].notna()].copy()
    sample = sample.dropna(subset=x_obs_columns).copy()
    sample = sample.sort_values(["date", "permno"]).reset_index(drop=True)

    sample["signal"] = signal
    sample["missing_indicator"] = sample[residual_col].astype(int)
    sample["observed_indicator"] = sample[target_signal_col].notna().astype(int)
    sample["signal_has_ever_observed"] = (
        sample.groupby("permno")["observed_indicator"].transform("max").fillna(0).astype(int)
    )
    sample["within_signal_coverage_flag"] = sample["signal_has_ever_observed"].eq(1)
    sample["no_signal_coverage_flag"] = sample["signal_has_ever_observed"].eq(0)
    sample["x_mar_cond_mean"] = np.nan
    sample["x_mar_filled"] = np.nan
    sample["train_observed_count"] = np.nan

    unique_dates = sorted(sample["date"].dropna().unique())
    if len(unique_dates) <= min_train_months:
        raise ValueError(
            f"Signal {signal} does not have enough months ({len(unique_dates)}) "
            f"for min_train_months={min_train_months}."
        )

    evaluation_dates = unique_dates[min_train_months:]
    for current_date in evaluation_dates:
        current_mask = sample["date"].eq(current_date)
        current_rows = sample.loc[current_mask].copy()
        if current_rows.empty:
            continue

        observed_train = sample.loc[
            sample["date"].lt(current_date) & sample["observed_indicator"].eq(1)
        ].copy()
        if len(observed_train) <= len(x_obs_columns) + 1:
            continue

        projection_bundle = fit_linear_projection(observed_train, target_signal_col, x_obs_columns)
        predicted = predict_linear_projection(projection_bundle, current_rows)

        sample.loc[current_mask, "x_mar_cond_mean"] = predicted.to_numpy(dtype=float)
        sample.loc[current_mask, "x_mar_filled"] = current_rows[target_signal_col].where(
            current_rows["observed_indicator"].eq(1),
            predicted,
        )
        sample.loc[current_mask, "train_observed_count"] = float(len(observed_train))

    sample["pricing_sample_flag"] = sample["x_mar_cond_mean"].notna()
    sample = sample.loc[sample["pricing_sample_flag"]].copy()
    sample["pricing_month"] = sample["date"]
    sample["return_col"] = return_col
    return sample


def build_coverage_subsamples(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    subsamples = {
        "full": panel.copy(),
        "within_signal_coverage": panel.loc[panel["within_signal_coverage_flag"]].copy(),
        "outside_signal_coverage": panel.loc[panel["no_signal_coverage_flag"]].copy(),
    }
    for sample_name, sample_frame in subsamples.items():
        sample_frame.attrs["sample_name"] = sample_name
    return subsamples
