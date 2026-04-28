from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class OLSResultBundle:
    model: sm.regression.linear_model.RegressionResultsWrapper
    design_columns: list[str]


def _clean_feature_columns(frame: pd.DataFrame, feature_columns: Sequence[str]) -> list[str]:
    clean_columns = []
    for column in feature_columns:
        if column not in frame.columns:
            continue
        if frame[column].notna().sum() == 0:
            continue
        if frame[column].nunique(dropna=True) <= 1:
            continue
        clean_columns.append(column)
    return clean_columns


def fit_linear_projection(
    frame: pd.DataFrame,
    target_col: str,
    feature_columns: Sequence[str],
) -> OLSResultBundle:
    design_columns = _clean_feature_columns(frame, feature_columns)
    design = sm.add_constant(frame[design_columns], has_constant="add")
    model = sm.OLS(frame[target_col], design).fit()
    return OLSResultBundle(model=model, design_columns=list(design_columns))


def predict_linear_projection(
    bundle: OLSResultBundle,
    frame: pd.DataFrame,
) -> pd.Series:
    design = sm.add_constant(frame[bundle.design_columns], has_constant="add")
    return pd.Series(bundle.model.predict(design), index=frame.index)


def build_signal_mar_panel(
    panel: pd.DataFrame,
    signal: str,
    x_obs_columns: Sequence[str],
    return_col: str = "ret_fwd_1m",
) -> tuple[pd.DataFrame, OLSResultBundle, float]:
    target_signal_col = f"{signal}_tier1"
    eligible_col = f"{signal}_eligible"
    residual_col = f"{signal}_residual_missing"

    required_columns = ["permno", "date", return_col, target_signal_col, eligible_col, residual_col, *x_obs_columns]
    sample = panel[required_columns].copy()
    sample = sample.loc[sample[eligible_col] & sample[return_col].notna()].copy()
    sample = sample.dropna(subset=x_obs_columns).copy()
    sample["missing_indicator"] = sample[residual_col].astype(int)
    sample["observed_indicator"] = sample[target_signal_col].notna().astype(int)

    observed_sample = sample.loc[sample["observed_indicator"].eq(1)].copy()
    if observed_sample.empty:
        print(f"DEBUG: len(sample)={len(sample)}, len(panel)={len(panel)}")
        print(f"DEBUG: eligible sum = {panel[eligible_col].sum()}")
        print(f"DEBUG: return notna sum = {panel[return_col].notna().sum()}")
        print(f"DEBUG: target signal notna sum = {panel[target_signal_col].notna().sum()}")
        raise ValueError(f"No observed eligible rows available for signal {signal}.")

    imputation_bundle = fit_linear_projection(observed_sample, target_signal_col, x_obs_columns)
    sample["x_cond_mean"] = predict_linear_projection(imputation_bundle, sample)
    unconditional_mean = float(observed_sample[target_signal_col].mean())
    sample["x_uncond_mean"] = unconditional_mean
    sample["x_complete_case"] = sample[target_signal_col]
    sample["x_conditional"] = sample[target_signal_col].where(sample["observed_indicator"].eq(1), sample["x_cond_mean"])
    sample["x_unconditional"] = sample[target_signal_col].where(sample["observed_indicator"].eq(1), unconditional_mean)

    return sample, imputation_bundle, unconditional_mean


def fit_return_model(
    frame: pd.DataFrame,
    signal_column: str,
    x_obs_columns: Sequence[str],
    return_col: str = "ret_fwd_1m",
) -> OLSResultBundle:
    required = frame.dropna(subset=[return_col, signal_column, *x_obs_columns]).copy()
    if required.empty:
        raise ValueError(f"No rows available to fit return model using {signal_column}.")
    features = [*x_obs_columns, signal_column]
    return fit_linear_projection(required, return_col, features)


def build_benchmark_comparison_rows(
    signal: str,
    sample: pd.DataFrame,
    x_obs_columns: Sequence[str],
    return_col: str = "ret_fwd_1m",
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    benchmark_map = {
        "complete_case": sample.loc[sample["observed_indicator"].eq(1)].copy(),
        "unconditional_mean": sample.copy(),
        "conditional_mean": sample.copy(),
    }
    signal_map = {
        "complete_case": f"{signal}_tier1",
        "unconditional_mean": "x_unconditional",
        "conditional_mean": "x_conditional",
    }

    for benchmark_name, frame in benchmark_map.items():
        signal_column = signal_map[benchmark_name]
        try:
            bundle = fit_return_model(frame, signal_column, x_obs_columns, return_col=return_col)
        except ValueError:
            rows.append(
                {
                    "signal": signal,
                    "benchmark": benchmark_name,
                    "n_obs": 0,
                    "r_squared": np.nan,
                    "signal_coef": np.nan,
                    "signal_pvalue": np.nan,
                }
            )
            continue

        coef_name = signal_column
        rows.append(
            {
                "signal": signal,
                "benchmark": benchmark_name,
                "n_obs": int(bundle.model.nobs),
                "r_squared": float(bundle.model.rsquared),
                "signal_coef": float(bundle.model.params.get(coef_name, np.nan)),
                "signal_pvalue": float(bundle.model.pvalues.get(coef_name, np.nan)),
            }
        )

    return rows
