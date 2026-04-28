from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


COUNTERFACTUAL_REGIMES = (
    "complete_case",
    "unconditional_mean",
    "conditional_mean",
    "residual_bootstrap",
    "conditional_quantile_draw",
)


@dataclass
class FastOLSBundle:
    coefficients: np.ndarray
    design_columns: list[str]


@dataclass
class CounterfactualImputationBundle:
    signal_col: str
    ols_bundle: FastOLSBundle
    unconditional_mean: float
    residuals: np.ndarray
    observed_signal_values: np.ndarray
    conditional_quantile_edges: np.ndarray
    conditional_quantile_signal_values: dict[int, np.ndarray]


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


def fit_fast_ols(
    frame: pd.DataFrame,
    target_col: str,
    feature_columns: Sequence[str],
) -> FastOLSBundle:
    design_columns = _clean_feature_columns(frame, feature_columns)
    x_matrix = _design_matrix(frame, design_columns)
    y_vector = frame[target_col].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(x_matrix, y_vector, rcond=None)
    return FastOLSBundle(coefficients=coefficients, design_columns=design_columns)


def predict_fast_ols(
    bundle: FastOLSBundle,
    frame: pd.DataFrame,
) -> np.ndarray:
    x_matrix = _design_matrix(frame, bundle.design_columns)
    return x_matrix @ bundle.coefficients


def build_counterfactual_columns(
    frame: pd.DataFrame,
    signal: str,
    x_obs_columns: Sequence[str],
    imputation_bundle: FastOLSBundle,
    unconditional_mean: float,
) -> pd.DataFrame:
    working = frame.copy()
    signal_col = f"{signal}_tier1"
    observed_mask = working["observed_indicator"].eq(1)
    conditional_mean = predict_fast_ols(imputation_bundle, working)

    working["x_complete_case"] = working[signal_col]
    working["x_unconditional"] = working[signal_col].where(observed_mask, unconditional_mean)
    working["x_conditional"] = working[signal_col].where(observed_mask, conditional_mean)
    return working


def fit_counterfactual_imputation_bundle(
    observed_train: pd.DataFrame,
    signal_col: str,
    x_obs_columns: Sequence[str],
    quantile_bins: int = 5,
) -> CounterfactualImputationBundle:
    ols_bundle = fit_fast_ols(observed_train, signal_col, x_obs_columns)
    conditional_mean = predict_fast_ols(ols_bundle, observed_train)
    signal_values = observed_train[signal_col].to_numpy(dtype=float)
    residuals = signal_values - conditional_mean

    quantile_edges = _build_quantile_edges(conditional_mean, quantile_bins)
    quantile_groups = _assign_quantile_groups(conditional_mean, quantile_edges)
    conditional_quantile_signal_values: dict[int, np.ndarray] = {}
    for group_id in np.unique(quantile_groups):
        conditional_quantile_signal_values[int(group_id)] = signal_values[quantile_groups == group_id]

    return CounterfactualImputationBundle(
        signal_col=signal_col,
        ols_bundle=ols_bundle,
        unconditional_mean=float(signal_values.mean()),
        residuals=residuals,
        observed_signal_values=signal_values,
        conditional_quantile_edges=quantile_edges,
        conditional_quantile_signal_values=conditional_quantile_signal_values,
    )


def build_stochastic_counterfactual_column(
    frame: pd.DataFrame,
    signal: str,
    bundle: CounterfactualImputationBundle,
    regime: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if regime not in {"residual_bootstrap", "conditional_quantile_draw"}:
        raise ValueError(f"Unsupported stochastic counterfactual regime: {regime}")

    working = frame.copy()
    signal_col = f"{signal}_tier1"
    observed_mask = working["observed_indicator"].eq(1)
    conditional_mean = predict_fast_ols(bundle.ols_bundle, working)

    if regime == "residual_bootstrap":
        sampled_residuals = _sample_from_array(bundle.residuals, int((~observed_mask).sum()), rng)
        sampled_values = conditional_mean[~observed_mask.to_numpy()] + sampled_residuals
        working["x_residual_bootstrap"] = working[signal_col]
        working.loc[~observed_mask, "x_residual_bootstrap"] = sampled_values
        return working

    group_ids = _assign_quantile_groups(conditional_mean, bundle.conditional_quantile_edges)
    sampled = np.empty(int((~observed_mask).sum()), dtype=float)
    missing_positions = np.flatnonzero(~observed_mask.to_numpy())
    for idx, row_pos in enumerate(missing_positions):
        group_id = int(group_ids[row_pos])
        group_values = bundle.conditional_quantile_signal_values.get(group_id)
        if group_values is None or len(group_values) == 0:
            group_values = bundle.observed_signal_values
        sampled[idx] = _sample_from_array(group_values, 1, rng)[0]

    working["x_conditional_quantile_draw"] = working[signal_col]
    working.loc[~observed_mask, "x_conditional_quantile_draw"] = sampled
    return working


def get_regime_frame(
    frame: pd.DataFrame,
    signal: str,
    regime: str,
) -> tuple[pd.DataFrame, str]:
    signal_col = f"{signal}_tier1"
    if regime == "complete_case":
        subset = frame.loc[frame["observed_indicator"].eq(1)].copy()
        return subset, signal_col
    if regime == "unconditional_mean":
        return frame.copy(), "x_unconditional"
    if regime == "conditional_mean":
        return frame.copy(), "x_conditional"
    if regime == "residual_bootstrap":
        return frame.copy(), "x_residual_bootstrap"
    if regime == "conditional_quantile_draw":
        return frame.copy(), "x_conditional_quantile_draw"
    raise ValueError(f"Unsupported counterfactual regime: {regime}")


def _design_matrix(
    frame: pd.DataFrame,
    design_columns: Sequence[str],
) -> np.ndarray:
    if design_columns:
        x_features = frame[list(design_columns)].to_numpy(dtype=float)
        intercept = np.ones((len(frame), 1), dtype=float)
        return np.hstack([intercept, x_features])
    return np.ones((len(frame), 1), dtype=float)


def _build_quantile_edges(values: np.ndarray, quantile_bins: int) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    requested_bins = max(2, int(quantile_bins))
    quantiles = np.linspace(0.0, 1.0, requested_bins + 1)
    edges = np.quantile(clean, quantiles)
    unique_edges = np.unique(edges)
    if len(unique_edges) < 2:
        center = float(clean.mean()) if len(clean) else 0.0
        unique_edges = np.array([center - 1e-9, center + 1e-9], dtype=float)
    return unique_edges


def _assign_quantile_groups(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    group_ids = np.digitize(clean, edges[1:-1], right=True)
    return group_ids.astype(int)


def _sample_from_array(
    values: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    if len(clean) == 0:
        raise ValueError("Cannot sample from an empty array.")
    draw_idx = rng.integers(0, len(clean), size=size)
    return clean[draw_idx]
