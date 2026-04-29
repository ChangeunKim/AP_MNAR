from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2


def build_signal_instrument_frame(
    sample: pd.DataFrame,
    x_obs_columns: Sequence[str],
    conditional_mean_col: str = "x_cond_mean_cf",
) -> pd.DataFrame:
    instruments = pd.DataFrame(index=sample.index)
    instruments["inst_const"] = 1.0

    for column in x_obs_columns:
        if column not in sample.columns:
            continue
        if sample[column].nunique(dropna=True) <= 1:
            continue
        instruments[f"inst_{column}"] = sample[column].astype(float)

    if conditional_mean_col in sample.columns and sample[conditional_mean_col].nunique(dropna=True) > 1:
        instruments[f"{conditional_mean_col}_sq"] = sample[conditional_mean_col].astype(float) ** 2

    return instruments


def run_signal_mar_jtest(
    sample: pd.DataFrame,
    instrument_frame: pd.DataFrame,
    signal_col: str,
    conditional_mean_col: str,
    mar_draw_col: str,
    observed_indicator_col: str = "observed_indicator",
    missing_indicator_col: str = "missing_indicator",
    observed_variance_col: str | None = None,
    missing_variance_col: str | None = None,
) -> dict[str, float | int]:
    observed_indicator = sample[observed_indicator_col].astype(float).to_numpy()
    missing_indicator = sample[missing_indicator_col].astype(float).to_numpy()
    conditional_mean = sample[conditional_mean_col].to_numpy(dtype=float)

    observed_signal = sample[signal_col].to_numpy(dtype=float)
    mar_draw_signal = sample[mar_draw_col].to_numpy(dtype=float)

    observed_residual = np.where(
        observed_indicator > 0,
        observed_signal - conditional_mean,
        0.0,
    )
    missing_residual = np.where(
        missing_indicator > 0,
        mar_draw_signal - conditional_mean,
        0.0,
    )
    observed_residual = np.nan_to_num(observed_residual, nan=0.0)
    missing_residual = np.nan_to_num(missing_residual, nan=0.0)
    observed_precision = _build_precision_weights(sample, observed_variance_col)
    missing_precision = _build_precision_weights(sample, missing_variance_col)
    observed_residual = observed_residual * observed_precision
    missing_residual = missing_residual * missing_precision

    instrument_columns = []
    moment_parts: list[np.ndarray] = []

    for column in instrument_frame.columns:
        instrument = instrument_frame[column].to_numpy(dtype=float)

        obs_moment = observed_residual * instrument
        if np.nanstd(obs_moment) > 0:
            instrument_columns.append(f"obs::{column}")
            moment_parts.append(obs_moment)

        missing_moment = missing_residual * instrument
        if np.nanstd(missing_moment) > 0:
            instrument_columns.append(f"mis::{column}")
            moment_parts.append(missing_moment)

    if not moment_parts:
        return {
            "j_stat": np.nan,
            "j_p_value": np.nan,
            "j_df": 0,
            "j_moment_count": 0,
            "instrument_count": 0,
        }

    moment_matrix = np.column_stack(moment_parts).astype(float, copy=False)
    return {
        **compute_gmm_jstat(moment_matrix, moment_labels=instrument_columns),
        "j_moment_count": len(instrument_columns),
        "instrument_count": int(len(instrument_frame.columns)),
    }


def _build_precision_weights(
    sample: pd.DataFrame,
    variance_col: str | None,
) -> np.ndarray:
    if variance_col is None or variance_col not in sample.columns:
        return np.ones(len(sample), dtype=float)
    variance = sample[variance_col].to_numpy(dtype=float)
    variance = np.nan_to_num(variance, nan=np.nanmedian(variance) if np.isfinite(np.nanmedian(variance)) else 1.0)
    variance = np.clip(variance, 1e-8, None)
    return 1.0 / np.sqrt(variance)


def compute_gmm_jstat(
    moment_matrix: np.ndarray,
    moment_labels: Sequence[str] | None = None,
) -> dict[str, float | int | list[float] | list[str]]:
    if moment_matrix.ndim != 2 or moment_matrix.shape[0] == 0 or moment_matrix.shape[1] == 0:
        return {
            "j_stat": np.nan,
            "j_p_value": np.nan,
            "j_df": 0,
            "n_obs": 0,
        }

    n_obs = int(moment_matrix.shape[0])
    gbar = moment_matrix.mean(axis=0)
    centered = moment_matrix - gbar
    covariance = centered.T @ centered / max(n_obs - 1, 1)

    covariance_inv = np.linalg.pinv(covariance)
    j_stat = float(n_obs * gbar.T @ covariance_inv @ gbar)
    j_df = int(np.linalg.matrix_rank(covariance))
    j_p_value = float(1.0 - chi2.cdf(j_stat, j_df)) if j_df > 0 else np.nan
    covariance_diag = np.clip(np.diag(covariance).astype(float, copy=False), 1e-12, None)
    standardized_moments = (np.sqrt(n_obs) * gbar / np.sqrt(covariance_diag)).astype(float, copy=False)
    weighted_moments = covariance_inv @ gbar
    j_contributions = (n_obs * gbar * weighted_moments).astype(float, copy=False)
    labels = list(moment_labels) if moment_labels is not None else [f"moment_{i}" for i in range(len(gbar))]
    observed_j_component = float(sum(contrib for label, contrib in zip(labels, j_contributions) if str(label).startswith("obs::")))
    missing_j_component = float(sum(contrib for label, contrib in zip(labels, j_contributions) if str(label).startswith("mis::")))

    return {
        "j_stat": j_stat,
        "j_p_value": j_p_value,
        "j_df": j_df,
        "n_obs": n_obs,
        "moment_labels": labels,
        "moment_means": gbar.astype(float, copy=False).tolist(),
        "standardized_moments": standardized_moments.tolist(),
        "j_contributions": j_contributions.tolist(),
        "observed_moment_norm": float(np.linalg.norm([x for label, x in zip(labels, gbar) if str(label).startswith("obs::")])),
        "missing_moment_norm": float(np.linalg.norm([x for label, x in zip(labels, gbar) if str(label).startswith("mis::")])),
        "stacked_moment_norm": float(np.linalg.norm(gbar)),
        "observed_j_component": observed_j_component,
        "missing_j_component": missing_j_component,
    }
