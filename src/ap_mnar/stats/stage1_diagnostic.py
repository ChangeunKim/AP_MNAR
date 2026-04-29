from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import f, t


def run_stage1_regression_diagnostic(
    sample: pd.DataFrame,
    x_obs_columns: Sequence[str],
    outcome_col: str,
) -> dict[str, float | int]:
    working = sample.dropna(subset=[outcome_col, *x_obs_columns]).copy()
    working["missing_indicator"] = working["missing_indicator"].astype(float).to_numpy()

    clean_x_obs = []
    design_parts = [np.ones((len(working), 1), dtype=float)]
    for column in x_obs_columns:
        if working[column].nunique(dropna=True) <= 1:
            continue
        clean_x_obs.append(column)

    for column in clean_x_obs:
        design_parts.append(working[[column]].to_numpy(dtype=float))
    design_restricted = np.hstack(design_parts)

    design_unrestricted_parts = list(design_parts)
    design_unrestricted_parts.append(working[["missing_indicator"]].to_numpy(dtype=float))
    for column in clean_x_obs:
        interaction = (working["missing_indicator"] * working[column]).to_numpy(dtype=float).reshape(-1, 1)
        design_unrestricted_parts.append(interaction)
    design_unrestricted = np.hstack(design_unrestricted_parts)

    y_vector = working[outcome_col].to_numpy(dtype=float)

    beta_restricted, *_ = np.linalg.lstsq(design_restricted, y_vector, rcond=None)
    beta_unrestricted, *_ = np.linalg.lstsq(design_unrestricted, y_vector, rcond=None)

    fitted_restricted = design_restricted @ beta_restricted
    fitted_unrestricted = design_unrestricted @ beta_unrestricted
    residual_restricted = y_vector - fitted_restricted
    residual_unrestricted = y_vector - fitted_unrestricted

    restricted_rss = float(residual_restricted.T @ residual_restricted)
    unrestricted_rss = float(residual_unrestricted.T @ residual_unrestricted)

    n_obs = int(len(working))
    unrestricted_param_count = int(design_unrestricted.shape[1])
    tested_restriction_count = 1 + len(clean_x_obs)
    residual_df = n_obs - unrestricted_param_count

    if residual_df <= 0 or tested_restriction_count <= 0:
        f_stat = np.nan
        p_value = np.nan
        aux_r_squared = np.nan
        missing_coef = np.nan
        missing_pvalue = np.nan
    else:
        rss_gap = max(restricted_rss - unrestricted_rss, 0.0)
        numerator = rss_gap / tested_restriction_count
        denominator = unrestricted_rss / residual_df if unrestricted_rss >= 0 else np.nan
        f_stat = numerator / denominator if denominator and denominator > 0 else np.nan
        p_value = float(1.0 - f.cdf(f_stat, tested_restriction_count, residual_df)) if np.isfinite(f_stat) else np.nan

        total_ss = float(((y_vector - y_vector.mean()) ** 2).sum())
        aux_r_squared = 1.0 - unrestricted_rss / total_ss if total_ss > 0 else np.nan

        xtx_inv = np.linalg.pinv(design_unrestricted.T @ design_unrestricted)
        sigma2 = unrestricted_rss / residual_df
        covariance = sigma2 * xtx_inv
        covariance = 0.5 * (covariance + covariance.T)
        covariance_diag = np.diag(covariance).astype(float, copy=False)
        covariance_diag = np.where(covariance_diag < 0.0, np.maximum(covariance_diag, 0.0), covariance_diag)
        se = np.sqrt(covariance_diag)
        missing_coef = float(beta_unrestricted[1 + len(clean_x_obs)])
        missing_se = float(se[1 + len(clean_x_obs)])
        if missing_se > 0:
            missing_t = missing_coef / missing_se
            missing_pvalue = float(2.0 * (1.0 - t.cdf(abs(missing_t), residual_df)))
        else:
            missing_pvalue = np.nan

    return {
        "aux_n_obs": n_obs,
        "aux_r_squared": float(aux_r_squared) if np.isfinite(aux_r_squared) else np.nan,
        "stage1_f_stat": float(f_stat) if np.isfinite(f_stat) else np.nan,
        "stage1_p_value": p_value,
        "tested_restriction_count": tested_restriction_count,
        "missing_indicator_coef": missing_coef,
        "missing_indicator_pvalue": missing_pvalue,
    }
