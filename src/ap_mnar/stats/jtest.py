from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2


def run_missingness_jtest(
    sample: pd.DataFrame,
    x_obs_columns: Sequence[str],
    return_residual_col: str,
) -> dict[str, float | int]:
    moment_columns = []
    moment_matrix_parts = []

    missing_indicator = sample["missing_indicator"].astype(float).to_numpy()
    residual = sample[return_residual_col].to_numpy()

    moment_columns.append("missing_indicator")
    moment_matrix_parts.append(missing_indicator * residual)

    clean_x_obs = []
    for column in x_obs_columns:
        values = sample[column].to_numpy()
        interaction = missing_indicator * values * residual
        if np.nanstd(interaction) == 0:
            continue
        clean_x_obs.append(column)
        moment_columns.append(f"missing_x_{column}")
        moment_matrix_parts.append(interaction)

    if not moment_matrix_parts:
        return {
            "j_stat": np.nan,
            "j_p_value": np.nan,
            "j_df": 0,
            "j_moment_count": 0,
        }

    moment_matrix = np.column_stack(moment_matrix_parts)
    n_obs = moment_matrix.shape[0]
    gbar = moment_matrix.mean(axis=0)

    if moment_matrix.shape[1] == 1:
        covariance = np.array([[np.var(moment_matrix[:, 0], ddof=1)]])
    else:
        covariance = np.cov(moment_matrix, rowvar=False, ddof=1)

    covariance_inv = np.linalg.pinv(covariance)
    j_stat = float(n_obs * gbar.T @ covariance_inv @ gbar)
    j_df = int(np.linalg.matrix_rank(covariance))
    j_p_value = float(1 - chi2.cdf(j_stat, j_df)) if j_df > 0 else np.nan

    return {
        "j_stat": j_stat,
        "j_p_value": j_p_value,
        "j_df": j_df,
        "j_moment_count": len(moment_columns),
    }
