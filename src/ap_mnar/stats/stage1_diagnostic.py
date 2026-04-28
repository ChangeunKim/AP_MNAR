from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_stage1_regression_diagnostic(
    sample: pd.DataFrame,
    x_obs_columns: Sequence[str],
    return_residual_col: str,
) -> dict[str, float | int]:
    working = sample.copy()
    working["missing_indicator"] = working["missing_indicator"].astype(float)

    design = pd.DataFrame(index=working.index)
    clean_x_obs = []
    for column in x_obs_columns:
        if working[column].nunique(dropna=True) <= 1:
            continue
        design[column] = working[column]
        interaction_col = f"missing_x_{column}"
        design[interaction_col] = working["missing_indicator"] * working[column]
        clean_x_obs.append(column)

    design["missing_indicator"] = working["missing_indicator"]
    design = sm.add_constant(design, has_constant="add")
    model = sm.OLS(working[return_residual_col], design).fit()

    restriction_cols = ["missing_indicator", *[f"missing_x_{column}" for column in clean_x_obs]]
    restriction = np.zeros((len(restriction_cols), len(model.params)))
    for row_idx, column in enumerate(restriction_cols):
        restriction[row_idx, model.params.index.get_loc(column)] = 1.0

    f_test = model.f_test(restriction)
    return {
        "aux_n_obs": int(model.nobs),
        "aux_r_squared": float(model.rsquared),
        "stage1_f_stat": float(np.asarray(f_test.fvalue).squeeze()),
        "stage1_p_value": float(np.asarray(f_test.pvalue).squeeze()),
        "tested_restriction_count": len(restriction_cols),
        "missing_indicator_coef": float(model.params.get("missing_indicator", np.nan)),
        "missing_indicator_pvalue": float(model.pvalues.get("missing_indicator", np.nan)),
    }
