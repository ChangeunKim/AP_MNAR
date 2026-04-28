from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_pooled_pricing_regression(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    signal: str,
    specification: str,
    sample_name: str,
    return_col: str = "ret_fwd_1m",
    focus_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    required_columns = [return_col, *feature_columns]
    working = frame.dropna(subset=required_columns).copy()
    if working.empty:
        return _empty_pooled_summary()

    if any(working[column].nunique(dropna=True) <= 1 for column in feature_columns):
        return _empty_pooled_summary()

    design = sm.add_constant(working[list(feature_columns)], has_constant="add")
    model = sm.OLS(working[return_col], design).fit(cov_type="HC3")
    focus_set = set(focus_columns or [])

    rows: list[dict[str, float | int | str]] = []
    for coefficient, value in model.params.items():
        rows.append(
            {
                "signal": signal,
                "sample_name": sample_name,
                "specification": specification,
                "coefficient": coefficient,
                "is_focus_regressor": coefficient in focus_set,
                "n_obs": int(model.nobs),
                "r_squared": float(model.rsquared),
                "coefficient_value": float(value),
                "t_stat": float(model.tvalues.get(coefficient, np.nan)),
                "p_value": float(model.pvalues.get(coefficient, np.nan)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["signal", "sample_name", "specification", "coefficient"],
        ignore_index=True,
    )


def build_coverage_decomposition_table(
    fama_macbeth_table: pd.DataFrame,
) -> pd.DataFrame:
    if fama_macbeth_table.empty:
        return fama_macbeth_table.copy()

    return fama_macbeth_table.loc[
        fama_macbeth_table["sample_name"].isin(["within_signal_coverage", "outside_signal_coverage"])
        & fama_macbeth_table["is_focus_regressor"].astype(bool)
    ].copy().sort_values(
        ["signal", "sample_name", "specification", "coefficient"],
        ignore_index=True,
    )


def build_missingness_premium_time_series(
    monthly_coefficients: pd.DataFrame,
) -> pd.DataFrame:
    if monthly_coefficients.empty:
        return monthly_coefficients.copy()

    return monthly_coefficients.loc[
        monthly_coefficients["specification"].eq("baseline_missing_only")
        & monthly_coefficients["coefficient"].eq("missing_indicator")
    ].copy().sort_values(
        ["signal", "sample_name", "date"],
        ignore_index=True,
    )


def _empty_pooled_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "signal",
            "sample_name",
            "specification",
            "coefficient",
            "is_focus_regressor",
            "n_obs",
            "r_squared",
            "coefficient_value",
            "t_stat",
            "p_value",
        ]
    )
