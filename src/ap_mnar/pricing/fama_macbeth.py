from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _clean_feature_columns(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> list[str]:
    clean_columns: list[str] = []
    for column in feature_columns:
        if column not in frame.columns:
            continue
        if frame[column].notna().sum() == 0:
            continue
        if frame[column].nunique(dropna=True) <= 1:
            continue
        clean_columns.append(column)
    return clean_columns


def run_fama_macbeth_regression(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    signal: str,
    benchmark_type: str,
    specification: str,
    sample_name: str,
    return_col: str = "ret_fwd_1m",
    min_cross_section: int = 25,
    nw_lags: int = 6,
    focus_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = ["date", return_col, *feature_columns]
    working = frame.dropna(subset=required_columns).copy()
    if working.empty:
        return _empty_fama_macbeth_summary(), _empty_monthly_coefficients()

    monthly_rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    min_required_obs = max(int(min_cross_section), len(feature_columns) + 2)

    for date, subset in working.groupby("date"):
        subset = subset.copy()
        if len(subset) < min_required_obs:
            continue
        clean_feature_columns = _clean_feature_columns(subset, feature_columns)
        if not clean_feature_columns:
            continue

        design = sm.add_constant(subset[clean_feature_columns], has_constant="add")
        model = sm.OLS(subset[return_col], design).fit()
        monthly_rows.extend(
            _extract_monthly_rows(
                model=model,
                date=date,
                signal=signal,
                benchmark_type=benchmark_type,
                specification=specification,
                sample_name=sample_name,
                n_obs=len(subset),
            )
        )

    monthly_frame = pd.DataFrame(monthly_rows)
    if monthly_frame.empty:
        return _empty_fama_macbeth_summary(), _empty_monthly_coefficients()

    summary_rows: list[dict[str, float | int | str]] = []
    focus_set = set(focus_columns or [])
    for coefficient, subset in monthly_frame.groupby("coefficient"):
        coef_series = subset["coefficient_value"].dropna()
        if coef_series.empty:
            continue

        mean_coef, t_stat, p_value = _hac_mean_inference(coef_series, nw_lags=nw_lags)
        summary_rows.append(
            {
                "signal": signal,
                "benchmark_type": benchmark_type,
                "sample_name": sample_name,
                "specification": specification,
                "coefficient": coefficient,
                "is_focus_regressor": coefficient in focus_set,
                "month_count": int(len(coef_series)),
                "mean_coefficient": mean_coef,
                "t_stat": t_stat,
                "p_value": p_value,
                "positive_month_share": float((coef_series > 0).mean()),
                "coefficient_std": float(coef_series.std(ddof=1)) if len(coef_series) > 1 else np.nan,
                "mean_cross_section_nobs": float(subset["cross_section_nobs"].mean()),
                "mean_cross_section_r2": float(subset["cross_section_r2"].mean()),
            }
        )

    return (
        pd.DataFrame(summary_rows).sort_values(
            ["signal", "benchmark_type", "sample_name", "specification", "coefficient"],
            ignore_index=True,
        ),
        monthly_frame.sort_values(
            ["signal", "benchmark_type", "sample_name", "specification", "coefficient", "date"],
            ignore_index=True,
        ),
    )


def _extract_monthly_rows(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    date: pd.Timestamp,
    signal: str,
    benchmark_type: str,
    specification: str,
    sample_name: str,
    n_obs: int,
) -> list[dict[str, float | int | str | pd.Timestamp]]:
    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    for coefficient, value in model.params.items():
        rows.append(
            {
                "date": date,
                "signal": signal,
                "benchmark_type": benchmark_type,
                "sample_name": sample_name,
                "specification": specification,
                "coefficient": coefficient,
                "coefficient_value": float(value),
                "cross_section_nobs": int(n_obs),
                "cross_section_r2": float(model.rsquared),
            }
        )
    return rows


def _hac_mean_inference(values: pd.Series, nw_lags: int) -> tuple[float, float, float]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return np.nan, np.nan, np.nan
    if len(clean) == 1:
        return float(clean.iloc[0]), np.nan, np.nan

    design = np.ones((len(clean), 1))
    max_lags = max(0, min(int(nw_lags), len(clean) - 1))
    result = sm.OLS(clean.to_numpy(dtype=float), design).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max_lags},
    )
    return float(result.params[0]), float(result.tvalues[0]), float(result.pvalues[0])


def _empty_fama_macbeth_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "signal",
            "benchmark_type",
            "sample_name",
            "specification",
            "coefficient",
            "is_focus_regressor",
            "month_count",
            "mean_coefficient",
            "t_stat",
            "p_value",
            "positive_month_share",
            "coefficient_std",
            "mean_cross_section_nobs",
            "mean_cross_section_r2",
        ]
    )


def _empty_monthly_coefficients() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "signal",
            "benchmark_type",
            "sample_name",
            "specification",
            "coefficient",
            "coefficient_value",
            "cross_section_nobs",
            "cross_section_r2",
        ]
    )
