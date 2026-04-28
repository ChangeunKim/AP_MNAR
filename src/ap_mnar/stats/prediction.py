from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_oos_r2(predictions: pd.DataFrame) -> float:
    if predictions.empty:
        return np.nan
    residual_ss = ((predictions["actual_return"] - predictions["predicted_return"]) ** 2).sum()
    benchmark_ss = ((predictions["actual_return"] - predictions["benchmark_return"]) ** 2).sum()
    if benchmark_ss <= 0:
        return np.nan
    return float(1.0 - residual_ss / benchmark_ss)


def summarize_rank_ic(predictions: pd.DataFrame) -> dict[str, float | int]:
    monthly_rows: list[dict[str, float | pd.Timestamp]] = []
    for date, subset in predictions.groupby("date"):
        working = subset[["predicted_return", "actual_return"]].dropna()
        if len(working) < 5:
            continue
        if working["predicted_return"].nunique() <= 1 or working["actual_return"].nunique() <= 1:
            continue
        ic = working["predicted_return"].corr(working["actual_return"], method="spearman")
        if pd.isna(ic):
            continue
        monthly_rows.append({"date": date, "rank_ic": float(ic)})

    monthly_frame = pd.DataFrame(monthly_rows)
    if monthly_frame.empty:
        return {
            "rank_ic_month_count": 0,
            "mean_rank_ic": np.nan,
            "median_rank_ic": np.nan,
            "rank_ic_t_stat": np.nan,
        }

    mean_ic = float(monthly_frame["rank_ic"].mean())
    median_ic = float(monthly_frame["rank_ic"].median())
    t_stat = _mean_t_stat(monthly_frame["rank_ic"])
    return {
        "rank_ic_month_count": int(len(monthly_frame)),
        "mean_rank_ic": mean_ic,
        "median_rank_ic": median_ic,
        "rank_ic_t_stat": t_stat,
    }


def build_portfolio_spread_table(
    predictions: pd.DataFrame,
    signal: str,
    regime: str,
    n_buckets: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | pd.Timestamp | int]] = []
    for date, subset in predictions.groupby("date"):
        working = subset[["permno", "predicted_return", "actual_return"]].dropna().copy()
        if len(working) < n_buckets:
            continue
        try:
            working["bucket"] = pd.qcut(working["predicted_return"], n_buckets, labels=False, duplicates="drop")
        except ValueError:
            continue
        if working["bucket"].nunique() < 2:
            continue

        low_bucket = int(working["bucket"].min())
        high_bucket = int(working["bucket"].max())
        long_return = float(working.loc[working["bucket"].eq(high_bucket), "actual_return"].mean())
        short_return = float(working.loc[working["bucket"].eq(low_bucket), "actual_return"].mean())
        rows.append(
            {
                "signal": signal,
                "regime": regime,
                "date": date,
                "n_assets": int(len(working)),
                "long_return": long_return,
                "short_return": short_return,
                "long_short_spread": long_return - short_return,
            }
        )

    return pd.DataFrame(rows)


def summarize_portfolio_spread(spread_table: pd.DataFrame) -> dict[str, float | int]:
    if spread_table.empty:
        return {
            "portfolio_month_count": 0,
            "mean_long_short_spread": np.nan,
            "median_long_short_spread": np.nan,
            "spread_t_stat": np.nan,
            "positive_spread_rate": np.nan,
        }

    spread_series = spread_table["long_short_spread"]
    return {
        "portfolio_month_count": int(len(spread_table)),
        "mean_long_short_spread": float(spread_series.mean()),
        "median_long_short_spread": float(spread_series.median()),
        "spread_t_stat": _mean_t_stat(spread_series),
        "positive_spread_rate": float((spread_series > 0).mean()),
    }


def build_signal_sorted_results(
    predictions: pd.DataFrame,
    signal: str,
    regime: str,
    n_signal_groups: int = 5,
    n_portfolio_buckets: int = 5,
) -> pd.DataFrame:
    required_columns = {"signal_sort_value", "predicted_return", "actual_return", "missing_indicator", "date"}
    if predictions.empty or not required_columns.issubset(predictions.columns):
        return _empty_signal_sorted_results()

    working = predictions.dropna(subset=["signal_sort_value", "predicted_return", "actual_return"]).copy()
    if working.empty:
        return _empty_signal_sorted_results()

    working["signal_sort_group"] = assign_quantile_groups(working["signal_sort_value"], n_signal_groups)
    working = working.dropna(subset=["signal_sort_group"]).copy()
    if working.empty:
        return _empty_signal_sorted_results()

    working["signal_sort_group"] = working["signal_sort_group"].astype(int)
    max_group = int(working["signal_sort_group"].max())

    rows: list[dict[str, float | int | str]] = []
    for group_id, subset in working.groupby("signal_sort_group"):
        subset = subset.copy()
        spread_table = build_portfolio_spread_table(
            subset,
            signal=signal,
            regime=regime,
            n_buckets=max(2, min(n_portfolio_buckets, 5)),
        )
        spread_summary = summarize_portfolio_spread(spread_table)
        rows.append(
            {
                "signal": signal,
                "regime": regime,
                "signal_sort_group": int(group_id),
                "signal_sort_group_label": f"Q{int(group_id)}",
                "signal_sort_group_count": max_group,
                "draw_count": (
                    int(subset["draw_count"].max())
                    if "draw_count" in subset.columns and not subset.empty
                    else 1
                ),
                "group_obs_count": int(len(subset)),
                "group_month_count": int(subset["date"].nunique()),
                "group_missing_obs_count": int(subset["missing_indicator"].sum()),
                "group_missing_obs_rate": float(subset["missing_indicator"].mean()),
                "mean_signal_sort_value": float(subset["signal_sort_value"].mean()),
                "median_signal_sort_value": float(subset["signal_sort_value"].median()),
                "mean_actual_return": float(subset["actual_return"].mean()),
                "mean_predicted_return": float(subset["predicted_return"].mean()),
                "mean_prediction_error": float((subset["actual_return"] - subset["predicted_return"]).mean()),
                "oos_r2": compute_oos_r2(subset),
                **spread_summary,
            }
        )

    return pd.DataFrame(rows).sort_values(["signal_sort_group"], ignore_index=True)


def assign_quantile_groups(
    values: pd.Series,
    n_groups: int,
) -> pd.Series:
    if values.empty:
        return pd.Series(dtype="float64", index=values.index)

    clean = values.astype(float)
    non_missing = clean.dropna()
    groups = pd.Series(np.nan, index=clean.index, dtype="float64")
    if non_missing.empty:
        return groups

    if non_missing.nunique() <= 1:
        groups.loc[non_missing.index] = 1.0
        return groups

    try:
        assigned = pd.qcut(
            non_missing,
            q=max(2, int(n_groups)),
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        groups.loc[non_missing.index] = 1.0
        return groups

    groups.loc[non_missing.index] = assigned.astype(float) + 1.0
    return groups


def _mean_t_stat(values: pd.Series) -> float:
    clean = values.dropna()
    if len(clean) < 2:
        return np.nan
    sample_std = float(clean.std(ddof=1))
    if sample_std == 0:
        return np.nan
    return float(clean.mean() / (sample_std / math.sqrt(len(clean))))


def _empty_signal_sorted_results() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "signal",
            "regime",
            "signal_sort_group",
            "signal_sort_group_label",
            "signal_sort_group_count",
            "draw_count",
            "group_obs_count",
            "group_month_count",
            "group_missing_obs_count",
            "group_missing_obs_rate",
            "mean_signal_sort_value",
            "median_signal_sort_value",
            "mean_actual_return",
            "mean_predicted_return",
            "mean_prediction_error",
            "oos_r2",
            "portfolio_month_count",
            "mean_long_short_spread",
            "median_long_short_spread",
            "spread_t_stat",
            "positive_spread_rate",
        ]
    )
