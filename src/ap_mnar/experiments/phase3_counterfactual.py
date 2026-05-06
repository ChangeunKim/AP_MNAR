from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ap_mnar.data.x_obs import load_x_obs_spec, merge_x_obs_columns
from ap_mnar.experiments.step0_audit import DEFAULT_SIGNALS
from ap_mnar.models.benchmark_variants import (
    augment_signal_history_features,
    classify_signal_pattern_slice,
    get_signal_benchmark_specs,
)
from ap_mnar.models.counterfactual import (
    COUNTERFACTUAL_REGIMES,
    build_counterfactual_columns,
    build_stochastic_counterfactual_column,
    fit_fast_ols,
    fit_counterfactual_imputation_bundle,
    get_regime_frame,
    predict_fast_ols,
)
from ap_mnar.models.mar_benchmark import build_signal_mar_panel
from ap_mnar.reporting.figures import (
    plot_counterfactual_delta_r2_by_signal_group,
    plot_counterfactual_sensitivity_by_signal,
)
from ap_mnar.stats.prediction import (
    build_signal_sorted_results,
    build_portfolio_spread_table,
    compute_oos_r2,
    summarize_portfolio_spread,
    summarize_rank_ic,
)

try:
    from tqdm.auto import tqdm

    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in minimal environments
    TQDM_AVAILABLE = False


@dataclass(frozen=True)
class Phase3Paths:
    panel_with_missingness_path: Path
    x_obs_config_path: Path
    firm_panel_path: Path
    output_root: Path


def run_phase3_counterfactual(
    paths: Phase3Paths,
    signals: Sequence[str] = DEFAULT_SIGNALS,
    min_train_years: int = 5,
    stochastic_draws: int = 25,
    quantile_bins: int = 5,
    random_seed: int = 0,
    signal_sort_groups: int = 5,
    show_progress: bool = True,
    include_augmented_signal_history: bool = True,
) -> dict[str, pd.DataFrame]:
    panel = pd.read_parquet(paths.panel_with_missingness_path)
    x_obs_spec = load_x_obs_spec(paths.x_obs_config_path)
    panel = merge_x_obs_columns(panel, paths.firm_panel_path, x_obs_spec.columns)

    oos_rows: list[dict[str, float | int | str]] = []
    portfolio_rows: list[dict[str, float | int | str]] = []
    signal_sorted_frames: list[pd.DataFrame] = []
    pattern_slice_rows: list[dict[str, float | int | str]] = []
    benchmark_specs = get_signal_benchmark_specs(include_augmented_signal_history)

    for signal in _progress_iter(signals, show_progress, desc="Phase 3 signals", unit="signal"):
        for benchmark_spec in benchmark_specs:
            benchmark_panel = panel.copy()
            benchmark_x_obs = list(x_obs_spec.columns)
            if benchmark_spec.augment_signal_history:
                benchmark_panel, benchmark_x_obs = augment_signal_history_features(
                    benchmark_panel,
                    signal,
                    benchmark_x_obs,
                )

            sample, _, _ = build_signal_mar_panel(benchmark_panel, signal, benchmark_x_obs)
            sample = sample.sort_values(["date", "permno"]).copy()
            sample["year"] = sample["date"].dt.year.astype(int)
            sample["pattern_slice"] = classify_signal_pattern_slice(sample)

            signal_oos_rows, signal_portfolio_rows, signal_sorted_table, signal_pattern_rows = run_signal_counterfactual_backtest(
                sample=sample,
                signal=signal,
                benchmark_type=benchmark_spec.benchmark_type,
                x_obs_columns=benchmark_x_obs,
                min_train_years=min_train_years,
                stochastic_draws=stochastic_draws,
                quantile_bins=quantile_bins,
                random_seed=random_seed,
                signal_sort_groups=signal_sort_groups,
                show_progress=show_progress,
            )
            oos_rows.extend(signal_oos_rows)
            portfolio_rows.extend(signal_portfolio_rows)
            pattern_slice_rows.extend(signal_pattern_rows)
            if not signal_sorted_table.empty:
                signal_sorted_frames.append(signal_sorted_table)

    oos_table = pd.DataFrame(oos_rows).sort_values(["signal", "benchmark_type", "regime"], ignore_index=True)
    portfolio_table = pd.DataFrame(portfolio_rows).sort_values(["signal", "benchmark_type", "regime"], ignore_index=True)
    signal_sorted_table = _concat_signal_sorted_frames(signal_sorted_frames)
    if not signal_sorted_table.empty:
        signal_sorted_table = signal_sorted_table.sort_values(
            ["signal", "benchmark_type", "regime", "signal_sort_group"],
            ignore_index=True,
        )
    pattern_slice_table = pd.DataFrame(pattern_slice_rows).sort_values(
        ["signal", "benchmark_type", "pattern_slice", "regime"],
        ignore_index=True,
    ) if pattern_slice_rows else pd.DataFrame()

    if not oos_table.empty:
        oos_table = add_complete_case_deltas(
            oos_table,
            metric_columns=["oos_r2", "mean_rank_ic"],
        )
    if not portfolio_table.empty:
        portfolio_table = add_complete_case_deltas(
            portfolio_table,
            metric_columns=["mean_long_short_spread"],
        )
    if not signal_sorted_table.empty:
        signal_sorted_table = add_complete_case_group_deltas(
            signal_sorted_table,
            metric_columns=["oos_r2", "mean_long_short_spread"],
        )
    if not pattern_slice_table.empty:
        pattern_slice_table = add_complete_case_group_deltas(
            pattern_slice_table,
            group_columns=["signal", "benchmark_type", "pattern_slice"],
            metric_columns=["oos_r2", "mean_long_short_spread"],
        )
    attenuation_summary = build_counterfactual_attenuation_summary(oos_table, portfolio_table)

    outputs = {
        "oos_table": oos_table,
        "portfolio_table": portfolio_table,
        "signal_sorted_table": signal_sorted_table,
        "pattern_slice_table": pattern_slice_table,
        "attenuation_summary": attenuation_summary,
    }
    write_phase3_outputs(outputs, paths.output_root)
    return outputs


def run_signal_counterfactual_backtest(
    sample: pd.DataFrame,
    signal: str,
    benchmark_type: str,
    x_obs_columns: Sequence[str],
    min_train_years: int,
    stochastic_draws: int,
    quantile_bins: int,
    random_seed: int,
    signal_sort_groups: int,
    show_progress: bool = True,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    pd.DataFrame,
    list[dict[str, float | int | str]],
]:
    years = sorted(sample["year"].dropna().unique())
    if len(years) <= min_train_years:
        raise ValueError(f"Signal {signal} does not have enough years for Phase 3 backtesting.")

    test_years = years[min_train_years:]
    test_mask = sample["year"].isin(test_years)
    total_test_rows = int(test_mask.sum())
    total_test_missing_rows = int(sample.loc[test_mask, "missing_indicator"].sum())
    total_test_months = int(sample.loc[test_mask, "date"].nunique())

    prediction_store: dict[str, list[pd.DataFrame]] = {regime: [] for regime in COUNTERFACTUAL_REGIMES}

    year_iter = _progress_iter(test_years, show_progress, desc=f"{signal}: test years", unit="year", leave=False)
    for test_year in year_iter:
        train = sample.loc[sample["year"] < test_year].copy()
        test = sample.loc[sample["year"] == test_year].copy()
        if train.empty or test.empty:
            continue

        observed_train = train.loc[train["observed_indicator"].eq(1)].copy()
        if observed_train.empty:
            continue

        signal_col = f"{signal}_tier1"
        imputation_bundle = fit_counterfactual_imputation_bundle(
            observed_train=observed_train,
            signal_col=signal_col,
            x_obs_columns=x_obs_columns,
            quantile_bins=quantile_bins,
        )

        train_cf = build_counterfactual_columns(
            train,
            signal,
            x_obs_columns,
            imputation_bundle.ols_bundle,
            imputation_bundle.unconditional_mean,
        )
        test_cf = build_counterfactual_columns(
            test,
            signal,
            x_obs_columns,
            imputation_bundle.ols_bundle,
            imputation_bundle.unconditional_mean,
        )

        deterministic_regimes = {"complete_case", "unconditional_mean", "conditional_mean"}
        for regime in COUNTERFACTUAL_REGIMES:
            if regime not in deterministic_regimes:
                continue
            train_regime, train_signal_col = get_regime_frame(train_cf, signal, regime)
            test_regime, test_signal_col = get_regime_frame(test_cf, signal, regime)

            train_regime = train_regime.dropna(subset=["ret_fwd_1m", *x_obs_columns, train_signal_col]).copy()
            test_regime = test_regime.dropna(subset=["ret_fwd_1m", *x_obs_columns, test_signal_col]).copy()
            if train_regime.empty or test_regime.empty:
                continue

            return_model = fit_fast_ols(train_regime, "ret_fwd_1m", [*x_obs_columns, train_signal_col])
            predicted_return = predict_fast_ols(return_model, test_regime)

            prediction_frame = test_regime[["permno", "date", "ret_fwd_1m", "missing_indicator"]].copy()
            prediction_frame.rename(columns={"ret_fwd_1m": "actual_return"}, inplace=True)
            prediction_frame["predicted_return"] = predicted_return
            prediction_frame["benchmark_return"] = float(train_regime["ret_fwd_1m"].mean())
            prediction_frame["signal_sort_value"] = test_regime[test_signal_col].to_numpy(dtype=float)
            prediction_frame["signal"] = signal
            prediction_frame["benchmark_type"] = benchmark_type
            prediction_frame["regime"] = regime
            prediction_frame["test_year"] = int(test_year)
            prediction_frame["draw_id"] = 0
            prediction_frame["pattern_slice"] = test_regime["pattern_slice"].to_numpy()
            prediction_store[regime].append(prediction_frame)

        stochastic_regimes = {"residual_bootstrap", "conditional_quantile_draw"}
        for regime in stochastic_regimes:
            draw_iter = _progress_iter(
                range(1, stochastic_draws + 1),
                show_progress,
                desc=f"{signal} {test_year} {regime}",
                unit="draw",
                leave=False,
            )
            for draw_id in draw_iter:
                draw_seed = _draw_seed(random_seed, signal, regime, test_year, draw_id)
                rng = np.random.default_rng(draw_seed)

                train_draw = build_stochastic_counterfactual_column(train, signal, imputation_bundle, regime, rng)
                test_draw = build_stochastic_counterfactual_column(test, signal, imputation_bundle, regime, rng)

                train_regime, train_signal_col = get_regime_frame(train_draw, signal, regime)
                test_regime, test_signal_col = get_regime_frame(test_draw, signal, regime)

                train_regime = train_regime.dropna(subset=["ret_fwd_1m", *x_obs_columns, train_signal_col]).copy()
                test_regime = test_regime.dropna(subset=["ret_fwd_1m", *x_obs_columns, test_signal_col]).copy()
                if train_regime.empty or test_regime.empty:
                    continue

                return_model = fit_fast_ols(train_regime, "ret_fwd_1m", [*x_obs_columns, train_signal_col])
                predicted_return = predict_fast_ols(return_model, test_regime)

                prediction_frame = test_regime[["permno", "date", "ret_fwd_1m", "missing_indicator"]].copy()
                prediction_frame.rename(columns={"ret_fwd_1m": "actual_return"}, inplace=True)
                prediction_frame["predicted_return"] = predicted_return
                prediction_frame["benchmark_return"] = float(train_regime["ret_fwd_1m"].mean())
                prediction_frame["signal_sort_value"] = test_regime[test_signal_col].to_numpy(dtype=float)
                prediction_frame["signal"] = signal
                prediction_frame["benchmark_type"] = benchmark_type
                prediction_frame["regime"] = regime
                prediction_frame["test_year"] = int(test_year)
                prediction_frame["draw_id"] = int(draw_id)
                prediction_frame["pattern_slice"] = test_regime["pattern_slice"].to_numpy()
                prediction_store[regime].append(prediction_frame)

    oos_rows: list[dict[str, float | int | str]] = []
    portfolio_rows: list[dict[str, float | int | str]] = []
    signal_sorted_frames: list[pd.DataFrame] = []
    pattern_slice_rows: list[dict[str, float | int | str]] = []

    for regime in COUNTERFACTUAL_REGIMES:
        prediction_frame = _aggregate_prediction_draws(_concat_frames(prediction_store[regime]))
        rank_ic_summary = summarize_rank_ic(prediction_frame)
        spread_table = build_portfolio_spread_table(prediction_frame, signal, regime)
        spread_summary = summarize_portfolio_spread(spread_table)
        signal_sorted_frame = build_signal_sorted_results(
            prediction_frame,
            signal=signal,
            regime=regime,
            n_signal_groups=signal_sort_groups,
        )
        if not signal_sorted_frame.empty:
            signal_sorted_frame["benchmark_type"] = benchmark_type
            signal_sorted_frames.append(signal_sorted_frame)

        predicted_obs_count = int(len(prediction_frame))
        predicted_month_count = int(prediction_frame["date"].nunique()) if not prediction_frame.empty else 0
        missing_prediction_count = int(prediction_frame["missing_indicator"].sum()) if not prediction_frame.empty else 0
        realized_draw_count = int(prediction_frame["draw_count"].max()) if "draw_count" in prediction_frame.columns and not prediction_frame.empty else 0
        mean_draw_std = float(prediction_frame["prediction_draw_std"].mean()) if "prediction_draw_std" in prediction_frame.columns and not prediction_frame.empty else np.nan
        missing_draw_std = (
            float(prediction_frame.loc[prediction_frame["missing_indicator"].eq(1), "prediction_draw_std"].mean())
            if "prediction_draw_std" in prediction_frame.columns and not prediction_frame.empty
            else np.nan
        )

        oos_rows.append(
            {
                "signal": signal,
                "benchmark_type": benchmark_type,
                "regime": regime,
                "draw_count": realized_draw_count,
                "test_year_count": int(len(test_years)),
                "test_month_count": total_test_months,
                "predicted_obs_count": predicted_obs_count,
                "prediction_coverage_rate": predicted_obs_count / total_test_rows if total_test_rows else np.nan,
                "missing_row_prediction_count": missing_prediction_count,
                "missing_row_prediction_rate": (
                    missing_prediction_count / total_test_missing_rows if total_test_missing_rows else np.nan
                ),
                "mean_prediction_draw_std": mean_draw_std,
                "missing_row_mean_prediction_draw_std": missing_draw_std,
                "oos_r2": compute_oos_r2(prediction_frame),
                **rank_ic_summary,
            }
        )

        portfolio_rows.append(
            {
                "signal": signal,
                "benchmark_type": benchmark_type,
                "regime": regime,
                "draw_count": realized_draw_count,
                "portfolio_test_month_count": total_test_months,
                "predicted_month_count": predicted_month_count,
                **spread_summary,
            }
        )

        pattern_slice_rows.extend(
            _build_pattern_slice_counterfactual_rows(
                prediction_frame=prediction_frame,
                signal=signal,
                benchmark_type=benchmark_type,
                regime=regime,
            )
        )

    signal_sorted_table = _concat_signal_sorted_frames(signal_sorted_frames)
    return oos_rows, portfolio_rows, signal_sorted_table, pattern_slice_rows


def add_complete_case_deltas(
    frame: pd.DataFrame,
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    working = frame.copy()
    for group_keys, subset in working.groupby(["signal", "benchmark_type"]):
        baseline = subset.loc[subset["regime"].eq("complete_case")]
        if baseline.empty:
            continue
        baseline_row = baseline.iloc[0]
        signal, benchmark_type = group_keys
        mask = working["signal"].eq(signal) & working["benchmark_type"].eq(benchmark_type)
        for column in metric_columns:
            delta_col = f"delta_{column}_vs_complete_case"
            working.loc[mask, delta_col] = working.loc[mask, column] - baseline_row[column]
    return working


def add_complete_case_group_deltas(
    frame: pd.DataFrame,
    metric_columns: Sequence[str],
    group_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    working = frame.copy()
    group_columns = list(group_columns or ["signal", "benchmark_type", "signal_sort_group"])
    for group_keys, subset in working.groupby(group_columns):
        baseline = subset.loc[subset["regime"].eq("complete_case")]
        if baseline.empty:
            continue
        baseline_row = baseline.iloc[0]
        mask = np.ones(len(working), dtype=bool)
        for column, value in zip(group_columns, group_keys):
            mask &= working[column].eq(value)
        for column in metric_columns:
            delta_col = f"delta_{column}_vs_complete_case"
            working.loc[mask, delta_col] = working.loc[mask, column] - baseline_row[column]
    return working


def write_phase3_outputs(outputs: dict[str, pd.DataFrame], output_root: Path) -> None:
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    outputs["oos_table"].to_csv(tables_dir / "counterfactual_oos_results.csv", index=False)
    outputs["oos_table"].to_csv(tables_dir / "counterfactual_oos_results_by_benchmark.csv", index=False)
    outputs["portfolio_table"].to_csv(tables_dir / "portfolio_spread_comparison.csv", index=False)
    outputs["portfolio_table"].to_csv(tables_dir / "portfolio_spread_comparison_by_benchmark.csv", index=False)
    outputs["signal_sorted_table"].to_csv(tables_dir / "counterfactual_signal_sorted_results.csv", index=False)
    outputs["signal_sorted_table"].to_csv(
        tables_dir / "counterfactual_signal_sorted_results_by_benchmark.csv",
        index=False,
    )
    outputs["pattern_slice_table"].to_csv(tables_dir / "counterfactual_pattern_slice_results.csv", index=False)
    outputs["attenuation_summary"].to_csv(tables_dir / "counterfactual_attenuation_summary.csv", index=False)

    plot_counterfactual_sensitivity_by_signal(
        _select_plot_benchmark(outputs["oos_table"]),
        figures_dir / "counterfactual_sensitivity_by_signal.png",
    )
    plot_counterfactual_delta_r2_by_signal_group(
        _select_plot_benchmark(outputs["signal_sorted_table"]),
        figures_dir / "counterfactual_delta_r2_by_signal_group.png",
    )


def _concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(
            columns=[
                "permno",
                "date",
                "actual_return",
                "missing_indicator",
                "pattern_slice",
                "predicted_return",
                "benchmark_return",
                "signal",
                "benchmark_type",
                "regime",
                "test_year",
                "draw_id",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _concat_signal_sorted_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(
            columns=[
                "signal",
                "benchmark_type",
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
    return pd.concat(frames, ignore_index=True)


def _aggregate_prediction_draws(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        predictions = predictions.copy()
        predictions["prediction_draw_std"] = pd.Series(dtype=float)
        predictions["draw_count"] = pd.Series(dtype=int)
        predictions["signal_sort_value"] = pd.Series(dtype=float)
        predictions["signal_sort_value_draw_std"] = pd.Series(dtype=float)
        return predictions

    group_keys = [
        "permno",
        "date",
        "missing_indicator",
        "pattern_slice",
        "signal",
        "benchmark_type",
        "regime",
        "test_year",
    ]
    aggregated = (
        predictions.groupby(group_keys, dropna=False)
        .agg(
            actual_return=("actual_return", "first"),
            benchmark_return=("benchmark_return", "mean"),
            predicted_return=("predicted_return", "mean"),
            prediction_draw_std=("predicted_return", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
            signal_sort_value=("signal_sort_value", "mean"),
            signal_sort_value_draw_std=(
                "signal_sort_value",
                lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            ),
            draw_count=("draw_id", "nunique"),
        )
        .reset_index()
    )
    return aggregated


def _draw_seed(
    base_seed: int,
    signal: str,
    regime: str,
    test_year: int,
    draw_id: int,
) -> int:
    signal_hash = sum(ord(char) for char in signal)
    regime_hash = sum(ord(char) for char in regime)
    return int(base_seed + signal_hash * 101 + regime_hash * 1009 + test_year * 17 + draw_id)


def _progress_iter(
    values: Sequence,
    show_progress: bool,
    desc: str,
    unit: str,
    leave: bool = True,
):
    if show_progress and TQDM_AVAILABLE:
        return tqdm(values, desc=desc, unit=unit, leave=leave)
    return values


def _build_pattern_slice_counterfactual_rows(
    prediction_frame: pd.DataFrame,
    signal: str,
    benchmark_type: str,
    regime: str,
) -> list[dict[str, float | int | str]]:
    if prediction_frame.empty or "pattern_slice" not in prediction_frame.columns:
        return []
    rows: list[dict[str, float | int | str]] = []
    for pattern_slice, subset in prediction_frame.groupby("pattern_slice", dropna=False):
        subset = subset.copy()
        spread_table = build_portfolio_spread_table(subset, signal=signal, regime=regime)
        spread_summary = summarize_portfolio_spread(spread_table)
        rows.append(
            {
                "signal": signal,
                "benchmark_type": benchmark_type,
                "pattern_slice": pattern_slice,
                "regime": regime,
                "n_obs": int(len(subset)),
                "missing_share": float(subset["missing_indicator"].mean()),
                "oos_r2": compute_oos_r2(subset),
                **spread_summary,
            }
        )
    return rows


def build_counterfactual_attenuation_summary(
    oos_table: pd.DataFrame,
    portfolio_table: pd.DataFrame,
) -> pd.DataFrame:
    if oos_table.empty or portfolio_table.empty:
        return pd.DataFrame()

    fixed_oos = oos_table.loc[oos_table["benchmark_type"].eq("fixed_x_obs")].copy()
    aug_oos = oos_table.loc[oos_table["benchmark_type"].eq("augmented_signal_history")].copy()
    fixed_port = portfolio_table.loc[portfolio_table["benchmark_type"].eq("fixed_x_obs")].copy()
    aug_port = portfolio_table.loc[portfolio_table["benchmark_type"].eq("augmented_signal_history")].copy()

    merged = (
        fixed_oos.merge(
            aug_oos,
            on=["signal", "regime"],
            how="inner",
            suffixes=("_fixed", "_augmented"),
        ).merge(
            fixed_port[["signal", "regime", "delta_mean_long_short_spread_vs_complete_case"]].rename(
                columns={"delta_mean_long_short_spread_vs_complete_case": "delta_spread_fixed"}
            ),
            on=["signal", "regime"],
            how="left",
        ).merge(
            aug_port[["signal", "regime", "delta_mean_long_short_spread_vs_complete_case"]].rename(
                columns={"delta_mean_long_short_spread_vs_complete_case": "delta_spread_augmented"}
            ),
            on=["signal", "regime"],
            how="left",
        )
    )
    if merged.empty:
        return pd.DataFrame()

    merged["attenuation_ratio_r2"] = _safe_ratio(
        merged["delta_oos_r2_vs_complete_case_augmented"],
        merged["delta_oos_r2_vs_complete_case_fixed"],
    )
    merged["attenuation_ratio_spread"] = _safe_ratio(
        merged["delta_spread_augmented"],
        merged["delta_spread_fixed"],
    )
    merged["attenuation_ratio_rank_ic"] = _safe_ratio(
        merged["delta_mean_rank_ic_vs_complete_case_augmented"],
        merged["delta_mean_rank_ic_vs_complete_case_fixed"],
    )
    return merged[
        [
            "signal",
            "regime",
            "delta_oos_r2_vs_complete_case_fixed",
            "delta_oos_r2_vs_complete_case_augmented",
            "attenuation_ratio_r2",
            "delta_spread_fixed",
            "delta_spread_augmented",
            "attenuation_ratio_spread",
            "delta_mean_rank_ic_vs_complete_case_fixed",
            "delta_mean_rank_ic_vs_complete_case_augmented",
            "attenuation_ratio_rank_ic",
        ]
    ].rename(
        columns={
            "delta_oos_r2_vs_complete_case_fixed": "delta_r2_fixed",
            "delta_oos_r2_vs_complete_case_augmented": "delta_r2_augmented",
            "delta_mean_rank_ic_vs_complete_case_fixed": "rank_ic_delta_fixed",
            "delta_mean_rank_ic_vs_complete_case_augmented": "rank_ic_delta_augmented",
        }
    ).sort_values(["signal", "regime"], ignore_index=True)


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    denom = denominator.astype(float).replace(0.0, np.nan)
    return numerator.astype(float) / denom


def _select_plot_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "benchmark_type" not in frame.columns:
        return frame
    subset = frame.loc[frame["benchmark_type"].eq("fixed_x_obs")].copy()
    return subset if not subset.empty else frame
