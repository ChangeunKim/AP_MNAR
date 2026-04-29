from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ap_mnar.data.x_obs import build_x_obs_availability_table, load_x_obs_spec, merge_x_obs_columns
from ap_mnar.experiments.step0_audit import DEFAULT_SIGNALS
from ap_mnar.models.counterfactual import (
    build_stochastic_counterfactual_column,
    fit_counterfactual_imputation_bundle,
    fit_fast_ols,
    predict_fast_ols,
)
from ap_mnar.models.mar_benchmark import build_signal_mar_panel
from ap_mnar.reporting.figures import plot_jtest_pvalue_distribution
from ap_mnar.stats.jtest import build_signal_instrument_frame, run_signal_mar_jtest
from ap_mnar.stats.stage1_diagnostic import run_stage1_regression_diagnostic


STEP1A_TABLE_FILENAMES = {
    "stage1_table": "step1a_signal_diagnostic.csv",
    "jtest_table": "step1a_mar_jtest_results.csv",
    "benchmark_table": "step1a_signal_benchmark_comparison.csv",
    "rejection_summary": "step1a_rejection_summary.csv",
    "pattern_slice_table": "step1a_pattern_slice_results.csv",
    "benchmark_strength_table": "step1a_benchmark_strength_comparison.csv",
    "draw_aggregation_table": "step1a_draw_aggregation_diagnostics.csv",
    "moment_contribution_table": "step1a_moment_contribution.csv",
    "x_obs_audit": "x_obs_phase2_availability_audit.csv",
}

LEGACY_TABLE_ALIASES = {
    "stage1_table": "mar_stage1_regression_diagnostic.csv",
    "jtest_table": "mar_jtest_results.csv",
    "benchmark_table": "mar_benchmark_comparison.csv",
    "rejection_summary": "mar_rejection_summary.csv",
}


@dataclass(frozen=True)
class Phase2Paths:
    panel_with_missingness_path: Path
    signal_registry_path: Path
    x_obs_config_path: Path
    firm_panel_path: Path
    output_root: Path


@dataclass(frozen=True)
class SignalPhase2Result:
    stage1_rows: list[dict[str, float | int | str]]
    jtest_rows: list[dict[str, float | int | str]]
    benchmark_rows: list[dict[str, float | int | str]]
    pattern_slice_rows: list[dict[str, float | int | str]]
    draw_aggregation_rows: list[dict[str, float | int | str]]
    moment_contribution_rows: list[dict[str, float | int | str]]


@dataclass(frozen=True)
class Phase2BenchmarkSpec:
    benchmark_type: str
    augment_signal_history: bool


PHASE2_BENCHMARK_SPECS = (
    Phase2BenchmarkSpec("fixed_x_obs", False),
    Phase2BenchmarkSpec("augmented_signal_history", True),
)


def run_phase2_mar_test(
    paths: Phase2Paths,
    signals: Sequence[str] = DEFAULT_SIGNALS,
    n_draws: int = 10,
    random_seed: int = 0,
    n_folds: int = 5,
    mar_draw_regime: str = "conditional_quantile_draw",
    augment_signal_history: bool = True,
    include_pattern_slices: bool = True,
) -> dict[str, pd.DataFrame]:
    panel, resolved_signals, x_obs_columns = _load_phase2_inputs(paths, signals)

    stage1_rows: list[dict[str, float | int | str]] = []
    jtest_rows: list[dict[str, float | int | str]] = []
    benchmark_rows: list[dict[str, float | int | str]] = []
    pattern_slice_rows: list[dict[str, float | int | str]] = []
    draw_aggregation_rows: list[dict[str, float | int | str]] = []
    moment_contribution_rows: list[dict[str, float | int | str]] = []

    if augment_signal_history:
        benchmark_specs = list(PHASE2_BENCHMARK_SPECS)
    else:
        benchmark_specs = [spec for spec in PHASE2_BENCHMARK_SPECS if not spec.augment_signal_history]

    for signal in resolved_signals:
        signal_result = _run_signal_phase2(
            panel=panel,
            signal=signal,
            x_obs_columns=x_obs_columns,
            n_draws=n_draws,
            random_seed=random_seed,
            n_folds=n_folds,
            mar_draw_regime=mar_draw_regime,
            benchmark_specs=benchmark_specs,
            include_pattern_slices=include_pattern_slices,
        )
        if signal_result is None:
            continue
        stage1_rows.extend(signal_result.stage1_rows)
        jtest_rows.extend(signal_result.jtest_rows)
        benchmark_rows.extend(signal_result.benchmark_rows)
        pattern_slice_rows.extend(signal_result.pattern_slice_rows)
        draw_aggregation_rows.extend(signal_result.draw_aggregation_rows)
        moment_contribution_rows.extend(signal_result.moment_contribution_rows)

    stage1_table = _build_output_frame(stage1_rows, ["signal", "benchmark_type"])
    jtest_table = _build_output_frame(jtest_rows, ["signal", "benchmark_type"])
    benchmark_table = _build_output_frame(benchmark_rows, ["signal", "benchmark_type", "benchmark"])
    pattern_slice_table = _build_output_frame(pattern_slice_rows, ["signal", "benchmark_type", "slice_name"])
    draw_aggregation_table = _build_output_frame(draw_aggregation_rows, ["signal", "benchmark_type"])
    moment_contribution_table = _build_output_frame(
        moment_contribution_rows,
        ["signal", "benchmark_type", "draw_id", "moment_family", "instrument"],
    )
    rejection_summary = build_rejection_summary(stage1_table, jtest_table)
    benchmark_strength_table = _build_benchmark_strength_table(stage1_table, jtest_table)
    x_obs_audit = build_x_obs_availability_table(panel, x_obs_columns)

    outputs = {
        "stage1_table": stage1_table,
        "jtest_table": jtest_table,
        "benchmark_table": benchmark_table,
        "rejection_summary": rejection_summary,
        "pattern_slice_table": pattern_slice_table,
        "benchmark_strength_table": benchmark_strength_table,
        "draw_aggregation_table": draw_aggregation_table,
        "moment_contribution_table": moment_contribution_table,
        "x_obs_audit": x_obs_audit,
    }
    write_phase2_outputs(outputs, paths.output_root)
    return outputs


def _load_phase2_inputs(
    paths: Phase2Paths,
    signals: Sequence[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    signal_registry = pd.read_csv(paths.signal_registry_path)
    available_signals = set(signal_registry["signal"].tolist())
    resolved_signals = [
        signal
        for signal in signals
        if signal in available_signals
    ]

    x_obs_spec = load_x_obs_spec(paths.x_obs_config_path)
    panel = pd.read_parquet(paths.panel_with_missingness_path)
    panel = merge_x_obs_columns(panel, paths.firm_panel_path, x_obs_spec.columns)
    return panel, resolved_signals, list(x_obs_spec.columns)


def _build_output_frame(
    rows: list[dict[str, float | int | str]],
    sort_columns: Sequence[str],
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(list(sort_columns), ignore_index=True)


def build_rejection_summary(
    stage1_table: pd.DataFrame,
    jtest_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for label, frame, p_col in [
        ("stage1", stage1_table, "stage1_p_value"),
        ("jtest", jtest_table, "j_p_value"),
    ]:
        if frame.empty:
            continue
        for benchmark_type, subset in frame.groupby("benchmark_type", dropna=False):
            valid = subset[p_col].dropna()
            combined_p, stouffer_z = _stouffer_combine_pvalues(valid.tolist())
            rows.append(
                {
                    "test": label,
                    "benchmark_type": benchmark_type,
                    "signals_tested": int(subset["signal"].nunique()),
                    "p_lt_0_10": int((valid < 0.10).sum()),
                    "p_lt_0_05": int((valid < 0.05).sum()),
                    "p_lt_0_01": int((valid < 0.01).sum()),
                    "mean_p_value": float(valid.mean()) if not valid.empty else np.nan,
                    "combined_p_value": combined_p,
                    "stouffer_z": stouffer_z,
                }
            )
    return pd.DataFrame(rows)


def _build_benchmark_strength_table(
    stage1_table: pd.DataFrame,
    jtest_table: pd.DataFrame,
) -> pd.DataFrame:
    if stage1_table.empty or jtest_table.empty:
        return pd.DataFrame()
    merged = stage1_table.merge(
        jtest_table,
        on=["signal", "benchmark_type"],
        how="inner",
        suffixes=("_stage1", "_jtest"),
    )
    return merged[[
        "signal",
        "benchmark_type",
        "n_forward_eval_obs",
        "n_missing_signal_stage1",
        "j_stat",
        "j_p_value",
        "j_mean_draw_p_value",
        "j_p_lt_0_05_share",
        "stage1_f_stat",
        "stage1_p_value",
        "stage1_mean_draw_p_value",
    ]].rename(
        columns={
            "n_forward_eval_obs": "n_obs",
            "n_missing_signal_stage1": "n_missing",
            "j_p_lt_0_05_share": "reject_share",
            "stage1_f_stat": "stage1_F_mean",
            "stage1_p_value": "stage1_p_combined",
        }
    )


def _augment_phase2_signal_history_features(
    panel: pd.DataFrame,
    signal: str,
    base_x_obs_columns: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    working = panel.sort_values(["permno", "date"]).copy()
    signal_col = f"{signal}_tier1"
    observed_flag = working[signal_col].notna().astype(int)
    lagged_signal = working.groupby("permno", sort=False)[signal_col].shift(1)
    lagged_observed = working.groupby("permno", sort=False)[signal_col].shift(1).notna().astype(int)

    history_group = working.groupby("permno", sort=False)
    ever_observed_before = history_group[signal_col].transform(
        lambda s: s.notna().astype(int).shift(1, fill_value=0).cummax()
    ).astype(int)
    missing_before = history_group[signal_col].transform(
        lambda s: s.isna().astype(int).shift(1, fill_value=0).cumsum()
    ).astype(int)

    lagged_fill_value = float(lagged_signal.mean()) if lagged_signal.notna().any() else 0.0
    added_columns = {
        f"{signal}_phase2_lag1_value": lagged_signal.fillna(lagged_fill_value).astype(float),
        f"{signal}_phase2_lag1_observed": lagged_observed.astype(float),
        f"{signal}_phase2_ever_observed_before": ever_observed_before.astype(float),
        f"{signal}_phase2_prior_missing_count": missing_before.astype(float),
        f"{signal}_phase2_before_support": working.get(f"{signal}_before_support", pd.Series(False, index=working.index)).astype(float),
        f"{signal}_phase2_after_support": working.get(f"{signal}_after_support", pd.Series(False, index=working.index)).astype(float),
        f"{signal}_phase2_no_support": working.get(f"{signal}_no_support", pd.Series(False, index=working.index)).astype(float),
        f"{signal}_phase2_prev_observed_flag": history_group[signal_col].transform(
            lambda s: s.notna().astype(int).shift(1, fill_value=0)
        ).astype(float),
    }
    for column, values in added_columns.items():
        working[column] = values.to_numpy(dtype=float)

    augmented_columns = list(base_x_obs_columns) + list(added_columns.keys())
    augmented_columns = list(dict.fromkeys(augmented_columns))
    return working, augmented_columns


def _estimate_forward_residual_variances(
    train_observed: pd.DataFrame,
    train_cond_mean: np.ndarray,
    test_cond_mean: np.ndarray,
    quantile_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    signal_col = train_observed.columns[0]
    train_signal = train_observed[signal_col].to_numpy(dtype=float)
    train_residual = train_signal - np.asarray(train_cond_mean, dtype=float)
    global_variance = float(np.var(train_residual, ddof=1)) if len(train_residual) > 1 else float(np.var(train_residual))
    global_variance = max(global_variance, 1e-6)

    train_edges = np.quantile(
        np.asarray(train_cond_mean, dtype=float),
        np.linspace(0.0, 1.0, max(2, int(quantile_bins)) + 1),
    )
    train_edges = np.unique(train_edges)
    if len(train_edges) < 2:
        center = float(np.mean(train_cond_mean)) if len(train_cond_mean) else 0.0
        train_edges = np.array([center - 1e-8, center + 1e-8], dtype=float)

    train_groups = np.digitize(np.asarray(train_cond_mean, dtype=float), train_edges[1:-1], right=True)
    test_groups = np.digitize(np.asarray(test_cond_mean, dtype=float), train_edges[1:-1], right=True)

    variance_by_group: dict[int, float] = {}
    for group_id in np.unique(train_groups):
        group_residual = train_residual[train_groups == group_id]
        if len(group_residual) <= 1:
            variance_by_group[int(group_id)] = global_variance
        else:
            variance_by_group[int(group_id)] = max(float(np.var(group_residual, ddof=1)), 1e-6)

    observed_variance = np.array(
        [variance_by_group.get(int(group_id), global_variance) for group_id in test_groups],
        dtype=float,
    )
    # A pragmatic paper-style correction: imputed rows are noisier because the signal is generated.
    missing_variance = np.array(
        [global_variance + variance_by_group.get(int(group_id), global_variance) for group_id in test_groups],
        dtype=float,
    )
    return observed_variance, missing_variance


def _classify_phase2_missing_pattern_slice(sample: pd.DataFrame) -> pd.Series:
    output = pd.Series(index=sample.index, dtype="object")
    ordered = sample.sort_values(["permno", "date"])

    for _, group in ordered.groupby("permno", sort=False):
        observed_mask = group["observed_indicator"].eq(1)
        if not observed_mask.any():
            labels = np.where(group["missing_indicator"].eq(1), "always_missing", "observed")
            output.loc[group.index] = labels
            continue

        observed_dates = group.loc[observed_mask, "date"]
        first_observed = observed_dates.min()
        last_observed = observed_dates.max()
        labels = np.full(len(group), "observed", dtype=object)
        missing_mask = group["missing_indicator"].eq(1).to_numpy()
        date_values = group["date"].to_numpy()
        labels[missing_mask & (date_values < first_observed)] = "start_missing"
        labels[missing_mask & (date_values > last_observed)] = "end_missing"
        labels[missing_mask & (date_values >= first_observed) & (date_values <= last_observed)] = "middle_missing"
        output.loc[group.index] = labels

    return output.reindex(sample.index)


def _run_signal_phase2(
    panel: pd.DataFrame,
    signal: str,
    x_obs_columns: Sequence[str],
    n_draws: int,
    random_seed: int,
    n_folds: int,
    mar_draw_regime: str,
    benchmark_specs: Sequence[Phase2BenchmarkSpec],
    include_pattern_slices: bool,
) -> SignalPhase2Result | None:
    stage1_rows: list[dict[str, float | int | str]] = []
    jtest_rows: list[dict[str, float | int | str]] = []
    benchmark_rows: list[dict[str, float | int | str]] = []
    pattern_slice_rows: list[dict[str, float | int | str]] = []
    draw_aggregation_rows: list[dict[str, float | int | str]] = []
    moment_contribution_rows: list[dict[str, float | int | str]] = []

    for benchmark_spec in benchmark_specs:
        benchmark_result = _run_signal_benchmark_phase2(
            panel=panel,
            signal=signal,
            x_obs_columns=x_obs_columns,
            n_draws=n_draws,
            random_seed=random_seed,
            n_folds=n_folds,
            mar_draw_regime=mar_draw_regime,
            benchmark_spec=benchmark_spec,
            include_pattern_slices=include_pattern_slices,
        )
        if benchmark_result is None:
            continue
        stage1_rows.append(benchmark_result["stage1_row"])
        jtest_rows.append(benchmark_result["jtest_row"])
        benchmark_rows.extend(benchmark_result["benchmark_rows"])
        pattern_slice_rows.extend(benchmark_result["pattern_slice_rows"])
        draw_aggregation_rows.extend(benchmark_result["draw_aggregation_rows"])
        moment_contribution_rows.extend(benchmark_result["moment_contribution_rows"])

    if not stage1_rows:
        return None
    return SignalPhase2Result(
        stage1_rows=stage1_rows,
        jtest_rows=jtest_rows,
        benchmark_rows=benchmark_rows,
        pattern_slice_rows=pattern_slice_rows,
        draw_aggregation_rows=draw_aggregation_rows,
        moment_contribution_rows=moment_contribution_rows,
    )


def _run_signal_benchmark_phase2(
    panel: pd.DataFrame,
    signal: str,
    x_obs_columns: Sequence[str],
    n_draws: int,
    random_seed: int,
    n_folds: int,
    mar_draw_regime: str,
    benchmark_spec: Phase2BenchmarkSpec,
    include_pattern_slices: bool,
) -> dict[str, object] | None:
    working_panel = panel.copy()
    signal_x_obs_columns = list(x_obs_columns)
    if benchmark_spec.augment_signal_history:
        working_panel, signal_x_obs_columns = _augment_phase2_signal_history_features(
            working_panel,
            signal,
            signal_x_obs_columns,
        )
    sample, _, _ = build_signal_mar_panel(
        working_panel,
        signal,
        signal_x_obs_columns,
        require_return=False,
    )
    sample = sample.sort_values(["date", "permno"]).copy()
    signal_col = f"{signal}_tier1"

    eval_sample, draw_store = build_forward_step1a_evaluation_panel(
        sample=sample,
        signal=signal,
        x_obs_columns=signal_x_obs_columns,
        n_folds=n_folds,
        n_draws=n_draws,
        random_seed=random_seed,
        mar_draw_regime=mar_draw_regime,
    )
    if eval_sample.empty or not draw_store:
        return None

    observed_eval = eval_sample.loc[eval_sample["observed_indicator"].eq(1)].copy()
    instrument_frame = build_signal_instrument_frame(
        eval_sample,
        signal_x_obs_columns,
        conditional_mean_col="x_cond_mean_cf",
    )
    eval_sample["phase2_pattern_slice"] = _classify_phase2_missing_pattern_slice(eval_sample)

    stage1_draw_results: list[dict[str, float | int]] = []
    jtest_draw_results: list[dict[str, float | int]] = []
    contribution_rows: list[dict[str, float | int | str]] = []
    mar_draw_sum = np.zeros(len(eval_sample), dtype=float)

    for draw_id, mar_draw_values in sorted(draw_store.items()):
        mar_draw_sum += mar_draw_values
        stage1_draw_results.append(
            _run_stage1_draw(
                sample=eval_sample,
                signal_col=signal_col,
                mar_draw_values=mar_draw_values,
                x_obs_columns=signal_x_obs_columns,
            )
        )
        jtest_result = _run_jtest_draw(
            sample=eval_sample,
            instrument_frame=instrument_frame,
            signal_col=signal_col,
            mar_draw_values=mar_draw_values,
        )
        jtest_draw_results.append(jtest_result)
        contribution_rows.extend(
            _build_moment_contribution_rows(
                signal=signal,
                benchmark_type=benchmark_spec.benchmark_type,
                draw_id=draw_id,
                jtest_result=jtest_result,
            )
        )

    realized_draw_count = len(draw_store)
    mar_draw_mean = mar_draw_sum / max(realized_draw_count, 1)
    projection_r_squared = _prediction_r_squared(
        observed_eval[signal_col].to_numpy(dtype=float),
        observed_eval["x_cond_mean_cf"].to_numpy(dtype=float),
    )
    stage1_summary = _combine_stage1_results(stage1_draw_results)
    jtest_summary = _combine_jtest_results(jtest_draw_results)

    stage1_row = {
        "signal": signal,
        "benchmark_type": benchmark_spec.benchmark_type,
        "n_eligible_with_x_obs": int(len(sample)),
        "n_forward_eval_obs": int(len(eval_sample)),
        "n_observed_signal": int(eval_sample["observed_indicator"].sum()),
        "n_missing_signal": int(eval_sample["missing_indicator"].sum()),
        "imputation_r_squared": projection_r_squared,
        "x_obs_column_count": len(signal_x_obs_columns),
        "weighted_jtest_flag": True,
        "signal_history_augmented": benchmark_spec.augment_signal_history,
        **stage1_summary,
    }
    jtest_row = {
        "signal": signal,
        "benchmark_type": benchmark_spec.benchmark_type,
        "n_obs": int(len(eval_sample)),
        "n_missing_signal": int(eval_sample["missing_indicator"].sum()),
        "x_obs_column_count": len(signal_x_obs_columns),
        "weighted_jtest_flag": True,
        "signal_history_augmented": benchmark_spec.augment_signal_history,
        "observed_variance_mean": float(eval_sample["x_obs_resid_variance_cf"].mean()),
        "missing_variance_mean": float(eval_sample["x_missing_resid_variance_cf"].mean()),
        **jtest_summary,
    }
    benchmark_rows = _build_signal_benchmark_comparison_rows(
        signal=signal,
        benchmark_type=benchmark_spec.benchmark_type,
        sample=eval_sample,
        x_obs_columns=signal_x_obs_columns,
        mar_draw_mean=mar_draw_mean,
        mar_draw_regime=mar_draw_regime,
    )
    pattern_slice_rows = (
        _build_pattern_slice_rows(
            signal=signal,
            benchmark_type=benchmark_spec.benchmark_type,
            sample=eval_sample,
            instrument_frame=instrument_frame,
            signal_col=signal_col,
            x_obs_columns=signal_x_obs_columns,
            draw_store=draw_store,
        )
        if include_pattern_slices
        else []
    )
    draw_aggregation_rows = [
        _build_draw_aggregation_row(
            signal=signal,
            benchmark_type=benchmark_spec.benchmark_type,
            jtest_draw_results=jtest_draw_results,
        )
    ]
    return {
        "stage1_row": stage1_row,
        "jtest_row": jtest_row,
        "benchmark_rows": benchmark_rows,
        "pattern_slice_rows": pattern_slice_rows,
        "draw_aggregation_rows": draw_aggregation_rows,
        "moment_contribution_rows": contribution_rows,
    }


def build_forward_step1a_evaluation_panel(
    sample: pd.DataFrame,
    signal: str,
    x_obs_columns: Sequence[str],
    n_folds: int,
    n_draws: int,
    random_seed: int,
    mar_draw_regime: str,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    unique_dates = pd.Index(sample["date"].dropna().sort_values().unique())
    if unique_dates.empty:
        return pd.DataFrame(), {}

    fold_count = max(1, min(int(n_folds), len(unique_dates)))
    fold_dates = np.array_split(unique_dates.to_numpy(), fold_count)
    eval_frames: list[pd.DataFrame] = []
    draw_store: dict[int, list[np.ndarray]] = {draw_id: [] for draw_id in range(1, n_draws + 1)}
    signal_col = f"{signal}_tier1"

    for fold_id, test_dates in enumerate(fold_dates, start=1):
        test_date_index = pd.Index(test_dates)
        if test_date_index.empty:
            continue
        test_start = test_date_index.min()
        train_mask = sample["date"] < test_start
        test_mask = sample["date"].isin(test_date_index)
        train_observed = sample.loc[
            train_mask & sample["observed_indicator"].eq(1),
            [signal_col, *x_obs_columns],
        ].dropna()
        if train_observed.empty:
            continue

        fold_frame = sample.loc[test_mask].copy()
        if fold_frame.empty:
            continue

        projection_bundle = fit_fast_ols(train_observed, signal_col, x_obs_columns)
        fold_frame["x_cond_mean_cf"] = predict_fast_ols(projection_bundle, fold_frame)
        fold_frame["forward_fold_id"] = int(fold_id)
        fold_frame["forward_train_observed_count"] = int(len(train_observed))
        fold_frame["x_uncond_mean_forward"] = float(train_observed[signal_col].mean())
        train_cond_mean = predict_fast_ols(projection_bundle, train_observed)
        observed_variance, missing_variance = _estimate_forward_residual_variances(
            train_observed=train_observed,
            train_cond_mean=train_cond_mean,
            test_cond_mean=fold_frame["x_cond_mean_cf"].to_numpy(dtype=float),
        )
        fold_frame["x_obs_resid_variance_cf"] = observed_variance
        fold_frame["x_missing_resid_variance_cf"] = missing_variance

        imputation_bundle = fit_counterfactual_imputation_bundle(
            observed_train=train_observed,
            signal_col=signal_col,
            x_obs_columns=x_obs_columns,
            quantile_bins=5,
        )
        for draw_id in range(1, n_draws + 1):
            rng = np.random.default_rng(_draw_seed(random_seed, signal, mar_draw_regime, draw_id, fold_id))
            draw_frame = build_stochastic_counterfactual_column(
                fold_frame,
                signal,
                imputation_bundle,
                mar_draw_regime,
                rng,
            )
            draw_store[draw_id].append(draw_frame[_draw_output_column(mar_draw_regime)].to_numpy(dtype=float))

        eval_frames.append(fold_frame)

    if not eval_frames:
        return pd.DataFrame(), {}

    evaluation_sample = pd.concat(eval_frames, ignore_index=True)
    final_draw_store = {
        draw_id: np.concatenate(draw_arrays)
        for draw_id, draw_arrays in draw_store.items()
        if draw_arrays
    }
    return evaluation_sample, final_draw_store


def _run_stage1_draw(
    sample: pd.DataFrame,
    signal_col: str,
    mar_draw_values: np.ndarray,
    x_obs_columns: Sequence[str],
) -> dict[str, float | int]:
    working = sample[[*x_obs_columns, "missing_indicator"]].copy()
    mar_series = pd.Series(mar_draw_values, index=sample.index)
    working["stage1_signal_value"] = sample[signal_col].where(
        sample["observed_indicator"].eq(1),
        mar_series,
    )
    return run_stage1_regression_diagnostic(
        working,
        x_obs_columns,
        outcome_col="stage1_signal_value",
    )


def _run_jtest_draw(
    sample: pd.DataFrame,
    instrument_frame: pd.DataFrame,
    signal_col: str,
    mar_draw_values: np.ndarray,
) -> dict[str, float | int]:
    working = sample[[signal_col, "x_cond_mean_cf", "observed_indicator", "missing_indicator"]].copy()
    working["x_mar_draw"] = mar_draw_values
    return run_signal_mar_jtest(
        sample=working,
        instrument_frame=instrument_frame,
        signal_col=signal_col,
        conditional_mean_col="x_cond_mean_cf",
        mar_draw_col="x_mar_draw",
        observed_variance_col="x_obs_resid_variance_cf",
        missing_variance_col="x_missing_resid_variance_cf",
    )


def _build_pattern_slice_rows(
    signal: str,
    benchmark_type: str,
    sample: pd.DataFrame,
    instrument_frame: pd.DataFrame,
    signal_col: str,
    x_obs_columns: Sequence[str],
    draw_store: dict[int, np.ndarray],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    slice_names = ["full_sample", "start_missing", "middle_missing", "end_missing", "always_missing"]

    for slice_name in slice_names:
        if slice_name == "full_sample":
            slice_sample = sample.copy()
            slice_instruments = instrument_frame.copy()
        else:
            slice_mask = sample["observed_indicator"].eq(1) | sample["phase2_pattern_slice"].eq(slice_name)
            slice_sample = sample.loc[slice_mask].copy()
            slice_instruments = instrument_frame.loc[slice_mask].copy()
        if slice_sample.empty or slice_sample["missing_indicator"].sum() == 0 or slice_sample["observed_indicator"].sum() == 0:
            continue

        stage1_draw_results: list[dict[str, float | int]] = []
        jtest_draw_results: list[dict[str, float | int]] = []
        for _, mar_draw_values in sorted(draw_store.items()):
            draw_subset = mar_draw_values[slice_sample.index.to_numpy(dtype=int)]
            stage1_draw_results.append(
                _run_stage1_draw(
                    sample=slice_sample,
                    signal_col=signal_col,
                    mar_draw_values=draw_subset,
                    x_obs_columns=x_obs_columns,
                )
            )
            jtest_draw_results.append(
                _run_jtest_draw(
                    sample=slice_sample,
                    instrument_frame=slice_instruments,
                    signal_col=signal_col,
                    mar_draw_values=draw_subset,
                )
            )

        rows.append(
            {
                "signal": signal,
                "benchmark_type": benchmark_type,
                "slice_name": slice_name,
                "n_obs": int(len(slice_sample)),
                "n_observed_signal": int(slice_sample["observed_indicator"].sum()),
                "n_missing_signal": int(slice_sample["missing_indicator"].sum()),
                "stage1_p_value": _combine_stage1_results(stage1_draw_results)["stage1_p_value"],
                "j_p_value": _combine_jtest_results(jtest_draw_results)["j_p_value"],
                "j_stat": _combine_jtest_results(jtest_draw_results)["j_stat"],
                "observed_variance_mean": float(slice_sample["x_obs_resid_variance_cf"].mean()),
                "missing_variance_mean": float(slice_sample["x_missing_resid_variance_cf"].mean()),
            }
        )
    return rows


def _combine_draw_test_results(
    draw_results: Sequence[dict[str, float | int]],
    pvalue_col: str,
) -> tuple[pd.DataFrame, list[float], float, float]:
    frame = pd.DataFrame(draw_results)
    p_values = frame[pvalue_col].dropna().tolist()
    combined_p, stouffer_z = _stouffer_combine_pvalues(p_values)
    return frame, p_values, combined_p, stouffer_z


def _combine_stage1_results(draw_results: Sequence[dict[str, float | int]]) -> dict[str, float | int]:
    frame, p_values, combined_p, stouffer_z = _combine_draw_test_results(
        draw_results,
        pvalue_col="stage1_p_value",
    )
    return {
        "aux_n_obs": int(frame["aux_n_obs"].iloc[0]) if not frame.empty else 0,
        "aux_r_squared": float(frame["aux_r_squared"].mean()) if not frame.empty else np.nan,
        "stage1_f_stat": float(frame["stage1_f_stat"].mean()) if not frame.empty else np.nan,
        "stage1_p_value": combined_p,
        "stage1_draw_count": int(len(frame)),
        "stage1_mean_draw_p_value": float(np.mean(p_values)) if p_values else np.nan,
        "stage1_p_lt_0_05_share": float(np.mean(np.asarray(p_values) < 0.05)) if p_values else np.nan,
        "stage1_stouffer_z": stouffer_z,
        "tested_restriction_count": int(frame["tested_restriction_count"].iloc[0]) if not frame.empty else 0,
        "missing_indicator_coef": float(frame["missing_indicator_coef"].mean()) if not frame.empty else np.nan,
        "missing_indicator_pvalue": float(frame["missing_indicator_pvalue"].mean()) if not frame.empty else np.nan,
    }


def _combine_jtest_results(draw_results: Sequence[dict[str, float | int]]) -> dict[str, float | int]:
    frame, p_values, combined_p, stouffer_z = _combine_draw_test_results(
        draw_results,
        pvalue_col="j_p_value",
    )
    return {
        "j_stat": float(frame["j_stat"].mean()) if not frame.empty else np.nan,
        "j_p_value": combined_p,
        "j_draw_count": int(len(frame)),
        "j_mean_draw_p_value": float(np.mean(p_values)) if p_values else np.nan,
        "j_p_lt_0_05_share": float(np.mean(np.asarray(p_values) < 0.05)) if p_values else np.nan,
        "j_stouffer_z": stouffer_z,
        "j_df": int(frame["j_df"].iloc[0]) if not frame.empty else 0,
        "j_moment_count": int(frame["j_moment_count"].iloc[0]) if not frame.empty else 0,
        "instrument_count": int(frame["instrument_count"].iloc[0]) if not frame.empty else 0,
        "observed_moment_norm": float(frame["observed_moment_norm"].mean()) if not frame.empty else np.nan,
        "missing_moment_norm": float(frame["missing_moment_norm"].mean()) if not frame.empty else np.nan,
        "stacked_moment_norm": float(frame["stacked_moment_norm"].mean()) if not frame.empty else np.nan,
        "observed_j_component": float(frame["observed_j_component"].mean()) if not frame.empty else np.nan,
        "missing_j_component": float(frame["missing_j_component"].mean()) if not frame.empty else np.nan,
    }


def _build_signal_benchmark_comparison_rows(
    signal: str,
    benchmark_type: str,
    sample: pd.DataFrame,
    x_obs_columns: Sequence[str],
    mar_draw_mean: np.ndarray,
    mar_draw_regime: str,
) -> list[dict[str, float | int | str]]:
    signal_col = f"{signal}_tier1"
    observed_mask = sample["observed_indicator"].eq(1)
    rows: list[dict[str, float | int | str]] = []
    benchmark_series = {
        "complete_case": sample.loc[observed_mask, signal_col],
        "unconditional_mean": sample[signal_col].where(observed_mask, sample["x_uncond_mean_forward"]),
        "conditional_mean": sample[signal_col].where(observed_mask, sample["x_cond_mean_cf"]),
        f"{mar_draw_regime}_mean": sample[signal_col].where(
            observed_mask,
            pd.Series(mar_draw_mean, index=sample.index),
        ),
    }

    for benchmark_name, values in benchmark_series.items():
        benchmark_frame = sample[[*x_obs_columns]].copy()
        benchmark_frame["signal_value"] = values
        benchmark_frame = benchmark_frame.dropna(subset=["signal_value", *x_obs_columns]).copy()
        if benchmark_frame.empty:
            rows.append(
                {
                    "signal": signal,
                    "benchmark_type": benchmark_type,
                    "benchmark": benchmark_name,
                    "n_obs": 0,
                    "projection_r_squared": np.nan,
                    "outcome_mean": np.nan,
                    "filled_missing_count": 0,
                }
            )
            continue

        model = fit_fast_ols(benchmark_frame, "signal_value", x_obs_columns)
        predicted = predict_fast_ols(model, benchmark_frame)
        rows.append(
            {
                "signal": signal,
                "benchmark_type": benchmark_type,
                "benchmark": benchmark_name,
                "n_obs": int(len(benchmark_frame)),
                "projection_r_squared": _prediction_r_squared(
                    benchmark_frame["signal_value"].to_numpy(dtype=float),
                    predicted,
                ),
                "outcome_mean": float(benchmark_frame["signal_value"].mean()),
                "filled_missing_count": int((~observed_mask).sum()) if benchmark_name != "complete_case" else 0,
            }
        )

    return rows


def _build_draw_aggregation_row(
    signal: str,
    benchmark_type: str,
    jtest_draw_results: Sequence[dict[str, float | int]],
) -> dict[str, float | int | str]:
    if not jtest_draw_results:
        return {
            "signal": signal,
            "benchmark_type": benchmark_type,
            "draw_count": 0,
            "J_mean": np.nan,
            "J_median": np.nan,
            "J_p05": np.nan,
            "J_p95": np.nan,
            "p_mean": np.nan,
            "p_median": np.nan,
            "reject_share": np.nan,
            "stouffer_p": np.nan,
            "J_avg_moment": np.nan,
            "p_avg_moment": np.nan,
        }
    frame = pd.DataFrame(jtest_draw_results)
    j_values = frame["j_stat"].to_numpy(dtype=float)
    p_values = frame["j_p_value"].to_numpy(dtype=float)
    avg_moment_result = _average_moment_jtest(jtest_draw_results)
    return {
        "signal": signal,
        "benchmark_type": benchmark_type,
        "draw_count": int(len(frame)),
        "J_mean": float(np.nanmean(j_values)),
        "J_median": float(np.nanmedian(j_values)),
        "J_p05": float(np.nanpercentile(j_values, 5)),
        "J_p95": float(np.nanpercentile(j_values, 95)),
        "p_mean": float(np.nanmean(p_values)),
        "p_median": float(np.nanmedian(p_values)),
        "reject_share": float(np.nanmean(p_values < 0.05)),
        "stouffer_p": _stouffer_combine_pvalues([p for p in p_values.tolist() if np.isfinite(p)])[0],
        "J_avg_moment": avg_moment_result["j_stat"],
        "p_avg_moment": avg_moment_result["j_p_value"],
    }


def _average_moment_jtest(
    jtest_draw_results: Sequence[dict[str, float | int]],
) -> dict[str, float]:
    if not jtest_draw_results:
        return {"j_stat": np.nan, "j_p_value": np.nan}
    labels = list(jtest_draw_results[0].get("moment_labels", []))
    if not labels:
        return {"j_stat": np.nan, "j_p_value": np.nan}
    moment_matrix = np.asarray(
        [result["moment_means"] for result in jtest_draw_results],
        dtype=float,
    )
    avg_cov = np.cov(moment_matrix, rowvar=False)
    avg_cov = np.atleast_2d(avg_cov)
    covariance_inv = np.linalg.pinv(avg_cov) if avg_cov.size else np.array([[np.nan]])
    gbar = moment_matrix.mean(axis=0)
    n_draws = moment_matrix.shape[0]
    j_stat = float(n_draws * gbar.T @ covariance_inv @ gbar) if avg_cov.size else np.nan
    j_df = int(np.linalg.matrix_rank(avg_cov)) if avg_cov.size else 0
    from scipy.stats import chi2
    p_value = float(1.0 - chi2.cdf(j_stat, j_df)) if j_df > 0 else np.nan
    return {"j_stat": j_stat, "j_p_value": p_value}


def _build_moment_contribution_rows(
    signal: str,
    benchmark_type: str,
    draw_id: int,
    jtest_result: dict[str, float | int],
) -> list[dict[str, float | int | str]]:
    labels = list(jtest_result.get("moment_labels", []))
    means = list(jtest_result.get("moment_means", []))
    standardized = list(jtest_result.get("standardized_moments", []))
    contributions = list(jtest_result.get("j_contributions", []))
    rows: list[dict[str, float | int | str]] = []
    for label, mean_value, std_value, contribution in zip(labels, means, standardized, contributions):
        moment_family, _, instrument = str(label).partition("::")
        rows.append(
            {
                "signal": signal,
                "benchmark_type": benchmark_type,
                "draw_id": int(draw_id),
                "moment_family": moment_family,
                "instrument": instrument,
                "moment_value": float(mean_value),
                "standardized_moment": float(std_value),
                "j_contribution": float(contribution),
            }
        )
    return rows


def _prediction_r_squared(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    total_ss = float(((actual - actual.mean()) ** 2).sum())
    if total_ss <= 0:
        return np.nan
    residual_ss = float(((actual - predicted) ** 2).sum())
    return float(1.0 - residual_ss / total_ss)


def _stouffer_combine_pvalues(pvalues: Sequence[float]) -> tuple[float, float]:
    if not pvalues:
        return np.nan, np.nan
    clipped = np.clip(np.asarray(pvalues, dtype=float), 1e-15, 1.0 - 1e-15)
    z_stat = float(norm.ppf(1.0 - clipped).sum() / np.sqrt(len(clipped)))
    combined_p = float(1.0 - norm.cdf(z_stat))
    return combined_p, z_stat


def _draw_seed(
    random_seed: int,
    signal: str,
    regime: str,
    draw_id: int,
    fold_id: int | None = None,
) -> int:
    fold_component = fold_id if fold_id is not None else 0
    seed_bytes = f"{random_seed}|{signal}|{regime}|{draw_id}|{fold_component}".encode("utf-8")
    return int(np.frombuffer(seed_bytes, dtype=np.uint8).sum()) + draw_id * 7919 + fold_component * 104729


def _draw_output_column(regime: str) -> str:
    if regime == "conditional_quantile_draw":
        return "x_conditional_quantile_draw"
    if regime == "residual_bootstrap":
        return "x_residual_bootstrap"
    raise ValueError(f"Unsupported Phase 2 MAR draw regime: {regime}")


def write_phase2_outputs(outputs: dict[str, pd.DataFrame], output_root: Path) -> None:
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for output_name, filename in STEP1A_TABLE_FILENAMES.items():
        outputs[output_name].to_csv(tables_dir / filename, index=False)
    for output_name, filename in LEGACY_TABLE_ALIASES.items():
        outputs[output_name].to_csv(tables_dir / filename, index=False)

    plot_jtest_pvalue_distribution(
        outputs["jtest_table"],
        figures_dir / "jtest_pvalue_distribution.png",
    )
