from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from ap_mnar.data.x_obs import load_x_obs_spec, merge_x_obs_columns
from ap_mnar.experiments.step0_audit import DEFAULT_SIGNALS
from ap_mnar.models.benchmark_variants import (
    augment_signal_history_features,
    get_signal_benchmark_specs,
)
from ap_mnar.pricing.design import (
    PricingSpecification,
    build_coverage_subsamples,
    build_pricing_specifications,
    build_signal_pricing_panel,
)
from ap_mnar.pricing.diagnostics import (
    build_coverage_decomposition_table,
    build_missingness_premium_time_series,
    run_pooled_pricing_regression,
)
from ap_mnar.pricing.fama_macbeth import run_fama_macbeth_regression
from ap_mnar.reporting.figures import (
    plot_missingness_alpha_by_signal,
    plot_missingness_premium_over_time,
)

try:
    from tqdm.auto import tqdm

    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in minimal environments
    TQDM_AVAILABLE = False


@dataclass(frozen=True)
class Phase4Paths:
    panel_with_missingness_path: Path
    x_obs_config_path: Path
    firm_panel_path: Path
    output_root: Path


def run_phase4_pricing(
    paths: Phase4Paths,
    signals: Sequence[str] = DEFAULT_SIGNALS,
    min_train_months: int = 60,
    min_cross_section: int = 25,
    nw_lags: int = 6,
    show_progress: bool = True,
    include_augmented_signal_history: bool = True,
) -> dict[str, pd.DataFrame]:
    panel = pd.read_parquet(paths.panel_with_missingness_path)
    x_obs_spec = load_x_obs_spec(paths.x_obs_config_path)
    panel = merge_x_obs_columns(panel, paths.firm_panel_path, x_obs_spec.columns)

    benchmark_specs = get_signal_benchmark_specs(include_augmented_signal_history)
    pricing_panels: list[pd.DataFrame] = []
    pooled_tables: list[pd.DataFrame] = []
    fama_macbeth_tables: list[pd.DataFrame] = []
    monthly_coef_tables: list[pd.DataFrame] = []

    for signal in _progress_iter(signals, show_progress, desc="Phase 4 signals", unit="signal"):
        for benchmark_spec in benchmark_specs:
            benchmark_panel = panel.copy()
            benchmark_x_obs = list(x_obs_spec.columns)
            if benchmark_spec.augment_signal_history:
                benchmark_panel, benchmark_x_obs = augment_signal_history_features(
                    benchmark_panel,
                    signal,
                    benchmark_x_obs,
                )
            specifications = build_pricing_specifications(benchmark_x_obs)
            pricing_panel = build_signal_pricing_panel(
                panel=benchmark_panel,
                signal=signal,
                x_obs_columns=benchmark_x_obs,
                benchmark_type=benchmark_spec.benchmark_type,
                min_train_months=min_train_months,
            )
            pricing_panels.append(pricing_panel)

            subsamples = build_coverage_subsamples(pricing_panel)
            subsample_iter = _progress_iter(
                list(subsamples.items()),
                show_progress,
                desc=f"{signal} {benchmark_spec.benchmark_type}: subsamples",
                unit="sample",
                leave=False,
            )
            for sample_name, sample_frame in subsample_iter:
                if sample_frame.empty:
                    continue
                spec_iter = _progress_iter(
                    specifications,
                    show_progress,
                    desc=f"{signal} {benchmark_spec.benchmark_type} {sample_name}: specs",
                    unit="spec",
                    leave=False,
                )
                for specification in spec_iter:
                    pooled_table, fama_macbeth_table, monthly_coef_table = run_signal_specification_suite(
                        sample_frame=sample_frame,
                        signal=signal,
                        benchmark_type=benchmark_spec.benchmark_type,
                        specification=specification,
                        min_cross_section=min_cross_section,
                        nw_lags=nw_lags,
                    )
                    if not pooled_table.empty:
                        pooled_tables.append(pooled_table)
                    if not fama_macbeth_table.empty:
                        fama_macbeth_tables.append(fama_macbeth_table)
                    if not monthly_coef_table.empty:
                        monthly_coef_tables.append(monthly_coef_table)

    pricing_panel_table = _concat_frames(pricing_panels)
    pooled_pricing_table = _concat_frames(pooled_tables)
    fama_macbeth_table = _concat_frames(fama_macbeth_tables)
    monthly_coef_table = _concat_frames(monthly_coef_tables)
    coverage_decomposition_table = build_coverage_decomposition_table(fama_macbeth_table)
    premium_time_series = build_missingness_premium_time_series(monthly_coef_table)
    h4a_table = build_h4_channel_table(fama_macbeth_table, "baseline_missing_only", "H4a")
    h4b_table = build_h4_channel_table(fama_macbeth_table, "mar_signal_only", "H4b")
    h4c_table = build_h4_joint_channel_table(fama_macbeth_table)
    coverage_channel_table = build_h4_coverage_channel_table(fama_macbeth_table)

    outputs = {
        "pricing_panel": pricing_panel_table,
        "pooled_pricing_table": pooled_pricing_table,
        "fama_macbeth_table": fama_macbeth_table,
        "coverage_decomposition_table": coverage_decomposition_table,
        "premium_time_series": premium_time_series,
        "h4a_table": h4a_table,
        "h4b_table": h4b_table,
        "h4c_table": h4c_table,
        "coverage_channel_table": coverage_channel_table,
    }
    write_phase4_outputs(outputs, paths.output_root)
    return outputs


def run_signal_specification_suite(
    sample_frame: pd.DataFrame,
    signal: str,
    benchmark_type: str,
    specification: PricingSpecification,
    min_cross_section: int,
    nw_lags: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_name = sample_frame.attrs.get("sample_name", "unknown")
    pooled_table = run_pooled_pricing_regression(
        frame=sample_frame,
        feature_columns=specification.feature_columns,
        signal=signal,
        benchmark_type=benchmark_type,
        specification=specification.name,
        sample_name=sample_name,
        focus_columns=specification.focus_columns,
    )
    fama_macbeth_table, monthly_coef_table = run_fama_macbeth_regression(
        frame=sample_frame,
        feature_columns=specification.feature_columns,
        signal=signal,
        benchmark_type=benchmark_type,
        specification=specification.name,
        sample_name=sample_name,
        min_cross_section=min_cross_section,
        nw_lags=nw_lags,
        focus_columns=specification.focus_columns,
    )
    return pooled_table, fama_macbeth_table, monthly_coef_table


def write_phase4_outputs(outputs: dict[str, pd.DataFrame], output_root: Path) -> None:
    interim_dir = output_root / "interim"
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    interim_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    outputs["pricing_panel"].to_parquet(interim_dir / "phase4_pricing_panel.parquet", index=False)
    outputs["pooled_pricing_table"].to_csv(tables_dir / "missingness_pricing_results.csv", index=False)
    outputs["fama_macbeth_table"].to_csv(tables_dir / "missingness_fama_macbeth_results.csv", index=False)
    outputs["coverage_decomposition_table"].to_csv(
        tables_dir / "missingness_coverage_decomposition.csv",
        index=False,
    )
    outputs["h4a_table"].to_csv(tables_dir / "h4a_direct_missingness_pricing.csv", index=False)
    outputs["h4b_table"].to_csv(tables_dir / "h4b_recovered_signal_pricing.csv", index=False)
    outputs["h4c_table"].to_csv(tables_dir / "h4c_joint_missingness_signal_pricing.csv", index=False)
    outputs["coverage_channel_table"].to_csv(
        tables_dir / "h4_coverage_decomposition_by_channel.csv",
        index=False,
    )
    outputs["premium_time_series"].to_csv(
        tables_dir / "missingness_premium_time_series.csv",
        index=False,
    )

    plot_missingness_alpha_by_signal(
        _select_plot_benchmark(outputs["fama_macbeth_table"]),
        figures_dir / "missingness_alpha_by_signal.png",
    )
    plot_missingness_premium_over_time(
        _select_plot_benchmark(outputs["premium_time_series"]),
        figures_dir / "missingness_premium_over_time.png",
    )


def _concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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


def build_h4_channel_table(
    fama_macbeth_table: pd.DataFrame,
    specification: str,
    channel_label: str,
) -> pd.DataFrame:
    if fama_macbeth_table.empty:
        return pd.DataFrame()
    focus = fama_macbeth_table.loc[
        fama_macbeth_table["specification"].eq(specification)
        & fama_macbeth_table["sample_name"].eq("full")
        & fama_macbeth_table["is_focus_regressor"].astype(bool)
    ].copy()
    if focus.empty:
        return pd.DataFrame()

    if channel_label == "H4a":
        focus = focus.loc[focus["coefficient"].eq("missing_indicator")].copy()
        return focus.rename(
            columns={
                "mean_coefficient": "coef_M",
                "t_stat": "t_M",
                "p_value": "p_M",
                "month_count": "n_months",
                "mean_cross_section_nobs": "avg_cross_section_n",
            }
        ).assign(control_set=focus["benchmark_type"])[
            [
                "signal",
                "benchmark_type",
                "control_set",
                "coef_M",
                "t_M",
                "p_M",
                "n_months",
                "avg_cross_section_n",
            ]
        ].assign(estimator="fama_macbeth")

    focus = focus.loc[focus["coefficient"].eq("x_mar_filled")].copy()
    return focus.rename(
        columns={
            "mean_coefficient": "coef_x_filled",
            "t_stat": "t_x_filled",
            "p_value": "p_x_filled",
            "month_count": "n_months",
            "mean_cross_section_nobs": "avg_cross_section_n",
        }
    ).assign(control_set=focus["benchmark_type"])[
        [
            "signal",
            "benchmark_type",
            "control_set",
            "coef_x_filled",
            "t_x_filled",
            "p_x_filled",
            "n_months",
            "avg_cross_section_n",
        ]
    ].assign(estimator="fama_macbeth")


def build_h4_joint_channel_table(
    fama_macbeth_table: pd.DataFrame,
) -> pd.DataFrame:
    if fama_macbeth_table.empty:
        return pd.DataFrame()
    focus = fama_macbeth_table.loc[
        fama_macbeth_table["specification"].eq("missing_plus_mar_signal")
        & fama_macbeth_table["sample_name"].eq("full")
        & fama_macbeth_table["is_focus_regressor"].astype(bool)
    ].copy()
    if focus.empty:
        return pd.DataFrame()

    pivot = focus.pivot_table(
        index=["signal", "benchmark_type"],
        columns="coefficient",
        values=["mean_coefficient", "t_stat", "p_value", "month_count", "mean_cross_section_nobs"],
        aggfunc="first",
    )
    if pivot.empty:
        return pd.DataFrame()
    pivot.columns = [f"{left}__{right}" for left, right in pivot.columns]
    pivot = pivot.reset_index()
    pivot["control_set"] = pivot["benchmark_type"]
    pivot["n_months"] = pivot.get("month_count__missing_indicator")
    pivot["avg_cross_section_n"] = pivot.get("mean_cross_section_nobs__missing_indicator")
    pivot["estimator"] = "fama_macbeth"
    return pivot.rename(
        columns={
            "mean_coefficient__missing_indicator": "coef_M",
            "t_stat__missing_indicator": "t_M",
            "p_value__missing_indicator": "p_M",
            "mean_coefficient__x_mar_filled": "coef_x_filled",
            "t_stat__x_mar_filled": "t_x_filled",
            "p_value__x_mar_filled": "p_x_filled",
        }
    )[
        [
            "signal",
            "benchmark_type",
            "control_set",
            "coef_M",
            "t_M",
            "p_M",
            "coef_x_filled",
            "t_x_filled",
            "p_x_filled",
            "n_months",
            "avg_cross_section_n",
            "estimator",
        ]
    ].sort_values(["signal", "benchmark_type"], ignore_index=True)


def build_h4_coverage_channel_table(
    fama_macbeth_table: pd.DataFrame,
) -> pd.DataFrame:
    if fama_macbeth_table.empty:
        return pd.DataFrame()
    focus = fama_macbeth_table.loc[fama_macbeth_table["is_focus_regressor"].astype(bool)].copy()
    if focus.empty:
        return pd.DataFrame()

    channel_map = {
        "baseline_missing_only": "direct_missingness",
        "mar_signal_only": "recovered_signal",
        "missing_plus_mar_signal": "joint",
    }
    focus["channel"] = focus["specification"].map(channel_map)
    focus = focus.dropna(subset=["channel"]).copy()
    return focus.rename(
        columns={
            "sample_name": "coverage_sample",
            "coefficient": "focus_regressor",
            "mean_coefficient": "coef",
            "t_stat": "tstat",
            "p_value": "pvalue",
            "month_count": "n_months",
            "mean_cross_section_nobs": "avg_cross_section_n",
        }
    )[
        [
            "signal",
            "channel",
            "benchmark_type",
            "coverage_sample",
            "focus_regressor",
            "coef",
            "tstat",
            "pvalue",
            "n_months",
            "avg_cross_section_n",
        ]
    ].sort_values(
        ["signal", "channel", "benchmark_type", "coverage_sample", "focus_regressor"],
        ignore_index=True,
    )


def _select_plot_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "benchmark_type" not in frame.columns:
        return frame
    subset = frame.loc[frame["benchmark_type"].eq("fixed_x_obs")].copy()
    return subset if not subset.empty else frame
