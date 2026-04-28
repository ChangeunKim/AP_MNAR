from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from ap_mnar.data.x_obs import load_x_obs_spec, merge_x_obs_columns
from ap_mnar.experiments.step0_audit import DEFAULT_SIGNALS
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
) -> dict[str, pd.DataFrame]:
    panel = pd.read_parquet(paths.panel_with_missingness_path)
    x_obs_spec = load_x_obs_spec(paths.x_obs_config_path)
    panel = merge_x_obs_columns(panel, paths.firm_panel_path, x_obs_spec.columns)

    specifications = build_pricing_specifications(x_obs_spec.columns)
    pricing_panels: list[pd.DataFrame] = []
    pooled_tables: list[pd.DataFrame] = []
    fama_macbeth_tables: list[pd.DataFrame] = []
    monthly_coef_tables: list[pd.DataFrame] = []

    for signal in _progress_iter(signals, show_progress, desc="Phase 4 signals", unit="signal"):
        pricing_panel = build_signal_pricing_panel(
            panel=panel,
            signal=signal,
            x_obs_columns=x_obs_spec.columns,
            min_train_months=min_train_months,
        )
        pricing_panels.append(pricing_panel)

        subsamples = build_coverage_subsamples(pricing_panel)
        subsample_iter = _progress_iter(
            list(subsamples.items()),
            show_progress,
            desc=f"{signal}: subsamples",
            unit="sample",
            leave=False,
        )
        for sample_name, sample_frame in subsample_iter:
            if sample_frame.empty:
                continue
            spec_iter = _progress_iter(
                specifications,
                show_progress,
                desc=f"{signal} {sample_name}: specs",
                unit="spec",
                leave=False,
            )
            for specification in spec_iter:
                pooled_table, fama_macbeth_table, monthly_coef_table = run_signal_specification_suite(
                    sample_frame=sample_frame,
                    signal=signal,
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

    outputs = {
        "pricing_panel": pricing_panel_table,
        "pooled_pricing_table": pooled_pricing_table,
        "fama_macbeth_table": fama_macbeth_table,
        "coverage_decomposition_table": coverage_decomposition_table,
        "premium_time_series": premium_time_series,
    }
    write_phase4_outputs(outputs, paths.output_root)
    return outputs


def run_signal_specification_suite(
    sample_frame: pd.DataFrame,
    signal: str,
    specification: PricingSpecification,
    min_cross_section: int,
    nw_lags: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_name = sample_frame.attrs.get("sample_name", "unknown")
    pooled_table = run_pooled_pricing_regression(
        frame=sample_frame,
        feature_columns=specification.feature_columns,
        signal=signal,
        specification=specification.name,
        sample_name=sample_name,
        focus_columns=specification.focus_columns,
    )
    fama_macbeth_table, monthly_coef_table = run_fama_macbeth_regression(
        frame=sample_frame,
        feature_columns=specification.feature_columns,
        signal=signal,
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
    outputs["premium_time_series"].to_csv(
        tables_dir / "missingness_premium_time_series.csv",
        index=False,
    )

    plot_missingness_alpha_by_signal(
        outputs["fama_macbeth_table"],
        figures_dir / "missingness_alpha_by_signal.png",
    )
    plot_missingness_premium_over_time(
        outputs["premium_time_series"],
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
