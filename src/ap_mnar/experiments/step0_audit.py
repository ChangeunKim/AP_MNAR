from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from ap_mnar.data.load_raw import load_firm_panel, load_signal_panel
from ap_mnar.data.merge_panel import (
    build_canonical_panel,
    build_id_merge_diagnostics,
    filter_out_of_universe_permnos,
)
from ap_mnar.data.metadata import load_signal_metadata
from ap_mnar.data.target import add_forward_return
from ap_mnar.reporting.tables import (
    build_eligibility_filter_impact_table,
    build_missingness_before_after_table,
    build_signal_coverage_table,
    build_x_obs_availability_audit,
    build_summary_statistics_table,
)
from ap_mnar.reporting.figures import plot_missingness_heatmap, plot_target_distribution


DEFAULT_SIGNALS = (
    "AnalystRevision",
    "REV6",
    "ForecastDispersion",
    "FEPS",
)
DEFAULT_ANALYST_START_DATE = pd.Timestamp("1988-01-01")


@dataclass(frozen=True)
class Phase0Paths:
    signal_panel_path: Path
    firm_panel_path: Path
    signal_metadata_path: Path
    output_root: Path


def run_phase0_audit(
    paths: Phase0Paths,
    signals: Sequence[str] = DEFAULT_SIGNALS,
    x_obs_columns: Sequence[str] = (),
    analyst_start_date: pd.Timestamp = DEFAULT_ANALYST_START_DATE,
    firm_nrows: int | None = None,
    signal_nrows: int | None = None,
) -> dict[str, pd.DataFrame]:
    signal_registry = load_signal_metadata(paths.signal_metadata_path, signals)
    firm_panel = load_firm_panel(paths.firm_panel_path, nrows=firm_nrows)
    firm_panel = add_forward_return(firm_panel, return_col="ret", horizon=1).rename(columns={"ret_fwd_1m": "ret_fwd_1m"})
    firm_panel = firm_panel.loc[firm_panel["date"] >= analyst_start_date].copy()
    signal_panel = load_signal_panel(paths.signal_panel_path, signals=signals, nrows=signal_nrows)
    signal_panel = signal_panel.loc[signal_panel["date"] >= analyst_start_date].copy()
    canonical_panel = build_canonical_panel(firm_panel, signal_panel, signal_registry)
    canonical_panel, out_of_universe_diagnostics = filter_out_of_universe_permnos(canonical_panel, signals)

    outputs = {
        "signal_registry": signal_registry,
        "panel_base": canonical_panel,
        "signal_coverage": build_signal_coverage_table(canonical_panel, signals),
        "id_merge_diagnostics": build_id_merge_diagnostics(
            firm_panel,
            signal_panel,
            canonical_panel,
            signals,
            extra_metrics=out_of_universe_diagnostics,
        ),
        "x_obs_availability_audit": build_x_obs_availability_audit(canonical_panel, x_obs_columns),
        "missingness_before_after_lag_adjustment": build_missingness_before_after_table(canonical_panel, signals),
        "eligibility_filter_impact": build_eligibility_filter_impact_table(canonical_panel, signals),
        "summary_statistics": build_summary_statistics_table(canonical_panel, signals),
    }
    write_phase0_outputs(outputs, paths.output_root)
    return outputs


def write_phase0_outputs(outputs: dict[str, pd.DataFrame], output_root: Path) -> None:
    interim_dir = output_root / "interim"
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    interim_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    outputs["panel_base"].to_parquet(interim_dir / "panel_base.parquet", index=False)
    outputs["signal_registry"].to_csv(tables_dir / "signal_registry.csv", index=False)
    outputs["signal_coverage"].to_csv(tables_dir / "signal_coverage.csv", index=False)
    outputs["id_merge_diagnostics"].to_csv(tables_dir / "id_merge_diagnostics.csv", index=False)
    outputs["x_obs_availability_audit"].to_csv(tables_dir / "x_obs_availability_audit.csv", index=False)
    outputs["missingness_before_after_lag_adjustment"].to_csv(
        tables_dir / "missingness_before_after_lag_adjustment.csv",
        index=False,
    )
    outputs["eligibility_filter_impact"].to_csv(tables_dir / "eligibility_filter_impact.csv", index=False)
    outputs["summary_statistics"].to_csv(tables_dir / "summary_statistics.csv", index=False)

    plot_missingness_heatmap(outputs["panel_base"], list(outputs["signal_registry"]["signal"].unique()), figures_dir / "missingness_heatmap.png")
    plot_target_distribution(outputs["panel_base"], figures_dir / "target_distribution.png")
