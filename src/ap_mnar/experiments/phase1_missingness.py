from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from ap_mnar.experiments.step0_audit import DEFAULT_SIGNALS
from ap_mnar.missingness.classify import build_missingness_panel
from ap_mnar.missingness.diagnostics import (
    build_missingness_pattern_counts,
    build_missingness_removed_by_eligibility,
    build_missingness_summary_by_signal,
    build_missingness_summary_by_year,
)
from ap_mnar.missingness.eligibility import (
    build_eligibility_matrix,
    compute_signal_support_windows,
    load_missingness_rules,
)
from ap_mnar.reporting.figures import plot_residual_missingness_heatmap


@dataclass(frozen=True)
class Phase1Paths:
    panel_base_path: Path
    signal_registry_path: Path
    missingness_rules_path: Path
    output_root: Path


def run_phase1_missingness(
    paths: Phase1Paths,
    signals: Sequence[str] = DEFAULT_SIGNALS,
) -> dict[str, pd.DataFrame]:
    panel = pd.read_parquet(paths.panel_base_path)
    signal_registry = pd.read_csv(paths.signal_registry_path)
    signal_registry = signal_registry.loc[signal_registry["signal"].isin(signals)].copy()
    signal_registry["tier1_delay_months"] = signal_registry["tier1_delay_months"].astype(int)

    rules = load_missingness_rules(paths.missingness_rules_path)
    support_windows = compute_signal_support_windows(panel, signal_registry, rules, signals)
    eligibility_matrix = build_eligibility_matrix(panel, support_windows, signals)
    panel_with_missingness = build_missingness_panel(panel, eligibility_matrix, signals)

    outputs = {
        "panel_with_missingness": panel_with_missingness,
        "signal_support_windows": support_windows,
        "missingness_summary_by_signal": build_missingness_summary_by_signal(
            panel_with_missingness,
            signal_registry,
            support_windows,
            signals,
        ),
        "missingness_summary_by_year": build_missingness_summary_by_year(panel_with_missingness, signals),
        "missingness_removed_by_eligibility": build_missingness_removed_by_eligibility(panel_with_missingness, signals),
        "missingness_pattern_counts": build_missingness_pattern_counts(panel_with_missingness),
    }

    write_phase1_outputs(outputs, paths.output_root)
    return outputs


def write_phase1_outputs(outputs: dict[str, pd.DataFrame], output_root: Path) -> None:
    interim_dir = output_root / "interim"
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"

    interim_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    outputs["panel_with_missingness"].to_parquet(interim_dir / "panel_with_missingness.parquet", index=False)
    outputs["signal_support_windows"].to_csv(tables_dir / "signal_support_windows.csv", index=False)
    outputs["missingness_summary_by_signal"].to_csv(tables_dir / "missingness_summary_by_signal.csv", index=False)
    outputs["missingness_summary_by_year"].to_csv(tables_dir / "missingness_summary_by_year.csv", index=False)
    outputs["missingness_removed_by_eligibility"].to_csv(
        tables_dir / "missingness_removed_by_eligibility.csv",
        index=False,
    )
    outputs["missingness_pattern_counts"].to_csv(tables_dir / "missingness_pattern_counts.csv", index=False)

    plot_residual_missingness_heatmap(
        outputs["missingness_summary_by_year"],
        figures_dir / "residual_missingness_heatmap.png",
    )
