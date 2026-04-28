from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ap_mnar.data.x_obs import build_x_obs_availability_table, load_x_obs_spec, merge_x_obs_columns
from ap_mnar.experiments.step0_audit import DEFAULT_SIGNALS
from ap_mnar.models.mar_benchmark import (
    build_benchmark_comparison_rows,
    build_signal_mar_panel,
    fit_return_model,
    predict_linear_projection,
)
from ap_mnar.reporting.figures import plot_jtest_pvalue_distribution
from ap_mnar.stats.jtest import run_missingness_jtest
from ap_mnar.stats.stage1_diagnostic import run_stage1_regression_diagnostic


@dataclass(frozen=True)
class Phase2Paths:
    panel_with_missingness_path: Path
    signal_registry_path: Path
    x_obs_config_path: Path
    firm_panel_path: Path
    output_root: Path


def run_phase2_mar_test(
    paths: Phase2Paths,
    signals: Sequence[str] = DEFAULT_SIGNALS,
) -> dict[str, pd.DataFrame]:
    panel = pd.read_parquet(paths.panel_with_missingness_path)
    signal_registry = pd.read_csv(paths.signal_registry_path)
    signal_registry = signal_registry.loc[signal_registry["signal"].isin(signals)].copy()
    signal_registry["tier1_delay_months"] = signal_registry["tier1_delay_months"].astype(int)

    x_obs_spec = load_x_obs_spec(paths.x_obs_config_path)
    panel = merge_x_obs_columns(panel, paths.firm_panel_path, x_obs_spec.columns)

    stage1_rows: list[dict[str, float | int | str]] = []
    jtest_rows: list[dict[str, float | int | str]] = []
    benchmark_rows: list[dict[str, float | int | str]] = []

    for signal in signals:
        sample, imputation_bundle, unconditional_mean = build_signal_mar_panel(panel, signal, x_obs_spec.columns)
        sample["x_cond_mean"] = predict_linear_projection(imputation_bundle, sample)
        sample["x_conditional"] = sample[f"{signal}_tier1"].where(
            sample["observed_indicator"].eq(1),
            sample["x_cond_mean"],
        )
        sample["x_unconditional"] = sample[f"{signal}_tier1"].where(
            sample["observed_indicator"].eq(1),
            unconditional_mean,
        )

        return_bundle = fit_return_model(sample, "x_conditional", x_obs_spec.columns)
        predicted_return = return_bundle.model.predict()
        residual_sample = sample.loc[return_bundle.model.model.data.row_labels].copy()
        residual_sample["mar_return_residual"] = residual_sample["ret_fwd_1m"] - predicted_return

        stage1_result = run_stage1_regression_diagnostic(
            residual_sample,
            x_obs_spec.columns,
            return_residual_col="mar_return_residual",
        )
        stage1_rows.append(
            {
                "signal": signal,
                "n_eligible_with_x_obs": int(len(sample)),
                "n_observed_signal": int(sample["observed_indicator"].sum()),
                "n_missing_signal": int(sample["missing_indicator"].sum()),
                "imputation_r_squared": float(imputation_bundle.model.rsquared),
                "return_model_r_squared": float(return_bundle.model.rsquared),
                "x_obs_column_count": len(x_obs_spec.columns),
                **stage1_result,
            }
        )

        jtest_result = run_missingness_jtest(
            residual_sample,
            x_obs_spec.columns,
            return_residual_col="mar_return_residual",
        )
        jtest_rows.append(
            {
                "signal": signal,
                "n_obs": int(len(residual_sample)),
                "n_missing_signal": int(residual_sample["missing_indicator"].sum()),
                **jtest_result,
            }
        )

        benchmark_rows.extend(
            build_benchmark_comparison_rows(
                signal,
                sample,
                x_obs_spec.columns,
            )
        )

    stage1_table = pd.DataFrame(stage1_rows).sort_values("signal", ignore_index=True)
    jtest_table = pd.DataFrame(jtest_rows).sort_values("signal", ignore_index=True)
    benchmark_table = pd.DataFrame(benchmark_rows).sort_values(["signal", "benchmark"], ignore_index=True)
    rejection_summary = build_rejection_summary(stage1_table, jtest_table)
    x_obs_audit = build_x_obs_availability_table(panel, x_obs_spec.columns)

    outputs = {
        "stage1_table": stage1_table,
        "jtest_table": jtest_table,
        "benchmark_table": benchmark_table,
        "rejection_summary": rejection_summary,
        "x_obs_audit": x_obs_audit,
    }
    write_phase2_outputs(outputs, paths.output_root)
    return outputs


def build_rejection_summary(
    stage1_table: pd.DataFrame,
    jtest_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for label, frame, p_col in [
        ("stage1", stage1_table, "stage1_p_value"),
        ("jtest", jtest_table, "j_p_value"),
    ]:
        valid = frame[p_col].dropna()
        rows.append(
            {
                "test": label,
                "signals_tested": int(frame["signal"].nunique()),
                "p_lt_0_10": int((valid < 0.10).sum()),
                "p_lt_0_05": int((valid < 0.05).sum()),
                "p_lt_0_01": int((valid < 0.01).sum()),
                "mean_p_value": float(valid.mean()) if not valid.empty else np.nan,
                "stouffer_z": float(valid.apply(lambda p: norm.ppf(1 - p)).sum() / np.sqrt(len(valid))) if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_phase2_outputs(outputs: dict[str, pd.DataFrame], output_root: Path) -> None:
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    outputs["stage1_table"].to_csv(tables_dir / "mar_stage1_regression_diagnostic.csv", index=False)
    outputs["jtest_table"].to_csv(tables_dir / "mar_jtest_results.csv", index=False)
    outputs["benchmark_table"].to_csv(tables_dir / "mar_benchmark_comparison.csv", index=False)
    outputs["rejection_summary"].to_csv(tables_dir / "mar_rejection_summary.csv", index=False)
    outputs["x_obs_audit"].to_csv(tables_dir / "x_obs_phase2_availability_audit.csv", index=False)

    plot_jtest_pvalue_distribution(
        outputs["jtest_table"],
        figures_dir / "jtest_pvalue_distribution.png",
    )
