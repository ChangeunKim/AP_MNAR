from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ap_mnar.experiments.phase3_counterfactual import Phase3Paths, run_phase3_counterfactual


def make_workspace_tmp_dir(name: str) -> Path:
    path = REPO_ROOT / ".tmp_test_artifacts" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_run_phase3_counterfactual_writes_oos_and_portfolio_outputs() -> None:
    tmp_path = make_workspace_tmp_dir("phase3_run")
    panel_path = tmp_path / "panel_with_missingness.parquet"
    x_obs_path = tmp_path / "x_obs.yaml"
    firm_panel_path = tmp_path / "firm_characs.csv"
    output_root = tmp_path / "outputs"

    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-31", "2005-12-31", freq="ME")
    permnos = np.arange(10000, 10024)

    panel_rows = []
    firm_rows = []
    for date_idx, date in enumerate(dates):
        time_effect = np.sin(date_idx / 6.0)
        for perm_idx, permno in enumerate(permnos):
            x1 = rng.normal() + perm_idx * 0.03
            x2 = rng.normal() + time_effect
            true_signal = 0.8 * x1 - 0.5 * x2 + 0.25 * time_effect
            missing = int(true_signal > 0.35)
            observed_signal = np.nan if missing else true_signal
            ret_fwd = 0.08 * x1 + 0.04 * x2 + 0.55 * true_signal + rng.normal(scale=0.10)

            panel_rows.append(
                {
                    "permno": permno,
                    "date": date,
                    "year": date.year,
                    "ret_fwd_1m": ret_fwd,
                    "REV6": observed_signal,
                    "REV6_tier1": observed_signal,
                    "REV6_eligible": True,
                    "REV6_before_support": False,
                    "REV6_after_support": False,
                    "REV6_no_support": False,
                    "REV6_raw_missing": pd.isna(observed_signal),
                    "REV6_tier1_missing": pd.isna(observed_signal),
                    "REV6_observed_eligible": not pd.isna(observed_signal),
                    "REV6_residual_missing": pd.isna(observed_signal),
                    "eligible_signal_count": 1,
                    "observed_signal_count": int(not pd.isna(observed_signal)),
                    "residual_missing_count": int(pd.isna(observed_signal)),
                    "has_any_eligible_signal": True,
                    "complete_case_flag": not pd.isna(observed_signal),
                    "missing_pattern_code": "1" if pd.isna(observed_signal) else "0",
                    "missing_pattern_label": "REV6" if pd.isna(observed_signal) else "COMPLETE",
                }
            )
            firm_rows.append(
                {
                    "permno": permno,
                    "date": date,
                    "mve": x1,
                    "bm": x2,
                }
            )

    pd.DataFrame(panel_rows).to_parquet(panel_path, index=False)
    pd.DataFrame(firm_rows).to_csv(firm_panel_path, index=False)
    x_obs_path.write_text(
        "\n".join(
            [
                "version: 1",
                "columns:",
                "  - mve",
                "  - bm",
            ]
        ),
        encoding="utf-8",
    )

    outputs = run_phase3_counterfactual(
        paths=Phase3Paths(
            panel_with_missingness_path=panel_path,
            x_obs_config_path=x_obs_path,
            firm_panel_path=firm_panel_path,
            output_root=output_root,
        ),
        signals=["REV6"],
        min_train_years=2,
        stochastic_draws=7,
        quantile_bins=4,
        random_seed=123,
        signal_sort_groups=4,
        show_progress=False,
    )

    oos_table = outputs["oos_table"]
    portfolio_table = outputs["portfolio_table"]
    signal_sorted_table = outputs["signal_sorted_table"]
    pattern_slice_table = outputs["pattern_slice_table"]
    attenuation_summary = outputs["attenuation_summary"]

    expected_regimes = {
        "complete_case",
        "unconditional_mean",
        "conditional_mean",
        "residual_bootstrap",
        "conditional_quantile_draw",
    }
    expected_benchmarks = {"fixed_x_obs", "augmented_signal_history"}
    assert set(oos_table["regime"]) == expected_regimes
    assert set(portfolio_table["regime"]) == expected_regimes
    assert set(oos_table["benchmark_type"]) == expected_benchmarks
    assert set(portfolio_table["benchmark_type"]) == expected_benchmarks

    fixed_oos = oos_table.loc[oos_table["benchmark_type"].eq("fixed_x_obs")].copy()
    complete_case = fixed_oos.loc[fixed_oos["regime"].eq("complete_case")].iloc[0]
    conditional = fixed_oos.loc[fixed_oos["regime"].eq("conditional_mean")].iloc[0]
    unconditional = fixed_oos.loc[fixed_oos["regime"].eq("unconditional_mean")].iloc[0]
    residual_bootstrap = fixed_oos.loc[fixed_oos["regime"].eq("residual_bootstrap")].iloc[0]
    quantile_draw = fixed_oos.loc[fixed_oos["regime"].eq("conditional_quantile_draw")].iloc[0]

    assert conditional["prediction_coverage_rate"] >= complete_case["prediction_coverage_rate"]
    assert unconditional["missing_row_prediction_count"] > 0
    assert conditional["missing_row_prediction_count"] > 0
    assert residual_bootstrap["missing_row_prediction_count"] > 0
    assert quantile_draw["missing_row_prediction_count"] > 0
    assert complete_case["missing_row_prediction_count"] == 0
    assert pd.notna(conditional["oos_r2"])
    assert residual_bootstrap["draw_count"] == 7
    assert quantile_draw["draw_count"] == 7
    assert residual_bootstrap["mean_prediction_draw_std"] >= 0
    assert quantile_draw["mean_prediction_draw_std"] >= 0
    assert float(oos_table["prediction_coverage_rate"].max()) <= 1.0
    assert float(oos_table["missing_row_prediction_rate"].max()) <= 1.0

    assert not signal_sorted_table.empty
    assert set(signal_sorted_table["regime"]) == expected_regimes
    assert set(signal_sorted_table["benchmark_type"]) == expected_benchmarks
    assert signal_sorted_table["signal_sort_group"].nunique() >= 3
    conditional_groups = signal_sorted_table.loc[
        signal_sorted_table["regime"].eq("conditional_mean")
        & signal_sorted_table["benchmark_type"].eq("fixed_x_obs")
    ].sort_values("signal_sort_group")
    assert conditional_groups["mean_signal_sort_value"].is_monotonic_increasing
    assert "delta_oos_r2_vs_complete_case" in signal_sorted_table.columns
    assert not pattern_slice_table.empty
    assert set(pattern_slice_table["benchmark_type"]) == expected_benchmarks
    assert not attenuation_summary.empty

    assert (output_root / "tables" / "counterfactual_oos_results.csv").exists()
    assert (output_root / "tables" / "counterfactual_oos_results_by_benchmark.csv").exists()
    assert (output_root / "tables" / "portfolio_spread_comparison.csv").exists()
    assert (output_root / "tables" / "portfolio_spread_comparison_by_benchmark.csv").exists()
    assert (output_root / "tables" / "counterfactual_signal_sorted_results.csv").exists()
    assert (output_root / "tables" / "counterfactual_signal_sorted_results_by_benchmark.csv").exists()
    assert (output_root / "tables" / "counterfactual_pattern_slice_results.csv").exists()
    assert (output_root / "tables" / "counterfactual_attenuation_summary.csv").exists()
    assert (output_root / "figures" / "counterfactual_sensitivity_by_signal.png").exists()
    assert (output_root / "figures" / "counterfactual_delta_r2_by_signal_group.png").exists()
