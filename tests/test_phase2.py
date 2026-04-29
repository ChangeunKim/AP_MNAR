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

from ap_mnar.experiments.phase2_mar_test import Phase2Paths, run_phase2_mar_test


def make_workspace_tmp_dir(name: str) -> Path:
    path = REPO_ROOT / ".tmp_test_artifacts" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_run_phase2_mar_test_detects_synthetic_mnar_signal() -> None:
    tmp_path = make_workspace_tmp_dir("phase2_run")
    panel_path = tmp_path / "panel_with_missingness.parquet"
    registry_path = tmp_path / "signal_registry.csv"
    x_obs_path = tmp_path / "x_obs.yaml"
    firm_panel_path = tmp_path / "firm_characs.csv"
    output_root = tmp_path / "outputs"

    rng = np.random.default_rng(0)
    n = 480
    dates = pd.date_range("2000-01-31", periods=n, freq="ME")
    permnos = np.arange(10000, 10000 + n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent_shock = rng.normal(scale=0.35, size=n)

    x_true = 1.0 + 0.9 * x1 - 0.5 * x2 + 1.25 * (x1**2 - 0.5) + 0.8 * x1 * x2 + latent_shock
    missing_score = x_true + 0.35 * x1 - 0.15 * x2
    missing_threshold = np.quantile(missing_score, 0.55)
    missing = (missing_score >= missing_threshold).astype(int)
    observed_x = np.where(missing == 1, np.nan, x_true)
    ret_fwd = 0.2 * x1 - 0.1 * x2 + rng.normal(scale=0.15, size=n)

    panel = pd.DataFrame(
        {
            "permno": permnos,
            "date": dates,
            "year": dates.year,
            "ret_fwd_1m": ret_fwd,
            "REV6": observed_x,
            "REV6_tier1": observed_x,
            "REV6_eligible": True,
            "REV6_before_support": False,
            "REV6_after_support": False,
            "REV6_no_support": False,
            "REV6_raw_missing": pd.isna(observed_x),
            "REV6_tier1_missing": pd.isna(observed_x),
            "REV6_observed_eligible": ~pd.isna(observed_x),
            "REV6_residual_missing": pd.isna(observed_x),
            "eligible_signal_count": np.ones(n, dtype=int),
            "observed_signal_count": (~pd.isna(observed_x)).astype(int),
            "residual_missing_count": pd.isna(observed_x).astype(int),
            "has_any_eligible_signal": True,
            "complete_case_flag": ~pd.isna(observed_x),
            "missing_pattern_code": np.where(pd.isna(observed_x), "1", "0"),
            "missing_pattern_label": np.where(pd.isna(observed_x), "REV6", "COMPLETE"),
        }
    )
    panel.to_parquet(panel_path, index=False)

    registry = pd.DataFrame(
        {
            "signal": ["REV6"],
            "Frequency": ["Monthly"],
            "cat_data": ["Analyst"],
            "cat_economic": ["earnings forecast"],
            "SampleStartYear": [1977],
            "SampleEndYear": [1992],
            "detailed_definition": ["rev6"],
            "tier1_delay_months": [0],
        }
    )
    registry.to_csv(registry_path, index=False)

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

    firm_panel = pd.DataFrame(
        {
            "permno": permnos,
            "date": dates,
            "mve": x1,
            "bm": x2,
        }
    )
    firm_panel.to_csv(firm_panel_path, index=False)

    outputs = run_phase2_mar_test(
        paths=Phase2Paths(
            panel_with_missingness_path=panel_path,
            signal_registry_path=registry_path,
            x_obs_config_path=x_obs_path,
            firm_panel_path=firm_panel_path,
            output_root=output_root,
        ),
        signals=["REV6"],
        n_draws=7,
        random_seed=17,
        n_folds=4,
    )

    stage1_table = outputs["stage1_table"]
    jtest_table = outputs["jtest_table"]
    benchmark = outputs["benchmark_table"]
    pattern_slice = outputs["pattern_slice_table"]
    draw_diag = outputs["draw_aggregation_table"]
    contribution = outputs["moment_contribution_table"]

    expected_benchmark_types = {"fixed_x_obs", "augmented_signal_history"}
    assert set(stage1_table["benchmark_type"]) == expected_benchmark_types
    assert set(jtest_table["benchmark_type"]) == expected_benchmark_types

    stage1 = stage1_table.loc[stage1_table["benchmark_type"].eq("fixed_x_obs")].iloc[0]
    jtest = jtest_table.loc[jtest_table["benchmark_type"].eq("fixed_x_obs")].iloc[0]

    assert stage1["signal"] == "REV6"
    assert stage1["stage1_p_value"] < 0.05
    assert jtest["j_p_value"] < 0.05
    assert bool(stage1["signal_history_augmented"]) is False
    assert bool(jtest["weighted_jtest_flag"]) is True
    assert set(benchmark["benchmark_type"]) == expected_benchmark_types
    assert set(benchmark["benchmark"]) == {
        "complete_case",
        "conditional_mean",
        "conditional_quantile_draw_mean",
        "unconditional_mean",
    }
    assert not pattern_slice.empty
    assert "full_sample" in set(pattern_slice["slice_name"])
    assert set(pattern_slice["benchmark_type"]) == expected_benchmark_types
    assert set(draw_diag["benchmark_type"]) == expected_benchmark_types
    assert not contribution.empty
    assert set(contribution["benchmark_type"]) == expected_benchmark_types
    assert stage1["stage1_draw_count"] == 7
    assert jtest["j_draw_count"] == 7
    assert (output_root / "tables" / "mar_stage1_regression_diagnostic.csv").exists()
    assert (output_root / "tables" / "mar_jtest_results.csv").exists()
    assert (output_root / "tables" / "step1a_signal_diagnostic.csv").exists()
    assert (output_root / "tables" / "step1a_mar_jtest_results.csv").exists()
    assert (output_root / "tables" / "step1a_pattern_slice_results.csv").exists()
    assert (output_root / "tables" / "step1a_benchmark_strength_comparison.csv").exists()
    assert (output_root / "tables" / "step1a_draw_aggregation_diagnostics.csv").exists()
    assert (output_root / "tables" / "step1a_moment_contribution.csv").exists()
    assert (output_root / "figures" / "jtest_pvalue_distribution.png").exists()
