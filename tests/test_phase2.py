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
    n = 240
    dates = pd.date_range("2000-01-31", periods=n, freq="ME")
    permnos = np.arange(10000, 10000 + n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    missing = (x1 > 0.2).astype(int)

    x_true = 1.0 + 0.8 * x1 - 0.4 * x2 + 1.5 * missing
    observed_x = np.where(missing == 1, np.nan, x_true)
    ret_fwd = 0.5 + 0.3 * x1 + 0.2 * x2 + 0.9 * x_true + rng.normal(scale=0.05, size=n)

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
    )

    stage1 = outputs["stage1_table"].iloc[0]
    jtest = outputs["jtest_table"].iloc[0]
    benchmark = outputs["benchmark_table"]

    assert stage1["signal"] == "REV6"
    assert stage1["stage1_p_value"] < 0.05
    assert jtest["j_p_value"] < 0.05
    assert set(benchmark["benchmark"]) == {"complete_case", "conditional_mean", "unconditional_mean"}
    assert (output_root / "tables" / "mar_stage1_regression_diagnostic.csv").exists()
    assert (output_root / "tables" / "mar_jtest_results.csv").exists()
    assert (output_root / "figures" / "jtest_pvalue_distribution.png").exists()
