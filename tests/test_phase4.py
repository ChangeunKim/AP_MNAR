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

from ap_mnar.experiments.phase4_pricing import Phase4Paths, run_phase4_pricing


def make_workspace_tmp_dir(name: str) -> Path:
    path = REPO_ROOT / ".tmp_test_artifacts" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_run_phase4_pricing_recovers_positive_missingness_premium() -> None:
    tmp_path = make_workspace_tmp_dir("phase4_run")
    panel_path = tmp_path / "panel_with_missingness.parquet"
    x_obs_path = tmp_path / "x_obs.yaml"
    firm_panel_path = tmp_path / "firm_characs.csv"
    output_root = tmp_path / "outputs"

    rng = np.random.default_rng(7)
    dates = pd.date_range("2000-01-31", periods=72, freq="ME")
    permnos = np.arange(10000, 10100)

    panel_rows: list[dict[str, float | int | bool | str | pd.Timestamp]] = []
    firm_rows: list[dict[str, float | int | pd.Timestamp]] = []
    for month_index, date in enumerate(dates):
        month_effect = rng.normal(scale=0.01)
        for permno in permnos:
            x1 = rng.normal()
            x2 = rng.normal()
            never_covered = permno < 10012

            latent_signal = 0.9 * x1 - 0.4 * x2 + rng.normal(scale=0.12)
            if never_covered:
                observed_signal = np.nan
            else:
                is_missing = latent_signal < -0.1 or rng.uniform() < 0.12
                observed_signal = np.nan if is_missing else latent_signal

            missing_indicator = int(pd.isna(observed_signal))
            ret_fwd = (
                0.01
                + 0.04 * x1
                - 0.03 * x2
                + 0.05 * latent_signal
                + 0.08 * missing_indicator
                + month_effect
                + rng.normal(scale=0.03)
            )

            panel_rows.append(
                {
                    "permno": permno,
                    "date": date,
                    "year": date.year,
                    "ret_fwd_1m": ret_fwd,
                    "ForecastDispersion": observed_signal,
                    "ForecastDispersion_tier1": observed_signal,
                    "ForecastDispersion_eligible": True,
                    "ForecastDispersion_before_support": False,
                    "ForecastDispersion_after_support": False,
                    "ForecastDispersion_no_support": False,
                    "ForecastDispersion_raw_missing": pd.isna(observed_signal),
                    "ForecastDispersion_tier1_missing": pd.isna(observed_signal),
                    "ForecastDispersion_observed_eligible": not pd.isna(observed_signal),
                    "ForecastDispersion_residual_missing": pd.isna(observed_signal),
                    "eligible_signal_count": 1,
                    "observed_signal_count": int(not pd.isna(observed_signal)),
                    "residual_missing_count": int(pd.isna(observed_signal)),
                    "has_any_eligible_signal": True,
                    "complete_case_flag": not pd.isna(observed_signal),
                    "missing_pattern_code": "1" if pd.isna(observed_signal) else "0",
                    "missing_pattern_label": "ForecastDispersion" if pd.isna(observed_signal) else "COMPLETE",
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

    outputs = run_phase4_pricing(
        paths=Phase4Paths(
            panel_with_missingness_path=panel_path,
            x_obs_config_path=x_obs_path,
            firm_panel_path=firm_panel_path,
            output_root=output_root,
        ),
        signals=["ForecastDispersion"],
        min_train_months=12,
        min_cross_section=10,
        nw_lags=3,
    )

    pricing_panel = outputs["pricing_panel"]
    pooled_table = outputs["pooled_pricing_table"]
    fama_macbeth_table = outputs["fama_macbeth_table"]
    coverage_table = outputs["coverage_decomposition_table"]
    premium_time_series = outputs["premium_time_series"]

    assert not pricing_panel.empty
    assert pricing_panel["x_mar_cond_mean"].notna().all()
    assert pricing_panel["x_mar_filled"].notna().all()

    baseline_pooled = pooled_table.loc[
        pooled_table["signal"].eq("ForecastDispersion")
        & pooled_table["sample_name"].eq("full")
        & pooled_table["specification"].eq("baseline_missing_only")
        & pooled_table["coefficient"].eq("missing_indicator")
    ].iloc[0]
    assert baseline_pooled["coefficient_value"] > 0

    baseline_fm = fama_macbeth_table.loc[
        fama_macbeth_table["signal"].eq("ForecastDispersion")
        & fama_macbeth_table["sample_name"].eq("full")
        & fama_macbeth_table["specification"].eq("baseline_missing_only")
        & fama_macbeth_table["coefficient"].eq("missing_indicator")
    ].iloc[0]
    assert baseline_fm["mean_coefficient"] > 0
    assert baseline_fm["month_count"] >= 40

    assert set(coverage_table["sample_name"]).issuperset({"within_signal_coverage", "outside_signal_coverage"})
    assert not premium_time_series.empty
    assert premium_time_series["coefficient_value"].notna().all()

    assert (output_root / "interim" / "phase4_pricing_panel.parquet").exists()
    assert (output_root / "tables" / "missingness_pricing_results.csv").exists()
    assert (output_root / "tables" / "missingness_fama_macbeth_results.csv").exists()
    assert (output_root / "tables" / "missingness_coverage_decomposition.csv").exists()
    assert (output_root / "tables" / "missingness_premium_time_series.csv").exists()
    assert (output_root / "figures" / "missingness_alpha_by_signal.png").exists()
    assert (output_root / "figures" / "missingness_premium_over_time.png").exists()
