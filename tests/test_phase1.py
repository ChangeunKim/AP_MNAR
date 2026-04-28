from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ap_mnar.experiments.phase1_missingness import Phase1Paths, run_phase1_missingness
from ap_mnar.missingness.eligibility import (
    build_eligibility_matrix,
    compute_signal_support_windows,
    load_missingness_rules,
)


def make_workspace_tmp_dir(name: str) -> Path:
    path = REPO_ROOT / ".tmp_test_artifacts" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_rules(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "version: 1",
                "global:",
                '  analyst_start_date: "1988-01-01"',
                '  support_window_basis: "tier1_nonmissing"',
                "  apply_signal_support_window: true",
                '  terminal_support_policy: "ineligible_outside_support"',
                "signals:",
                "  REV6:",
                "    min_history_months: 0",
                "    special_constraints: []",
                "  FEPS:",
                "    min_history_months: 0",
                "    special_constraints: []",
            ]
        ),
        encoding="utf-8",
    )


def test_compute_signal_support_windows_uses_tier1_nonmissing_bounds() -> None:
    tmp_path = make_workspace_tmp_dir("phase1_support")
    rules_path = tmp_path / "rules.yaml"
    write_rules(rules_path)

    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "1988-01-31",
                    "1988-02-29",
                    "1988-03-31",
                    "1988-04-30",
                    "1988-05-31",
                    "1988-06-30",
                ]
            ),
            "REV6_tier1": [1.0, 2.0, pd.NA, 4.0, pd.NA, pd.NA],
            "FEPS_tier1": [pd.NA, pd.NA, pd.NA, 10.0, pd.NA, 12.0],
        }
    )
    registry = pd.DataFrame(
        {
            "signal": ["REV6", "FEPS"],
            "Frequency": ["Monthly", "Annual"],
            "tier1_delay_months": [0, 6],
        }
    )

    rules = load_missingness_rules(rules_path)
    support = compute_signal_support_windows(panel, registry, rules, ["REV6", "FEPS"])
    support_lookup = support.set_index("signal")

    assert support_lookup.loc["REV6", "support_start_date"] == pd.Timestamp("1988-01-31")
    assert support_lookup.loc["REV6", "support_end_date"] == pd.Timestamp("1988-04-30")
    assert support_lookup.loc["FEPS", "support_start_date"] == pd.Timestamp("1988-04-30")
    assert support_lookup.loc["FEPS", "support_end_date"] == pd.Timestamp("1988-06-30")


def test_build_eligibility_matrix_marks_before_and_after_support() -> None:
    tmp_path = make_workspace_tmp_dir("phase1_eligibility")
    rules_path = tmp_path / "rules.yaml"
    write_rules(rules_path)

    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["1988-01-31", "1988-02-29", "1988-03-31", "1988-04-30"]),
            "REV6_tier1": [1.0, pd.NA, 2.0, pd.NA],
        }
    )
    registry = pd.DataFrame(
        {
            "signal": ["REV6"],
            "Frequency": ["Monthly"],
            "tier1_delay_months": [0],
        }
    )

    rules = load_missingness_rules(rules_path)
    support = compute_signal_support_windows(panel, registry, rules, ["REV6"])
    eligibility = build_eligibility_matrix(panel, support, ["REV6"])

    assert eligibility["REV6_eligible"].tolist() == [True, True, True, False]
    assert eligibility["REV6_after_support"].tolist() == [False, False, False, True]


def test_run_phase1_missingness_builds_expected_outputs() -> None:
    tmp_path = make_workspace_tmp_dir("phase1_run")
    panel_path = tmp_path / "panel_base.parquet"
    registry_path = tmp_path / "signal_registry.csv"
    rules_path = tmp_path / "rules.yaml"
    output_root = tmp_path / "outputs"
    write_rules(rules_path)

    panel = pd.DataFrame(
        {
            "permno": [1] * 6,
            "date": pd.to_datetime(
                [
                    "1988-01-31",
                    "1988-02-29",
                    "1988-03-31",
                    "1988-04-30",
                    "1988-05-31",
                    "1988-06-30",
                ]
            ),
            "year": [1988] * 6,
            "ret_fwd_1m": [0.1, 0.2, 0.3, 0.4, 0.5, pd.NA],
            "REV6": [1.0, pd.NA, 2.0, pd.NA, pd.NA, pd.NA],
            "REV6_tier1": [1.0, pd.NA, 2.0, pd.NA, pd.NA, pd.NA],
            "FEPS": [pd.NA, pd.NA, 10.0, pd.NA, pd.NA, pd.NA],
            "FEPS_tier1": [pd.NA, pd.NA, 10.0, pd.NA, pd.NA, 12.0],
        }
    )
    panel.to_parquet(panel_path, index=False)

    registry = pd.DataFrame(
        {
            "signal": ["REV6", "FEPS"],
            "Frequency": ["Monthly", "Annual"],
            "cat_data": ["Analyst", "Analyst"],
            "cat_economic": ["earnings forecast", "profitability"],
            "SampleStartYear": [1977, 1983],
            "SampleEndYear": [1992, 2002],
            "detailed_definition": ["rev6", "feps"],
            "tier1_delay_months": [0, 6],
        }
    )
    registry.to_csv(registry_path, index=False)

    outputs = run_phase1_missingness(
        paths=Phase1Paths(
            panel_base_path=panel_path,
            signal_registry_path=registry_path,
            missingness_rules_path=rules_path,
            output_root=output_root,
        ),
        signals=["REV6", "FEPS"],
    )

    result = outputs["panel_with_missingness"]
    feb_row = result.loc[result["date"] == pd.Timestamp("1988-02-29")].iloc[0]
    apr_row = result.loc[result["date"] == pd.Timestamp("1988-04-30")].iloc[0]
    may_row = result.loc[result["date"] == pd.Timestamp("1988-05-31")].iloc[0]
    jun_row = result.loc[result["date"] == pd.Timestamp("1988-06-30")].iloc[0]

    assert bool(feb_row["REV6_residual_missing"]) is True
    assert bool(apr_row["FEPS_eligible"]) is True
    assert bool(apr_row["FEPS_residual_missing"]) is True
    assert bool(may_row["FEPS_eligible"]) is True
    assert bool(may_row["FEPS_residual_missing"]) is True
    assert bool(jun_row["REV6_eligible"]) is False
    assert jun_row["missing_pattern_label"] == "COMPLETE"

    pattern_counts = outputs["missingness_pattern_counts"]
    assert "NO_ELIGIBLE_SIGNAL" not in pattern_counts["missing_pattern_label"].tolist()
    assert (output_root / "interim" / "panel_with_missingness.parquet").exists()
    assert (output_root / "tables" / "missingness_summary_by_signal.csv").exists()
    assert (output_root / "figures" / "residual_missingness_heatmap.png").exists()
