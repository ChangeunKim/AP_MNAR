from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ap_mnar.data.load_raw import normalize_to_month_end
from ap_mnar.data.metadata import load_signal_metadata
from ap_mnar.data.target import add_forward_return
from ap_mnar.experiments.step0_audit import Phase0Paths, run_phase0_audit


def make_workspace_tmp_dir(name: str) -> Path:
    path = REPO_ROOT / ".tmp_test_artifacts" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_normalize_to_month_end_aligns_mixed_monthly_dates() -> None:
    values = pd.Series(["1987-03-01", "1987-03-31", "1987-02-15"])
    normalized = normalize_to_month_end(values)
    expected = pd.to_datetime(["1987-03-31", "1987-03-31", "1987-02-28"])
    pd.testing.assert_series_equal(normalized, pd.Series(expected), check_names=False)


def test_add_forward_return_shifts_within_permno() -> None:
    frame = pd.DataFrame(
        {
            "permno": [1, 1, 1, 2, 2],
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-01-31", "2020-02-29"]),
            "ret": [0.1, 0.2, 0.3, -0.1, 0.0],
        }
    )
    result = add_forward_return(frame)
    permno1 = result.loc[result["permno"].eq(1), "ret_fwd_1m"].tolist()
    permno2 = result.loc[result["permno"].eq(2), "ret_fwd_1m"].tolist()
    assert permno1[:2] == [0.2, 0.3]
    assert pd.isna(permno1[2])
    assert permno2[:1] == [0.0]
    assert pd.isna(permno2[1])


def test_load_signal_metadata_assigns_tier1_delays() -> None:
    tmp_path = make_workspace_tmp_dir("metadata_delay")
    metadata_path = tmp_path / "SignalDoc.csv"
    metadata = pd.DataFrame(
        {
            "Acronym": ["REV6", "FEPS"],
            "Frequency": ["Monthly", "Annual"],
            "Cat.Data": ["Analyst", "Analyst"],
            "Cat.Economic": ["earnings forecast", "profitability"],
            "SampleStartYear": [1977, 1983],
            "SampleEndYear": [1992, 2002],
            "Detailed Definition": ["rev", "feps"],
        }
    )
    metadata.to_csv(metadata_path, index=False)
    result = load_signal_metadata(metadata_path, ["REV6", "FEPS"])
    delays = dict(zip(result["signal"], result["tier1_delay_months"], strict=True))
    assert delays == {"FEPS": 6, "REV6": 0}


def test_run_phase0_audit_builds_expected_outputs() -> None:
    tmp_path = make_workspace_tmp_dir("phase0_audit")
    firm_path = tmp_path / "firm.csv"
    signal_path = tmp_path / "signals.csv"
    metadata_path = tmp_path / "SignalDoc.csv"
    output_root = tmp_path / "outputs"

    firm = pd.DataFrame(
        {
            "permno": [1] * 7 + [2] * 7,
            "date": [
                "2020-01-31",
                "2020-02-29",
                "2020-03-31",
                "2020-04-30",
                "2020-05-31",
                "2020-06-30",
                "2020-07-31",
            ]
            * 2,
            "ret": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7],
            "mve": [10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26],
            "exchcd": [1] * 7 + [3] * 7,
        }
    )
    firm.to_csv(firm_path, index=False)

    signals = pd.DataFrame(
        {
            "permno": [1, 1, 1, 2, 2],
            "date": ["2020-01-01", "2020-02-01", "2020-01-01", "2020-01-01", "2020-02-01"],
            "AnalystRevision": [1.5, 1.6, pd.NA, 2.5, pd.NA],
            "REV6": [0.3, 0.4, pd.NA, 0.1, 0.2],
            "ForecastDispersion": [0.05, pd.NA, pd.NA, 0.07, pd.NA],
            "FEPS": [3.0, pd.NA, pd.NA, 4.0, pd.NA],
        }
    )
    signals.to_csv(signal_path, index=False)

    metadata = pd.DataFrame(
        {
            "Acronym": ["AnalystRevision", "REV6", "ForecastDispersion", "FEPS"],
            "Frequency": ["Annual", "Monthly", "Annual", "Annual"],
            "Cat.Data": ["Analyst"] * 4,
            "Cat.Economic": ["earnings forecast", "earnings forecast", "volatility", "profitability"],
            "SampleStartYear": [1975, 1977, 1976, 1983],
            "SampleEndYear": [1980, 1992, 2000, 2002],
            "Detailed Definition": ["ar", "rev6", "disp", "feps"],
        }
    )
    metadata.to_csv(metadata_path, index=False)

    outputs = run_phase0_audit(
        paths=Phase0Paths(
            signal_panel_path=signal_path,
            firm_panel_path=firm_path,
            signal_metadata_path=metadata_path,
            output_root=output_root,
        ),
        x_obs_columns=["mve"],
    )

    panel = outputs["panel_base"]
    jan_row = panel.loc[(panel["permno"] == 1) & (panel["date"] == pd.Timestamp("2020-01-31"))].iloc[0]
    jul_row = panel.loc[(panel["permno"] == 1) & (panel["date"] == pd.Timestamp("2020-07-31"))].iloc[0]

    assert jan_row["REV6"] == 0.3
    assert jan_row["REV6_tier1"] == 0.3
    assert pd.isna(jan_row["FEPS_tier1"])
    assert jul_row["FEPS_tier1"] == 3.0
    assert jan_row["ret_fwd_1m"] == 0.2
    assert (output_root / "interim" / "panel_base.parquet").exists()
    assert (output_root / "tables" / "signal_coverage.csv").exists()
    assert outputs["x_obs_availability_audit"].loc[0, "column"] == "mve"


def test_run_phase0_audit_filters_pre_1988_rows() -> None:
    tmp_path = make_workspace_tmp_dir("phase0_start_date")
    firm_path = tmp_path / "firm.csv"
    signal_path = tmp_path / "signals.csv"
    metadata_path = tmp_path / "SignalDoc.csv"
    output_root = tmp_path / "outputs"

    firm = pd.DataFrame(
        {
            "permno": [1, 1, 1],
            "date": ["1987-12-31", "1988-01-31", "1988-02-29"],
            "ret": [0.1, 0.2, 0.3],
            "mve": [10, 11, 12],
            "exchcd": [1, 1, 1],
        }
    )
    firm.to_csv(firm_path, index=False)

    signals = pd.DataFrame(
        {
            "permno": [1, 1, 1],
            "date": ["1987-12-01", "1988-01-01", "1988-02-01"],
            "AnalystRevision": [1.0, 1.1, 1.2],
            "REV6": [0.1, 0.2, 0.3],
            "ForecastDispersion": [0.01, 0.02, 0.03],
            "FEPS": [2.0, 2.1, 2.2],
        }
    )
    signals.to_csv(signal_path, index=False)

    metadata = pd.DataFrame(
        {
            "Acronym": ["AnalystRevision", "REV6", "ForecastDispersion", "FEPS"],
            "Frequency": ["Annual", "Monthly", "Annual", "Annual"],
            "Cat.Data": ["Analyst"] * 4,
            "Cat.Economic": ["earnings forecast", "earnings forecast", "volatility", "profitability"],
            "SampleStartYear": [1975, 1977, 1976, 1983],
            "SampleEndYear": [1980, 1992, 2000, 2002],
            "Detailed Definition": ["ar", "rev6", "disp", "feps"],
        }
    )
    metadata.to_csv(metadata_path, index=False)

    outputs = run_phase0_audit(
        paths=Phase0Paths(
            signal_panel_path=signal_path,
            firm_panel_path=firm_path,
            signal_metadata_path=metadata_path,
            output_root=output_root,
        ),
    )

    panel = outputs["panel_base"]
    assert panel["date"].min() == pd.Timestamp("1988-01-31")
    assert len(panel) == 2


def test_run_phase0_audit_removes_permnos_with_zero_analyst_coverage() -> None:
    tmp_path = make_workspace_tmp_dir("phase0_zero_coverage")
    firm_path = tmp_path / "firm.csv"
    signal_path = tmp_path / "signals.csv"
    metadata_path = tmp_path / "SignalDoc.csv"
    output_root = tmp_path / "outputs"

    firm = pd.DataFrame(
        {
            "permno": [1, 1, 2, 2],
            "date": ["2020-01-31", "2020-02-29", "2020-01-31", "2020-02-29"],
            "ret": [0.1, 0.2, -0.1, -0.2],
            "mve": [10, 11, 20, 21],
            "exchcd": [1, 1, 3, 3],
        }
    )
    firm.to_csv(firm_path, index=False)

    signals = pd.DataFrame(
        {
            "permno": [1, 1, 2, 2],
            "date": ["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"],
            "AnalystRevision": [1.5, 1.6, pd.NA, pd.NA],
            "REV6": [0.3, 0.4, pd.NA, pd.NA],
            "ForecastDispersion": [0.05, pd.NA, pd.NA, pd.NA],
            "FEPS": [3.0, pd.NA, pd.NA, pd.NA],
        }
    )
    signals.to_csv(signal_path, index=False)

    metadata = pd.DataFrame(
        {
            "Acronym": ["AnalystRevision", "REV6", "ForecastDispersion", "FEPS"],
            "Frequency": ["Annual", "Monthly", "Annual", "Annual"],
            "Cat.Data": ["Analyst"] * 4,
            "Cat.Economic": ["earnings forecast", "earnings forecast", "volatility", "profitability"],
            "SampleStartYear": [1975, 1977, 1976, 1983],
            "SampleEndYear": [1980, 1992, 2000, 2002],
            "Detailed Definition": ["ar", "rev6", "disp", "feps"],
        }
    )
    metadata.to_csv(metadata_path, index=False)

    outputs = run_phase0_audit(
        paths=Phase0Paths(
            signal_panel_path=signal_path,
            firm_panel_path=firm_path,
            signal_metadata_path=metadata_path,
            output_root=output_root,
        ),
    )

    panel = outputs["panel_base"]
    diagnostics = outputs["id_merge_diagnostics"]
    assert panel["permno"].tolist() == [1, 1]
    removed_permnos = diagnostics.loc[diagnostics["metric"].eq("out_of_universe_permnos_removed"), "value"].iloc[0]
    removed_rows = diagnostics.loc[diagnostics["metric"].eq("out_of_universe_rows_removed"), "value"].iloc[0]
    assert removed_permnos == 1
    assert removed_rows == 2
