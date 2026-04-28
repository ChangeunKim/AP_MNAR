from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ap_mnar.reporting.figures import plot_missingness_heatmap, plot_target_distribution
from ap_mnar.reporting.tables import build_summary_statistics_table


def make_workspace_tmp_dir(name: str) -> Path:
    path = REPO_ROOT / ".tmp_test_artifacts" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


@pytest.fixture
def mock_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "permno": [1, 1, 2, 2],
            "date": pd.to_datetime(["2020-01-31", "2020-12-31", "2020-01-31", "2021-01-31"]),
            "ret_fwd_1m": [0.05, -0.02, 0.01, pd.NA],
            "REV6": [1.1, 1.2, pd.NA, 1.4],
            "REV6_tier1": [1.1, 1.2, pd.NA, 1.4],
            "FEPS": [pd.NA, 2.5, 2.0, 2.1],
            "FEPS_tier1": [pd.NA, pd.NA, pd.NA, 2.0],  # Mock shift
        }
    )


def test_build_summary_statistics_table(mock_panel: pd.DataFrame) -> None:
    signals = ["REV6", "FEPS"]
    stats = build_summary_statistics_table(mock_panel, signals)

    # Asserts
    assert "ret_fwd_1m" in stats["variable"].values
    assert "REV6" in stats["variable"].values
    assert "REV6_tier1" in stats["variable"].values
    assert "FEPS" in stats["variable"].values
    assert "FEPS_tier1" in stats["variable"].values

    # Check correct counts
    feps_count = stats.loc[stats["variable"] == "FEPS", "non_missing_count"].iloc[0]
    assert feps_count == 3.0

    ret_count = stats.loc[stats["variable"] == "ret_fwd_1m", "non_missing_count"].iloc[0]
    assert ret_count == 3.0


def test_plot_missingness_heatmap(mock_panel: pd.DataFrame) -> None:
    tmp_path = make_workspace_tmp_dir("figures")
    output_path = tmp_path / "heatmap_test.png"
    signals = ["REV6", "FEPS"]

    plot_missingness_heatmap(mock_panel, signals, output_path)

    # Test file was created successfully if matplotlib is installed
    try:
        import matplotlib.pyplot
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    except ImportError:
        pass


def test_plot_target_distribution(mock_panel: pd.DataFrame) -> None:
    tmp_path = make_workspace_tmp_dir("figures")
    output_path = tmp_path / "distribution_test.png"

    plot_target_distribution(mock_panel, output_path)

    # Test file was created successfully if matplotlib is installed
    try:
        import matplotlib.pyplot
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    except ImportError:
        pass
