from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


TIER1_DELAY_BY_FREQUENCY = {
    "Monthly": 0,
    "Quarterly": 3,
    "Annual": 6,
}


def load_signal_metadata(
    path: str | Path,
    signals: Iterable[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    subset = frame.loc[frame["Acronym"].isin(list(signals))].copy()
    subset["tier1_delay_months"] = subset["Frequency"].map(TIER1_DELAY_BY_FREQUENCY).fillna(0).astype(int)
    subset = subset.rename(
        columns={
            "Acronym": "signal",
            "Cat.Data": "cat_data",
            "Cat.Economic": "cat_economic",
            "Detailed Definition": "detailed_definition",
        }
    )
    return subset[
        [
            "signal",
            "Frequency",
            "cat_data",
            "cat_economic",
            "SampleStartYear",
            "SampleEndYear",
            "detailed_definition",
            "tier1_delay_months",
        ]
    ].sort_values("signal", ignore_index=True)

