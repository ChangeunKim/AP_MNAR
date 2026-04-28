from __future__ import annotations

import pandas as pd


def add_forward_return(
    frame: pd.DataFrame,
    id_col: str = "permno",
    date_col: str = "date",
    return_col: str = "ret",
    horizon: int = 1,
) -> pd.DataFrame:
    result = frame.sort_values([id_col, date_col]).copy()
    result[f"{return_col}_fwd_{horizon}m"] = result.groupby(id_col, sort=False)[return_col].shift(-horizon)
    return result

