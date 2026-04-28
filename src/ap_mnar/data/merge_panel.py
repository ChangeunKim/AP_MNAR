from __future__ import annotations

from typing import Iterable

import pandas as pd
from pandas.tseries.offsets import MonthEnd


def assign_size_bucket(
    frame: pd.DataFrame,
    size_col: str = "mve",
    date_col: str = "date",
    buckets: int = 5,
) -> pd.Series:
    def bucket_one_month(values: pd.Series) -> pd.Series:
        if values.notna().sum() < buckets:
            return pd.Series(pd.NA, index=values.index, dtype="Int64")
        ranked = values.rank(method="first")
        bucketed = pd.qcut(ranked, q=buckets, labels=range(1, buckets + 1))
        return pd.Series(bucketed.astype("Int64"), index=values.index)

    return frame.groupby(date_col, group_keys=False)[size_col].apply(bucket_one_month)


def build_lagged_signal_panel(
    signal_panel: pd.DataFrame,
    signal_registry: pd.DataFrame,
    id_col: str = "permno",
    date_col: str = "date",
) -> pd.DataFrame:
    lagged_frames: list[pd.DataFrame] = []
    keys = signal_panel[[id_col, date_col]].drop_duplicates().copy()

    for row in signal_registry.itertuples(index=False):
        signal = row.signal
        delay = int(row.tier1_delay_months)
        temp = signal_panel[[id_col, date_col, signal]].copy()
        temp[date_col] = temp[date_col] + MonthEnd(delay)
        temp = temp.rename(columns={signal: f"{signal}_tier1"})
        temp = temp.groupby([id_col, date_col], as_index=False).last()
        lagged_frames.append(temp)

    result = keys.copy()
    for temp in lagged_frames:
        result = result.merge(temp, on=[id_col, date_col], how="outer")
    return result


def build_canonical_panel(
    firm_panel: pd.DataFrame,
    signal_panel: pd.DataFrame,
    signal_registry: pd.DataFrame,
    id_col: str = "permno",
    date_col: str = "date",
) -> pd.DataFrame:
    raw_merged = firm_panel.merge(signal_panel, on=[id_col, date_col], how="left")
    lagged_panel = build_lagged_signal_panel(signal_panel, signal_registry, id_col=id_col, date_col=date_col)
    merged = raw_merged.merge(lagged_panel, on=[id_col, date_col], how="left")
    merged["size_bucket"] = assign_size_bucket(merged, size_col="mve", date_col=date_col)
    merged["year"] = merged[date_col].dt.year.astype("Int64")
    return merged.sort_values([id_col, date_col]).reset_index(drop=True)


def filter_out_of_universe_permnos(
    panel: pd.DataFrame,
    signals: Iterable[str],
    id_col: str = "permno",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_cols = [signal for signal in signals if signal in panel.columns]
    if not signal_cols:
        diagnostics = pd.DataFrame(
            [
                {"metric": "out_of_universe_permnos_removed", "value": 0},
                {"metric": "out_of_universe_rows_removed", "value": 0},
            ]
        )
        return panel.copy(), diagnostics

    permno_has_any_signal = panel.groupby(id_col)[signal_cols].apply(lambda frame: frame.notna().any().any())
    keep_permnos = permno_has_any_signal.index[permno_has_any_signal]
    removed_permnos = permno_has_any_signal.index[~permno_has_any_signal]

    filtered = panel.loc[panel[id_col].isin(keep_permnos)].copy()
    diagnostics = pd.DataFrame(
        [
            {"metric": "out_of_universe_permnos_removed", "value": int(len(removed_permnos))},
            {
                "metric": "out_of_universe_rows_removed",
                "value": int(panel.loc[panel[id_col].isin(removed_permnos)].shape[0]),
            },
        ]
    )
    return filtered.reset_index(drop=True), diagnostics


def build_id_merge_diagnostics(
    firm_panel: pd.DataFrame,
    signal_panel: pd.DataFrame,
    canonical_panel: pd.DataFrame,
    signals: Iterable[str],
    extra_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    lagged_signals = [f"{signal}_tier1" for signal in signals]
    diagnostics = [
        ("firm_rows", len(firm_panel)),
        ("firm_permnos", firm_panel["permno"].nunique(dropna=True)),
        ("signal_rows", len(signal_panel)),
        ("signal_permnos", signal_panel["permno"].nunique(dropna=True)),
        ("canonical_rows", len(canonical_panel)),
        ("canonical_permnos", canonical_panel["permno"].nunique(dropna=True)),
        ("rows_with_ret", canonical_panel["ret"].notna().sum()),
        ("rows_with_ret_fwd_1m", canonical_panel["ret_fwd_1m"].notna().sum()),
        ("rows_with_any_raw_signal", canonical_panel[list(signals)].notna().any(axis=1).sum()),
        ("rows_with_any_tier1_signal", canonical_panel[lagged_signals].notna().any(axis=1).sum()),
    ]
    diagnostics_frame = pd.DataFrame(diagnostics, columns=["metric", "value"])
    if extra_metrics is not None and not extra_metrics.empty:
        diagnostics_frame = pd.concat([diagnostics_frame, extra_metrics], ignore_index=True)
    return diagnostics_frame
