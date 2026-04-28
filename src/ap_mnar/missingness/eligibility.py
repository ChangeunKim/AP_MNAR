from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml


@dataclass(frozen=True)
class MissingnessRules:
    analyst_start_date: pd.Timestamp
    support_window_basis: str
    apply_signal_support_window: bool
    terminal_support_policy: str
    signal_settings: dict[str, dict[str, Any]]


def _to_month_end(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.to_period("M").to_timestamp("M")


def load_missingness_rules(path: str | Path) -> MissingnessRules:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    global_rules = payload.get("global", {})
    return MissingnessRules(
        analyst_start_date=_to_month_end(global_rules.get("analyst_start_date", "1988-01-01")),
        support_window_basis=global_rules.get("support_window_basis", "tier1_nonmissing"),
        apply_signal_support_window=bool(global_rules.get("apply_signal_support_window", True)),
        terminal_support_policy=global_rules.get("terminal_support_policy", "ineligible_outside_support"),
        signal_settings=payload.get("signals", {}),
    )


def compute_signal_support_windows(
    panel: pd.DataFrame,
    signal_registry: pd.DataFrame,
    rules: MissingnessRules,
    signals: Sequence[str],
    date_col: str = "date",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for row in signal_registry.loc[signal_registry["signal"].isin(signals)].itertuples(index=False):
        signal = row.signal
        basis_column = f"{signal}_tier1" if rules.support_window_basis == "tier1_nonmissing" else signal
        nonmissing_dates = panel.loc[panel[basis_column].notna(), date_col]

        if nonmissing_dates.empty:
            support_start = pd.NaT
            support_end = pd.NaT
            has_support = False
        else:
            support_start = max(nonmissing_dates.min(), rules.analyst_start_date)
            support_end = nonmissing_dates.max()
            has_support = support_start <= support_end

        signal_rule = rules.signal_settings.get(signal, {})
        rows.append(
            {
                "signal": signal,
                "frequency": row.Frequency,
                "tier1_delay_months": int(row.tier1_delay_months),
                "support_basis_column": basis_column,
                "support_start_date": support_start,
                "support_end_date": support_end,
                "has_support": has_support,
                "min_history_months": int(signal_rule.get("min_history_months", 0)),
                "special_constraints": ",".join(signal_rule.get("special_constraints", [])),
            }
        )

    return pd.DataFrame(rows).sort_values("signal", ignore_index=True)


def build_eligibility_matrix(
    panel: pd.DataFrame,
    support_windows: pd.DataFrame,
    signals: Sequence[str],
    date_col: str = "date",
) -> pd.DataFrame:
    result = panel[[date_col]].copy()

    for row in support_windows.loc[support_windows["signal"].isin(signals)].itertuples(index=False):
        signal = row.signal
        eligible_col = f"{signal}_eligible"
        before_col = f"{signal}_before_support"
        after_col = f"{signal}_after_support"
        no_support_col = f"{signal}_no_support"

        if not row.has_support or pd.isna(row.support_start_date) or pd.isna(row.support_end_date):
            result[eligible_col] = False
            result[before_col] = False
            result[after_col] = False
            result[no_support_col] = True
            continue

        result[eligible_col] = panel[date_col].between(row.support_start_date, row.support_end_date, inclusive="both")
        result[before_col] = panel[date_col] < row.support_start_date
        result[after_col] = panel[date_col] > row.support_end_date
        result[no_support_col] = False

    return result.drop(columns=[date_col])
