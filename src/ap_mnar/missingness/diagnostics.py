from __future__ import annotations

from typing import Sequence

import pandas as pd


def build_missingness_summary_by_signal(
    panel: pd.DataFrame,
    signal_registry: pd.DataFrame,
    support_windows: pd.DataFrame,
    signals: Sequence[str],
) -> pd.DataFrame:
    total_count = len(panel)
    rows = []

    registry_lookup = signal_registry.set_index("signal")
    support_lookup = support_windows.set_index("signal")

    for signal in signals:
        eligible_count = int(panel[f"{signal}_eligible"].sum())
        residual_missing_count = int(panel[f"{signal}_residual_missing"].sum())
        tier1_missing_count = int(panel[f"{signal}_tier1_missing"].sum())
        observed_eligible_count = int(panel[f"{signal}_observed_eligible"].sum())

        registry_row = registry_lookup.loc[signal]
        support_row = support_lookup.loc[signal]

        rows.append(
            {
                "signal": signal,
                "frequency": registry_row["Frequency"],
                "tier1_delay_months": int(registry_row["tier1_delay_months"]),
                "support_start_date": support_row["support_start_date"],
                "support_end_date": support_row["support_end_date"],
                "total_count": total_count,
                "raw_missing_count": int(panel[f"{signal}_raw_missing"].sum()),
                "tier1_missing_count": tier1_missing_count,
                "eligible_count": eligible_count,
                "ineligible_count": total_count - eligible_count,
                "observed_eligible_count": observed_eligible_count,
                "residual_missing_count": residual_missing_count,
                "missing_excluded_by_eligibility": tier1_missing_count - residual_missing_count,
                "eligibility_rate": eligible_count / total_count if total_count else 0.0,
                "residual_missing_rate_total": residual_missing_count / total_count if total_count else 0.0,
                "residual_missing_rate_given_eligible": residual_missing_count / eligible_count if eligible_count else 0.0,
                "eligible_observation_rate": observed_eligible_count / eligible_count if eligible_count else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values("signal", ignore_index=True)


def build_missingness_summary_by_year(
    panel: pd.DataFrame,
    signals: Sequence[str],
) -> pd.DataFrame:
    rows = []
    if "year" not in panel.columns:
        panel = panel.copy()
        panel["year"] = pd.to_datetime(panel["date"]).dt.year.astype("Int64")

    for signal in signals:
        for year, subset in panel.groupby("year", dropna=False):
            total_count = len(subset)
            eligible_count = int(subset[f"{signal}_eligible"].sum())
            residual_missing_count = int(subset[f"{signal}_residual_missing"].sum())
            observed_eligible_count = int(subset[f"{signal}_observed_eligible"].sum())
            rows.append(
                {
                    "signal": signal,
                    "year": year,
                    "total_count": total_count,
                    "eligible_count": eligible_count,
                    "observed_eligible_count": observed_eligible_count,
                    "residual_missing_count": residual_missing_count,
                    "residual_missing_rate_total": residual_missing_count / total_count if total_count else 0.0,
                    "residual_missing_rate_given_eligible": residual_missing_count / eligible_count if eligible_count else 0.0,
                    "eligible_observation_rate": observed_eligible_count / eligible_count if eligible_count else 0.0,
                }
            )

    return pd.DataFrame(rows).sort_values(["signal", "year"], ignore_index=True)


def build_missingness_removed_by_eligibility(
    panel: pd.DataFrame,
    signals: Sequence[str],
) -> pd.DataFrame:
    rows = []

    for signal in signals:
        tier1_missing_count = int(panel[f"{signal}_tier1_missing"].sum())
        residual_missing_count = int(panel[f"{signal}_residual_missing"].sum())
        removed = tier1_missing_count - residual_missing_count
        rows.append(
            {
                "signal": signal,
                "tier1_missing_count": tier1_missing_count,
                "residual_missing_count": residual_missing_count,
                "missing_excluded_by_eligibility": removed,
                "excluded_share_of_tier1_missing": removed / tier1_missing_count if tier1_missing_count else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values("signal", ignore_index=True)


def build_missingness_pattern_counts(panel: pd.DataFrame) -> pd.DataFrame:
    counts = (
        panel.groupby(["missing_pattern_code", "missing_pattern_label"], dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values("row_count", ascending=False, ignore_index=True)
    )
    counts["share_of_rows"] = counts["row_count"] / len(panel) if len(panel) else 0.0
    return counts
