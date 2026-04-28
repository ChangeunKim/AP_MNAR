from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

# If matplotlib is not installed, fail gracefully during reporting rather than crashing the pipeline
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib is not installed. Figure generation will be skipped.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def plot_missingness_heatmap(
    panel: pd.DataFrame, 
    signals: Sequence[str], 
    output_path: Path
) -> None:
    """Plot a bar chart of missingness rate by year natively using pandas and save it."""
    if not MATPLOTLIB_AVAILABLE:
        return

    # Calculate coverage by year
    if "year" not in panel.columns:
        panel_with_year = panel.copy()
        panel_with_year["year"] = panel_with_year["date"].dt.year
    else:
        panel_with_year = panel

    yearly_coverage = panel_with_year.groupby("year")[signals].apply(lambda x: x.notna().mean())
    
    # Exclude the last year as it is likely incomplete and distorts the chart
    if not yearly_coverage.empty:
        max_year = yearly_coverage.index.max()
        yearly_coverage = yearly_coverage[yearly_coverage.index < max_year]
    
    plt.figure(figsize=(10, 6))
    yearly_coverage.plot(kind='line', marker="o", ax=plt.gca())
    plt.title("Signal Observation Rate (Raw) over Years")
    plt.ylabel("Observation Rate (0.0 - 1.0)")
    plt.xlabel("Year")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Signals")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_residual_missingness_heatmap(
    summary_by_year: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot residual missingness rate among eligible observations by year and signal."""
    if not MATPLOTLIB_AVAILABLE or summary_by_year.empty:
        return

    heatmap_data = summary_by_year.pivot(
        index="signal",
        columns="year",
        values="residual_missing_rate_given_eligible",
    )
    if heatmap_data.empty:
        return

    plt.figure(figsize=(12, 4))
    if SEABORN_AVAILABLE:
        sns.heatmap(
            heatmap_data,
            cmap="YlOrRd",
            cbar_kws={"label": "Residual missingness rate | eligible"},
        )
    else:
        plt.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")
        plt.colorbar(label="Residual missingness rate | eligible")
        plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
        plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=45)

    plt.title("Residual Missingness Rate among Eligible Observations")
    plt.xlabel("Year")
    plt.ylabel("Signal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_target_distribution(
    panel: pd.DataFrame, 
    output_path: Path
) -> None:
    """Plot histogram of the return target ret_fwd_1m."""
    if not MATPLOTLIB_AVAILABLE:
        return

    if "ret_fwd_1m" not in panel.columns:
        return

    plt.figure(figsize=(8, 5))
    panel["ret_fwd_1m"].dropna().hist(bins=100, color="skyblue", edgecolor="black", alpha=0.7)
    plt.title("Distribution of 1-Month Forward Returns (ret_fwd_1m)")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_jtest_pvalue_distribution(
    jtest_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot histogram of J-test p-values."""
    if not MATPLOTLIB_AVAILABLE or jtest_results.empty:
        return

    pvalues = jtest_results["j_p_value"].dropna()
    if pvalues.empty:
        return

    plt.figure(figsize=(8, 5))
    pvalues.hist(bins=20, color="salmon", edgecolor="black", alpha=0.7)
    plt.title("Distribution of Phase 2 J-test p-values")
    plt.xlabel("J-test p-value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_counterfactual_sensitivity_by_signal(
    oos_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot grouped OOS R^2 bars by signal and counterfactual regime."""
    if not MATPLOTLIB_AVAILABLE or oos_results.empty:
        return

    plot_frame = oos_results.dropna(subset=["oos_r2"]).copy()
    if plot_frame.empty:
        return

    pivot = plot_frame.pivot(index="signal", columns="regime", values="oos_r2")
    if pivot.empty:
        return

    pivot = pivot.reindex(
        columns=[
            col
            for col in [
                "complete_case",
                "unconditional_mean",
                "conditional_mean",
                "residual_bootstrap",
                "conditional_quantile_draw",
            ]
            if col in pivot.columns
        ]
    )
    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Counterfactual OOS Sensitivity by Signal")
    ax.set_xlabel("Signal")
    ax.set_ylabel("OOS R^2")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Regime")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_counterfactual_delta_r2_by_signal_group(
    signal_sorted_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot delta OOS R^2 against complete case across imputed-signal quantile groups."""
    if not MATPLOTLIB_AVAILABLE or signal_sorted_results.empty:
        return

    plot_frame = signal_sorted_results.loc[
        signal_sorted_results["regime"].ne("complete_case")
    ].dropna(subset=["delta_oos_r2_vs_complete_case"]).copy()
    if plot_frame.empty:
        return

    signals = list(plot_frame["signal"].dropna().unique())
    if not signals:
        return

    fig, axes = plt.subplots(len(signals), 1, figsize=(10, 3.5 * len(signals)), sharex=True)
    if len(signals) == 1:
        axes = [axes]

    regime_order = [
        "unconditional_mean",
        "conditional_mean",
        "residual_bootstrap",
        "conditional_quantile_draw",
    ]
    for ax, signal in zip(axes, signals):
        subset = plot_frame.loc[plot_frame["signal"].eq(signal)].copy()
        if subset.empty:
            continue
        for regime in regime_order:
            regime_subset = subset.loc[subset["regime"].eq(regime)].sort_values("signal_sort_group")
            if regime_subset.empty:
                continue
            ax.plot(
                regime_subset["signal_sort_group"],
                regime_subset["delta_oos_r2_vs_complete_case"],
                marker="o",
                label=regime,
            )

        ax.set_title(f"{signal}: Delta OOS R^2 by Imputed Signal Group")
        ax.set_ylabel("Delta OOS R^2")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.legend(title="Regime")

    axes[-1].set_xlabel("Imputed Signal Quantile Group")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_missingness_alpha_by_signal(
    fama_macbeth_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the mean missingness premium by signal from the baseline Fama-MacBeth spec."""
    if not MATPLOTLIB_AVAILABLE or fama_macbeth_results.empty:
        return

    plot_frame = fama_macbeth_results.loc[
        fama_macbeth_results["sample_name"].eq("full")
        & fama_macbeth_results["specification"].eq("baseline_missing_only")
        & fama_macbeth_results["coefficient"].eq("missing_indicator")
    ].copy()
    if plot_frame.empty:
        return

    plot_frame = plot_frame.sort_values("mean_coefficient")
    ax = plot_frame.plot(
        x="signal",
        y="mean_coefficient",
        kind="bar",
        legend=False,
        figsize=(9, 5),
    )
    ax.set_title("Phase 4 Missingness Premium by Signal")
    ax.set_xlabel("Signal")
    ax.set_ylabel("Mean Fama-MacBeth Coefficient on Missingness")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_missingness_premium_over_time(
    premium_time_series: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the monthly missingness premium time series for the baseline pricing spec."""
    if not MATPLOTLIB_AVAILABLE or premium_time_series.empty:
        return

    plot_frame = premium_time_series.loc[premium_time_series["sample_name"].eq("full")].copy()
    if plot_frame.empty:
        return

    plt.figure(figsize=(10, 6))
    for signal, subset in plot_frame.groupby("signal"):
        subset = subset.sort_values("date")
        plt.plot(
            subset["date"],
            subset["coefficient_value"],
            label=signal,
            alpha=0.85,
        )

    plt.title("Phase 4 Monthly Missingness Premium")
    plt.xlabel("Date")
    plt.ylabel("Monthly Fama-MacBeth Coefficient on Missingness")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Signal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
