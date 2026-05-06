from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ap_mnar.experiments.phase3_counterfactual import Phase3Paths, run_phase3_counterfactual
from ap_mnar.experiments.step0_audit import DEFAULT_SIGNALS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 counterfactual characterization for AP_MNAR.")
    parser.add_argument(
        "--panel-with-missingness",
        default=str(REPO_ROOT / "outputs" / "interim" / "panel_with_missingness.parquet"),
        help="Path to the Phase 1 panel_with_missingness parquet.",
    )
    parser.add_argument(
        "--x-obs-config",
        default=str(REPO_ROOT / "configs" / "x_obs.yaml"),
        help="Path to the fixed X_obs configuration YAML.",
    )
    parser.add_argument(
        "--firm-panel",
        default=str(REPO_ROOT / "data" / "raw" / "firm_characs.csv"),
        help="Path to the firm characteristic panel used to source X_obs columns.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "outputs"),
        help="Directory where Phase 3 outputs will be written.",
    )
    parser.add_argument(
        "--signals",
        nargs="*",
        default=list(DEFAULT_SIGNALS),
        help="Signals to include in the Phase 3 counterfactual backtest.",
    )
    parser.add_argument(
        "--min-train-years",
        type=int,
        default=5,
        help="Minimum number of calendar years used for the expanding-window training sample.",
    )
    parser.add_argument(
        "--stochastic-draws",
        type=int,
        default=25,
        help="Number of Monte Carlo draws used for stochastic counterfactual regimes.",
    )
    parser.add_argument(
        "--quantile-bins",
        type=int,
        default=5,
        help="Number of train-sample conditional-mean quantile bins used for conditional quantile draws.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Base random seed for stochastic counterfactual draws.",
    )
    parser.add_argument(
        "--signal-sort-groups",
        type=int,
        default=5,
        help="Number of quantile groups used for signal-sorted counterfactual diagnostics.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars during long Phase 3 runs.",
    )
    parser.add_argument(
        "--disable-signal-history-augmentation",
        action="store_true",
        help="Run only the fixed_x_obs benchmark and skip the augmented_signal_history benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Phase3Paths(
        panel_with_missingness_path=Path(args.panel_with_missingness),
        x_obs_config_path=Path(args.x_obs_config),
        firm_panel_path=Path(args.firm_panel),
        output_root=Path(args.output_root),
    )
    outputs = run_phase3_counterfactual(
        paths=paths,
        signals=args.signals,
        min_train_years=args.min_train_years,
        stochastic_draws=args.stochastic_draws,
        quantile_bins=args.quantile_bins,
        random_seed=args.random_seed,
        signal_sort_groups=args.signal_sort_groups,
        show_progress=not args.no_progress,
        include_augmented_signal_history=not args.disable_signal_history_augmentation,
    )
    print("Phase 3 completed.")
    print(f"signals: {', '.join(args.signals)}")
    print(f"oos rows: {len(outputs['oos_table'])}")
    print(f"portfolio rows: {len(outputs['portfolio_table'])}")
    print(f"output root: {paths.output_root}")


if __name__ == "__main__":
    main()
