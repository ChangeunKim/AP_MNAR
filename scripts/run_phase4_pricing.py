from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ap_mnar.experiments.phase4_pricing import Phase4Paths, run_phase4_pricing
from ap_mnar.experiments.step0_audit import DEFAULT_SIGNALS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 pricing-of-missingness analysis for AP_MNAR.")
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
        help="Directory where Phase 4 outputs will be written.",
    )
    parser.add_argument(
        "--signals",
        nargs="*",
        default=list(DEFAULT_SIGNALS),
        help="Signals to include in the Phase 4 pricing analysis.",
    )
    parser.add_argument(
        "--min-train-months",
        type=int,
        default=60,
        help="Minimum number of monthly training periods before walk-forward MAR x_hat evaluation begins.",
    )
    parser.add_argument(
        "--min-cross-section",
        type=int,
        default=25,
        help="Minimum number of stocks required for a monthly Fama-MacBeth cross-section.",
    )
    parser.add_argument(
        "--nw-lags",
        type=int,
        default=6,
        help="Newey-West lag length used when aggregating monthly Fama-MacBeth coefficients.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars during long Phase 4 runs.",
    )
    parser.add_argument(
        "--disable-signal-history-augmentation",
        action="store_true",
        help="Run only the fixed_x_obs benchmark and skip the augmented_signal_history benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Phase4Paths(
        panel_with_missingness_path=Path(args.panel_with_missingness),
        x_obs_config_path=Path(args.x_obs_config),
        firm_panel_path=Path(args.firm_panel),
        output_root=Path(args.output_root),
    )
    outputs = run_phase4_pricing(
        paths=paths,
        signals=args.signals,
        min_train_months=args.min_train_months,
        min_cross_section=args.min_cross_section,
        nw_lags=args.nw_lags,
        show_progress=not args.no_progress,
        include_augmented_signal_history=not args.disable_signal_history_augmentation,
    )
    print("Phase 4 completed.")
    print(f"signals: {', '.join(args.signals)}")
    print(f"pricing panel rows: {len(outputs['pricing_panel'])}")
    print(f"fama-macbeth rows: {len(outputs['fama_macbeth_table'])}")
    print(f"output root: {paths.output_root}")


if __name__ == "__main__":
    main()
