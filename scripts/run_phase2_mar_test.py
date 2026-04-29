from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ap_mnar.experiments.phase2_mar_test import Phase2Paths, run_phase2_mar_test
from ap_mnar.experiments.step0_audit import DEFAULT_SIGNALS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 MAR falsification diagnostics for AP_MNAR.")
    parser.add_argument(
        "--panel-with-missingness",
        default=str(REPO_ROOT / "outputs" / "interim" / "panel_with_missingness.parquet"),
        help="Path to the Phase 1 panel_with_missingness parquet.",
    )
    parser.add_argument(
        "--signal-registry",
        default=str(REPO_ROOT / "outputs" / "tables" / "signal_registry.csv"),
        help="Path to the Phase 0 signal_registry CSV.",
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
        help="Directory where Phase 2 outputs will be written.",
    )
    parser.add_argument(
        "--signals",
        nargs="*",
        default=list(DEFAULT_SIGNALS),
        help="Signals to include in the Phase 2 MAR test.",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=10,
        help="Number of MAR-implied counterfactual draws used in the signal-level Phase 2 tests.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Base random seed used for MAR-implied counterfactual draws.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of time-based folds used for cross-fit conditional-mean prediction.",
    )
    parser.add_argument(
        "--mar-draw-regime",
        choices=["conditional_quantile_draw", "residual_bootstrap"],
        default="conditional_quantile_draw",
        help="MAR counterfactual draw mechanism used in the Step 1A signal-level test.",
    )
    parser.add_argument(
        "--disable-signal-history-augmentation",
        action="store_true",
        help="Disable the additional augmented benchmark and run only the fixed-X_obs Phase 2 benchmark.",
    )
    parser.add_argument(
        "--disable-pattern-slices",
        action="store_true",
        help="Disable Phase 2 pattern-sliced diagnostics for start/middle/end/always-missing rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Phase2Paths(
        panel_with_missingness_path=Path(args.panel_with_missingness),
        signal_registry_path=Path(args.signal_registry),
        x_obs_config_path=Path(args.x_obs_config),
        firm_panel_path=Path(args.firm_panel),
        output_root=Path(args.output_root),
    )
    outputs = run_phase2_mar_test(
        paths=paths,
        signals=args.signals,
        n_draws=args.n_draws,
        random_seed=args.random_seed,
        n_folds=args.n_folds,
        mar_draw_regime=args.mar_draw_regime,
        augment_signal_history=not args.disable_signal_history_augmentation,
        include_pattern_slices=not args.disable_pattern_slices,
    )
    print("Phase 2 completed.")
    print(f"signals: {', '.join(args.signals)}")
    print(f"signal-diagnostic rows: {len(outputs['stage1_table'])}")
    print(f"signal-jtest rows: {len(outputs['jtest_table'])}")
    print(f"output root: {paths.output_root}")


if __name__ == "__main__":
    main()
