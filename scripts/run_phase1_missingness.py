from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ap_mnar.experiments.phase1_missingness import Phase1Paths, run_phase1_missingness
from ap_mnar.experiments.step0_audit import DEFAULT_SIGNALS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 missingness classification for AP_MNAR.")
    parser.add_argument(
        "--panel-base",
        default=str(REPO_ROOT / "outputs" / "interim" / "panel_base.parquet"),
        help="Path to the Phase 0 panel_base parquet.",
    )
    parser.add_argument(
        "--signal-registry",
        default=str(REPO_ROOT / "outputs" / "tables" / "signal_registry.csv"),
        help="Path to the Phase 0 signal_registry CSV.",
    )
    parser.add_argument(
        "--missingness-rules",
        default=str(REPO_ROOT / "configs" / "missingness_rules.yaml"),
        help="Path to the Phase 1 missingness rules YAML.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "outputs"),
        help="Directory where Phase 1 outputs will be written.",
    )
    parser.add_argument(
        "--signals",
        nargs="*",
        default=list(DEFAULT_SIGNALS),
        help="Signals to include in the Phase 1 classification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Phase1Paths(
        panel_base_path=Path(args.panel_base),
        signal_registry_path=Path(args.signal_registry),
        missingness_rules_path=Path(args.missingness_rules),
        output_root=Path(args.output_root),
    )
    outputs = run_phase1_missingness(paths=paths, signals=args.signals)
    print("Phase 1 completed.")
    print(f"panel rows: {len(outputs['panel_with_missingness'])}")
    print(f"signals: {', '.join(args.signals)}")
    print(f"output root: {paths.output_root}")


if __name__ == "__main__":
    main()
