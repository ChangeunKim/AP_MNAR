from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ap_mnar.experiments.step0_audit import DEFAULT_SIGNALS, Phase0Paths, run_phase0_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 0 audit for AP_MNAR.")
    parser.add_argument(
        "--signal-panel",
        default=str(REPO_ROOT / "data" / "raw" / "open_source_asset_pricing.csv"),
        help="Path to the signal panel CSV.",
    )
    parser.add_argument(
        "--firm-panel",
        default=str(REPO_ROOT / "data" / "raw" / "firm_characs.csv"),
        help="Path to the firm panel CSV with returns.",
    )
    parser.add_argument(
        "--signal-metadata",
        default=str(REPO_ROOT / "data" / "info" / "SignalDoc.csv"),
        help="Path to the signal metadata CSV.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "outputs"),
        help="Directory where Phase 0 outputs will be written.",
    )
    parser.add_argument(
        "--signals",
        nargs="*",
        default=list(DEFAULT_SIGNALS),
        help="Signals to include in the Phase 0 build.",
    )
    parser.add_argument(
        "--x-obs-cols",
        nargs="*",
        default=[],
        help="Fixed X_obs columns to audit if already mirrored into code.",
    )
    parser.add_argument("--firm-nrows", type=int, default=None, help="Optional row cap for the firm panel.")
    parser.add_argument("--signal-nrows", type=int, default=None, help="Optional row cap for the signal panel.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Phase0Paths(
        signal_panel_path=Path(args.signal_panel),
        firm_panel_path=Path(args.firm_panel),
        signal_metadata_path=Path(args.signal_metadata),
        output_root=Path(args.output_root),
    )
    outputs = run_phase0_audit(
        paths=paths,
        signals=args.signals,
        x_obs_columns=args.x_obs_cols,
        firm_nrows=args.firm_nrows,
        signal_nrows=args.signal_nrows,
    )
    print("Phase 0 completed.")
    print(f"panel rows: {len(outputs['panel_base'])}")
    print(f"signals: {', '.join(args.signals)}")
    print(f"output root: {paths.output_root}")


if __name__ == "__main__":
    main()

