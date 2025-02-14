from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the IMU-based human activity recognition demo pipeline."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/demo"))
    parser.add_argument("--subjects", type=int, default=8)
    parser.add_argument("--sequences-per-subject", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=320)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--step-size", type=int, default=32)
    parser.add_argument("--lstm-epochs", type=int, default=4)
    parser.add_argument("--skip-lstm", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_demo(
        output_dir=args.output_dir,
        n_subjects=args.subjects,
        sequences_per_subject=args.sequences_per_subject,
        sequence_length=args.sequence_length,
        window_size=args.window_size,
        step_size=args.step_size,
        lstm_epochs=args.lstm_epochs,
        include_lstm=not args.skip_lstm,
    )
    print(json.dumps(result["best_model"], indent=2))


if __name__ == "__main__":
    main()
