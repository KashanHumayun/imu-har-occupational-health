from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the IMU-based human activity recognition pipeline on real UCI datasets."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/results"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/results"))
    parser.add_argument("--train-lstm", action="store_true")
    parser.add_argument("--pamap2-subject-limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_pipeline(
        project_root=args.project_root.resolve(),
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        train_lstm=args.train_lstm,
        pamap2_subject_limit=args.pamap2_subject_limit,
    )
    print(
        json.dumps(
            {
                "uci_har_best_model": result["uci_har"]["best_model"],
                "pamap2_best_model": result["pamap2"]["best_model"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
