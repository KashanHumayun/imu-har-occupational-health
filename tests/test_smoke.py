from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from imu_har.pipeline import run_demo


class HarPipelineSmokeTest(unittest.TestCase):
    def test_demo_pipeline_creates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            summary = run_demo(
                output_dir=output_dir,
                n_subjects=4,
                sequences_per_subject=2,
                sequence_length=192,
                window_size=48,
                step_size=24,
                lstm_epochs=1,
                include_lstm=False,
            )
            self.assertIn("random_forest", summary["classical_models"])
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "window_predictions.csv").exists())
            self.assertGreater(summary["best_model"]["accuracy"], 0.6)


if __name__ == "__main__":
    unittest.main()
