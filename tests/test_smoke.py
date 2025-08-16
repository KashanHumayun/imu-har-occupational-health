from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from imu_har.pipeline import run_pipeline


class HarPipelineSmokeTest(unittest.TestCase):
    def test_pipeline_creates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model_dir = output_dir / "models"
            project_root = Path(__file__).resolve().parents[1]
            summary = run_pipeline(
                project_root=project_root,
                output_dir=output_dir,
                model_dir=model_dir,
                train_lstm=False,
                pamap2_subject_limit=3,
            )
            self.assertIn("random_forest", summary["uci_har"]["models"])
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "uci_har_predictions.csv").exists())
            self.assertTrue((output_dir / "pamap2_predictions.csv").exists())
            self.assertGreater(summary["uci_har"]["models"]["svm"]["accuracy"], 0.85)
            self.assertGreater(summary["pamap2"]["classical_models"]["random_forest"]["mean_accuracy"], 0.7)


if __name__ == "__main__":
    unittest.main()
