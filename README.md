# IMU-Based Human Activity Recognition for Occupational Health

End-to-end wearable-sensor pipeline for classifying occupationally relevant activities such as walking, sitting, standing, and lifting. The repo includes a runnable synthetic-data demo so the pipeline works immediately, plus code that mirrors the real project workflow used for UCI HAR or PAMAP2-style sensor streams.

## What This Repo Includes

- Sliding-window segmentation of continuous IMU streams
- Statistical and FFT-derived feature engineering
- Cross-subject evaluation with Random Forest and SVM baselines
- Optional PyTorch LSTM sequence classifier
- HMM-style workflow smoothing over continuous task sequences
- CLI entry point that writes metrics, model artifacts, and demo predictions

## Quick Start

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m imu_har.cli --output-dir reports/demo
```

To skip the neural baseline:

```bash
python -m imu_har.cli --skip-lstm
```

## Project Structure

- `src/imu_har/` package code for feature extraction, modeling, and CLI entry points
- `tests/` smoke test for the demo pipeline
- `reports/` generated metrics and prediction exports
- `data/` place raw UCI HAR, PAMAP2, or custom wearable data here
- `models/` reserved for longer-lived trained artifacts

## Adapting To Real Data

The current demo generates structured synthetic IMU streams with subject-specific variation so the full training and evaluation loop can be exercised. To plug in real datasets:

1. Replace the synthetic generator with your dataset loader.
2. Preserve the expected columns: subject identifier, workflow identifier, timestamp, activity label, and six IMU channels.
3. Reuse `segment_windows`, `extract_window_features`, and the evaluation utilities in `src/imu_har/pipeline.py`.

## Output Artifacts

Running the demo writes:

- `reports/demo/metrics.json`
- `reports/demo/window_predictions.csv`
- `reports/demo/random_forest_model.joblib` or the best-performing model artifact
- `reports/demo/synthetic_sample.csv`
