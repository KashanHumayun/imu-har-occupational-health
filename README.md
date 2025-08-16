# IMU-Based Human Activity Recognition for Occupational Health

Real-data activity-recognition pipeline built on the official UCI HAR and PAMAP2 datasets. The repository now runs against downloaded benchmark files rather than synthetic placeholders and produces saved metrics, predictions, model artifacts, and visual summaries.

![Real-data overview](reports/results/performance_overview.png)

## Datasets

- UCI HAR: subject-disjoint train/test split from smartphone inertial windows
- PAMAP2: raw continuous IMU protocol files windowed into `sitting`, `standing`, and `walking` segments for leave-one-subject-out evaluation

## Current Results

| Dataset | Model | Evaluation | Accuracy | Macro F1 |
| --- | --- | --- | ---: | ---: |
| UCI HAR | Linear SVM | Official subject-disjoint test split | 0.960 | 0.960 |
| UCI HAR | Random Forest | Official subject-disjoint test split | 0.926 | 0.924 |
| PAMAP2 | Random Forest | Leave-one-subject-out on raw-window features | 0.930 | 0.928 |
| PAMAP2 | Linear SVM | Leave-one-subject-out on raw-window features | 0.889 | 0.882 |

## What The Pipeline Does

- loads the real UCI HAR feature matrices and inertial windows
- windows raw PAMAP2 IMU streams into subject-level motion segments
- extracts statistical and FFT-based features for raw PAMAP2 windows
- evaluates Random Forest and SVM baselines
- supports an optional PyTorch LSTM path for UCI HAR inertial windows
- saves reproducible outputs to `reports/results` and `models/results`

## Run It

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m imu_har.cli --output-dir reports/results --model-dir models/results
```

Optional LSTM training:

```bash
python -m imu_har.cli --train-lstm
```

## Output Files

- `reports/results/metrics.json`
- `reports/results/uci_har_predictions.csv`
- `reports/results/pamap2_predictions.csv`
- `reports/results/performance_overview.png`
- `models/results/uci_har_best_model.joblib`
- `models/results/pamap2_best_model.joblib`
- `notebooks/real_data_walkthrough.ipynb`

## Notes

- Raw datasets are downloaded locally into `data/raw/` and intentionally ignored by git.
- The default checked-in results come from the real downloaded datasets, not generated samples.
