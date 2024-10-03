# IMU-Based Human Activity Recognition for Occupational Health

End-to-end activity recognition pipeline for occupationally relevant activities using wearable IMU and accelerometer data.

## Scope

- Datasets: UCI HAR and PAMAP2
- Activities: walking, sitting, standing, lifting
- Models: Random Forest, SVM, LSTM
- Sequence modelling: Hidden Markov Models (HMM)

## Planned Workflow

1. Preprocess raw sensor streams
2. Segment signals with sliding windows
3. Extract statistical and FFT-based features
4. Train and evaluate activity classifiers
5. Model multi-step nursing-relevant workflows from continuous streams

## Repository Structure

- `data/` for datasets or processed exports
- `notebooks/` for experiments and analysis
- `src/` for reusable pipeline code
- `models/` for saved checkpoints or artifacts
- `reports/` for figures, tables, and writeups
