from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - handled gracefully in runtime flow
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

SENSOR_COLUMNS = [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]

ACTIVITY_LABELS = ["walking", "sitting", "standing", "lifting"]
WORKFLOW_TEMPLATES = [
    ["standing", "walking", "lifting", "walking", "sitting"],
    ["sitting", "standing", "walking", "lifting", "standing"],
    ["walking", "walking", "lifting", "standing", "sitting"],
]


@dataclass(slots=True)
class WindowConfig:
    window_size: int = 64
    step_size: int = 32


def generate_synthetic_har_dataset(
    n_subjects: int = 8,
    sequences_per_subject: int = 4,
    sequence_length: int = 320,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a subject-wise wearable-sensor dataset for demo and testing."""

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    activity_params = {
        "walking": {"amp": 1.7, "freq": 1.8, "gyro": 1.2, "vertical": 1.3},
        "sitting": {"amp": 0.2, "freq": 0.2, "gyro": 0.1, "vertical": 0.25},
        "standing": {"amp": 0.35, "freq": 0.35, "gyro": 0.18, "vertical": 0.45},
        "lifting": {"amp": 1.15, "freq": 0.95, "gyro": 1.6, "vertical": 1.9},
    }

    for subject_id in range(1, n_subjects + 1):
        subject_scale = rng.normal(1.0, 0.08, size=len(SENSOR_COLUMNS))
        subject_bias = rng.normal(0.0, 0.1, size=len(SENSOR_COLUMNS))

        for sequence_index in range(sequences_per_subject):
            workflow_id = f"S{subject_id:02d}_W{sequence_index:02d}"
            workflow = WORKFLOW_TEMPLATES[(subject_id + sequence_index) % len(WORKFLOW_TEMPLATES)]
            target_segments = _balanced_segment_lengths(
                total_length=sequence_length,
                segments=len(workflow),
                rng=rng,
            )
            time_cursor = 0
            phase_shift = rng.uniform(0.0, np.pi)

            for activity, block_length in zip(workflow, target_segments):
                params = activity_params[activity]
                t = np.arange(block_length, dtype=float)
                base_wave = np.sin(2 * np.pi * params["freq"] * (t / 50.0) + phase_shift)
                harmonic = np.cos(2 * np.pi * (params["freq"] / 2.0) * (t / 50.0) + phase_shift / 2.0)
                burst = np.sin(2 * np.pi * 0.15 * (t / 50.0) + phase_shift)
                noise = rng.normal(0.0, 0.08, size=(block_length, len(SENSOR_COLUMNS)))

                accel_x = params["amp"] * base_wave + 0.3 * harmonic
                accel_y = params["amp"] * 0.7 * harmonic + 0.2 * burst
                accel_z = params["vertical"] * np.abs(base_wave) + 0.15 * harmonic
                gyro_x = params["gyro"] * np.gradient(base_wave)
                gyro_y = params["gyro"] * np.gradient(harmonic)
                gyro_z = params["gyro"] * 0.5 * burst

                signal_block = np.column_stack(
                    [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
                )
                signal_block = signal_block * subject_scale + subject_bias + noise

                for offset, values in enumerate(signal_block):
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "workflow_id": workflow_id,
                            "timestep": time_cursor + offset,
                            "activity": activity,
                            **{column: float(value) for column, value in zip(SENSOR_COLUMNS, values)},
                        }
                    )
                time_cursor += block_length

    return pd.DataFrame(rows)


def _balanced_segment_lengths(total_length: int, segments: int, rng: np.random.Generator) -> list[int]:
    base = total_length // segments
    lengths = [base] * segments
    remainder = total_length - base * segments
    for index in range(remainder):
        lengths[index] += 1
    jitter = rng.integers(-6, 7, size=segments)
    adjusted = np.maximum(np.array(lengths) + jitter, 48)
    diff = int(total_length - adjusted.sum())
    adjusted[-1] += diff
    if adjusted[-1] < 48:
        adjusted[-1] = 48
        adjusted[0] -= 48 - adjusted[-1]
    return adjusted.tolist()


def segment_windows(
    frame: pd.DataFrame,
    config: WindowConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Turn continuous sensor streams into overlapped labelled windows."""

    feature_rows: list[np.ndarray] = []
    raw_windows: list[np.ndarray] = []
    labels: list[str] = []
    metadata: list[dict[str, Any]] = []

    grouped = frame.sort_values(["subject_id", "workflow_id", "timestep"]).groupby(
        ["subject_id", "workflow_id"], sort=False
    )
    for (_, workflow_id), group in grouped:
        signals = group[SENSOR_COLUMNS].to_numpy(dtype=float)
        activity_labels = group["activity"].to_numpy(dtype=object)
        subject_id = int(group["subject_id"].iloc[0])

        for start in range(0, len(group) - config.window_size + 1, config.step_size):
            stop = start + config.window_size
            window = signals[start:stop]
            window_labels = activity_labels[start:stop]
            majority_label = Counter(window_labels).most_common(1)[0][0]

            raw_windows.append(window)
            feature_rows.append(extract_window_features(window))
            labels.append(str(majority_label))
            metadata.append(
                {
                    "subject_id": subject_id,
                    "workflow_id": workflow_id,
                    "window_start": int(start),
                    "window_stop": int(stop),
                }
            )

    return (
        np.vstack(feature_rows),
        np.stack(raw_windows),
        np.array(labels, dtype=object),
        pd.DataFrame(metadata),
    )


def extract_window_features(window: np.ndarray) -> np.ndarray:
    features: list[float] = []
    for channel in range(window.shape[1]):
        signal = window[:, channel]
        diff_signal = np.diff(signal, prepend=signal[0])
        fft_magnitude = np.abs(np.fft.rfft(signal))
        top_fft = fft_magnitude[1:5]
        if len(top_fft) < 4:
            top_fft = np.pad(top_fft, (0, 4 - len(top_fft)))

        features.extend(
            [
                float(signal.mean()),
                float(signal.std()),
                float(signal.min()),
                float(signal.max()),
                float(np.median(signal)),
                float(np.percentile(signal, 25)),
                float(np.percentile(signal, 75)),
                float(np.sqrt(np.mean(signal**2))),
                float(np.mean(np.abs(diff_signal))),
                float(np.std(diff_signal)),
                *top_fft.astype(float).tolist(),
            ]
        )
    return np.asarray(features, dtype=float)


def evaluate_classical_models(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    logo = LeaveOneGroupOut()
    estimators = {
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
            ]
        ),
    }

    metrics: dict[str, Any] = {}
    out_of_fold_predictions: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Any] = {}

    for model_name, estimator in estimators.items():
        predictions = np.empty(len(labels), dtype=object)
        fold_metrics: list[dict[str, float]] = []

        for fold_index, (train_idx, test_idx) in enumerate(logo.split(features, labels, groups), start=1):
            model = clone(estimator)
            model.fit(features[train_idx], labels[train_idx])
            fold_predictions = model.predict(features[test_idx])
            predictions[test_idx] = fold_predictions
            fold_metrics.append(
                {
                    "fold": fold_index,
                    "accuracy": float(accuracy_score(labels[test_idx], fold_predictions)),
                    "macro_f1": float(
                        f1_score(labels[test_idx], fold_predictions, average="macro", zero_division=0)
                    ),
                }
            )

        final_model = clone(estimator).fit(features, labels)
        fitted_models[model_name] = final_model
        out_of_fold_predictions[model_name] = predictions
        metrics[model_name] = {
            "mean_accuracy": float(np.mean([fold["accuracy"] for fold in fold_metrics])),
            "mean_macro_f1": float(np.mean([fold["macro_f1"] for fold in fold_metrics])),
            "folds": fold_metrics,
            "classification_report": classification_report(labels, predictions, output_dict=True, zero_division=0),
        }

    return metrics, out_of_fold_predictions, fitted_models


class WorkflowHMM:
    """A lightweight discrete HMM for smoothing workflow labels."""

    def __init__(self) -> None:
        self.states_: list[str] = []
        self.initial_: np.ndarray | None = None
        self.transition_: np.ndarray | None = None
        self.emission_: np.ndarray | None = None
        self.state_index_: dict[str, int] = {}
        self.observation_index_: dict[str, int] = {}

    def fit(
        self,
        true_labels: np.ndarray,
        observed_labels: np.ndarray,
        sequence_ids: np.ndarray,
    ) -> "WorkflowHMM":
        self.states_ = sorted({str(label) for label in true_labels})
        observations = sorted({str(label) for label in observed_labels})
        self.state_index_ = {label: index for index, label in enumerate(self.states_)}
        self.observation_index_ = {label: index for index, label in enumerate(observations)}

        state_count = len(self.states_)
        obs_count = len(observations)
        initial_counts = np.ones(state_count, dtype=float)
        transition_counts = np.ones((state_count, state_count), dtype=float)
        emission_counts = np.ones((state_count, obs_count), dtype=float)

        frame = pd.DataFrame(
            {
                "sequence_id": sequence_ids,
                "true": true_labels,
                "observed": observed_labels,
            }
        )
        for _, group in frame.groupby("sequence_id", sort=False):
            true_seq = [self.state_index_[str(label)] for label in group["true"]]
            obs_seq = [self.observation_index_[str(label)] for label in group["observed"]]
            initial_counts[true_seq[0]] += 1.0

            for state_idx, obs_idx in zip(true_seq, obs_seq):
                emission_counts[state_idx, obs_idx] += 1.0
            for current_state, next_state in zip(true_seq[:-1], true_seq[1:]):
                transition_counts[current_state, next_state] += 1.0

        self.initial_ = initial_counts / initial_counts.sum()
        self.transition_ = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        self.emission_ = emission_counts / emission_counts.sum(axis=1, keepdims=True)
        return self

    def decode(self, observed_labels: np.ndarray | list[str]) -> np.ndarray:
        if self.initial_ is None or self.transition_ is None or self.emission_ is None:
            raise RuntimeError("WorkflowHMM must be fit before decoding.")

        encoded_obs = []
        for label in observed_labels:
            encoded_obs.append(self.observation_index_.get(str(label), 0))

        log_initial = np.log(self.initial_)
        log_transition = np.log(self.transition_)
        log_emission = np.log(self.emission_)

        time_steps = len(encoded_obs)
        state_count = len(self.states_)
        dp = np.full((time_steps, state_count), -np.inf)
        backpointer = np.zeros((time_steps, state_count), dtype=int)

        dp[0] = log_initial + log_emission[:, encoded_obs[0]]
        for step in range(1, time_steps):
            for state in range(state_count):
                transitions = dp[step - 1] + log_transition[:, state]
                best_prev = int(np.argmax(transitions))
                dp[step, state] = transitions[best_prev] + log_emission[state, encoded_obs[step]]
                backpointer[step, state] = best_prev

        best_last = int(np.argmax(dp[-1]))
        path = [best_last]
        for step in range(time_steps - 1, 0, -1):
            best_last = int(backpointer[step, best_last])
            path.append(best_last)
        path.reverse()
        return np.array([self.states_[state] for state in path], dtype=object)


def evaluate_workflow_hmm(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sequence_ids: np.ndarray,
    base_estimator: Any,
) -> dict[str, Any]:
    logo = LeaveOneGroupOut()
    observed_predictions = np.empty(len(labels), dtype=object)
    decoded_predictions = np.empty(len(labels), dtype=object)

    for train_idx, test_idx in logo.split(features, labels, groups):
        estimator = clone(base_estimator)
        estimator.fit(features[train_idx], labels[train_idx])
        train_observed = estimator.predict(features[train_idx])
        test_observed = estimator.predict(features[test_idx])

        hmm = WorkflowHMM().fit(labels[train_idx], train_observed, sequence_ids[train_idx])
        observed_predictions[test_idx] = test_observed

        fold_sequence_ids = sequence_ids[test_idx]
        for sequence_id in pd.unique(fold_sequence_ids):
            local_mask = fold_sequence_ids == sequence_id
            positions = test_idx[local_mask]
            decoded_predictions[positions] = hmm.decode(test_observed[local_mask])

    return {
        "base_accuracy": float(accuracy_score(labels, observed_predictions)),
        "decoded_accuracy": float(accuracy_score(labels, decoded_predictions)),
        "decoded_macro_f1": float(f1_score(labels, decoded_predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(labels, decoded_predictions, output_dict=True, zero_division=0),
        "decoded_predictions": decoded_predictions,
    }


class ActivityLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.encoder(inputs)
        return self.head(outputs[:, -1, :])


def evaluate_lstm(
    windows: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    epochs: int = 4,
    hidden_size: int = 48,
    batch_size: int = 32,
) -> dict[str, Any]:
    if torch is None:
        return {"available": False, "reason": "PyTorch is not installed."}

    torch.manual_seed(42)
    torch.set_num_threads(1)
    classes = sorted({str(label) for label in labels})
    label_to_index = {label: index for index, label in enumerate(classes)}
    encoded = np.array([label_to_index[str(label)] for label in labels], dtype=np.int64)
    logo = LeaveOneGroupOut()
    predictions = np.zeros(len(labels), dtype=np.int64)

    for train_idx, test_idx in logo.split(windows, encoded, groups):
        train_windows = windows[train_idx]
        test_windows = windows[test_idx]

        channel_mean = train_windows.mean(axis=(0, 1), keepdims=True)
        channel_std = train_windows.std(axis=(0, 1), keepdims=True) + 1e-6
        train_windows = (train_windows - channel_mean) / channel_std
        test_windows = (test_windows - channel_mean) / channel_std

        model = ActivityLSTM(input_size=windows.shape[2], hidden_size=hidden_size, num_classes=len(classes))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        train_dataset = TensorDataset(
            torch.tensor(train_windows, dtype=torch.float32),
            torch.tensor(encoded[train_idx], dtype=torch.long),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for _ in range(max(1, epochs)):
            for batch_windows, batch_labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_windows), batch_labels)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(test_windows, dtype=torch.float32))
        predictions[test_idx] = logits.argmax(dim=1).cpu().numpy()

    decoded_predictions = np.array([classes[index] for index in predictions], dtype=object)
    return {
        "available": True,
        "accuracy": float(accuracy_score(labels, decoded_predictions)),
        "macro_f1": float(f1_score(labels, decoded_predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(labels, decoded_predictions, output_dict=True, zero_division=0),
    }


def run_demo(
    output_dir: Path,
    n_subjects: int = 8,
    sequences_per_subject: int = 4,
    sequence_length: int = 320,
    window_size: int = 64,
    step_size: int = 32,
    lstm_epochs: int = 4,
    include_lstm: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_har_dataset(
        n_subjects=n_subjects,
        sequences_per_subject=sequences_per_subject,
        sequence_length=sequence_length,
    )
    config = WindowConfig(window_size=window_size, step_size=step_size)
    features, raw_windows, labels, metadata = segment_windows(dataset, config)
    groups = metadata["subject_id"].to_numpy()
    sequence_ids = metadata["workflow_id"].to_numpy(dtype=object)

    model_metrics, predictions, fitted_models = evaluate_classical_models(features, labels, groups)
    best_model_name = max(
        model_metrics,
        key=lambda name: model_metrics[name]["mean_accuracy"],
    )
    hmm_metrics = evaluate_workflow_hmm(
        features=features,
        labels=labels,
        groups=groups,
        sequence_ids=sequence_ids,
        base_estimator=fitted_models[best_model_name],
    )

    lstm_metrics = {"available": False, "reason": "LSTM evaluation disabled."}
    if include_lstm:
        lstm_metrics = evaluate_lstm(raw_windows, labels, groups, epochs=lstm_epochs)

    prediction_frame = metadata.copy()
    prediction_frame["true_activity"] = labels
    for model_name, model_predictions in predictions.items():
        prediction_frame[f"{model_name}_prediction"] = model_predictions
    prediction_frame["hmm_decoded_prediction"] = hmm_metrics["decoded_predictions"]
    prediction_frame.to_csv(output_dir / "window_predictions.csv", index=False)

    best_model = fitted_models[best_model_name]
    joblib.dump(best_model, output_dir / f"{best_model_name}_model.joblib")
    dataset.head(500).to_csv(output_dir / "synthetic_sample.csv", index=False)

    summary = {
        "config": {
            "n_subjects": n_subjects,
            "sequences_per_subject": sequences_per_subject,
            "sequence_length": sequence_length,
            **asdict(config),
        },
        "classical_models": model_metrics,
        "best_model": {
            "name": best_model_name,
            "accuracy": model_metrics[best_model_name]["mean_accuracy"],
            "macro_f1": model_metrics[best_model_name]["mean_macro_f1"],
        },
        "workflow_hmm": {
            key: value
            for key, value in hmm_metrics.items()
            if key != "decoded_predictions"
        },
        "lstm": lstm_metrics,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, indent=2)

    return summary


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
