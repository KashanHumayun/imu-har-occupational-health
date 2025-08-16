from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

UCI_LABEL_MAP = {
    1: "walking",
    2: "walking_upstairs",
    3: "walking_downstairs",
    4: "sitting",
    5: "standing",
    6: "laying",
}

PAMAP2_LABEL_MAP = {
    2: "sitting",
    3: "standing",
    4: "walking",
}

PAMAP2_COLUMNS = ["timestamp", "activity_id", "heart_rate"]
for _sensor in ["hand", "chest", "ankle"]:
    PAMAP2_COLUMNS.extend(
        [
            f"{_sensor}_temp",
            f"{_sensor}_acc16_x",
            f"{_sensor}_acc16_y",
            f"{_sensor}_acc16_z",
            f"{_sensor}_acc6_x",
            f"{_sensor}_acc6_y",
            f"{_sensor}_acc6_z",
            f"{_sensor}_gyro_x",
            f"{_sensor}_gyro_y",
            f"{_sensor}_gyro_z",
            f"{_sensor}_mag_x",
            f"{_sensor}_mag_y",
            f"{_sensor}_mag_z",
            f"{_sensor}_ori_1",
            f"{_sensor}_ori_2",
            f"{_sensor}_ori_3",
            f"{_sensor}_ori_4",
        ]
    )

PAMAP2_SENSOR_COLUMNS = [
    "hand_acc16_x",
    "hand_acc16_y",
    "hand_acc16_z",
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",
    "chest_acc16_x",
    "chest_acc16_y",
    "chest_acc16_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",
]

UCI_INERTIAL_SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


@dataclass(slots=True)
class PamapWindowConfig:
    window_size: int = 200
    step_size: int = 100
    downsample: int = 3


def find_uci_har_root(project_root: Path) -> Path:
    return project_root / "data" / "raw" / "uci_har" / "UCI HAR Dataset" / "UCI HAR Dataset"


def find_pamap2_root(project_root: Path) -> Path:
    return project_root / "data" / "raw" / "pamap2" / "PAMAP2_Dataset" / "PAMAP2_Dataset" / "Protocol"


def load_uci_har_feature_split(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    X_train = pd.read_csv(root / "train" / "X_train.txt", sep=r"\s+", header=None)
    X_test = pd.read_csv(root / "test" / "X_test.txt", sep=r"\s+", header=None)
    y_train = pd.read_csv(root / "train" / "y_train.txt", header=None)[0].map(UCI_LABEL_MAP)
    y_test = pd.read_csv(root / "test" / "y_test.txt", header=None)[0].map(UCI_LABEL_MAP)
    subject_train = pd.read_csv(root / "train" / "subject_train.txt", header=None)[0].to_numpy()
    subject_test = pd.read_csv(root / "test" / "subject_test.txt", header=None)[0].to_numpy()
    return X_train, X_test, y_train, y_test, subject_train, subject_test


def load_uci_har_inertial_split(root: Path) -> tuple[np.ndarray, np.ndarray]:
    windows: dict[str, np.ndarray] = {}
    for split_name in ["train", "test"]:
        signal_arrays = []
        signal_root = root / split_name / "Inertial Signals"
        for signal_name in UCI_INERTIAL_SIGNALS:
            signal_arrays.append(
                pd.read_csv(signal_root / f"{signal_name}_{split_name}.txt", sep=r"\s+", header=None).to_numpy()
            )
        windows[split_name] = np.stack(signal_arrays, axis=2).astype(np.float32)
    return windows["train"], windows["test"]


def evaluate_uci_har_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    estimators = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearSVC(random_state=42, dual=False)),
            ]
        ),
    }

    metrics: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Any] = {}

    for model_name, estimator in estimators.items():
        model = clone(estimator)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        metrics[model_name] = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            "classification_report": classification_report(y_test, predicted, output_dict=True, zero_division=0),
        }
        predictions[model_name] = predicted
        fitted_models[model_name] = model
    return metrics, predictions, fitted_models


def extract_window_features(window: np.ndarray) -> np.ndarray:
    features: list[float] = []
    for channel in range(window.shape[1]):
        signal = window[:, channel]
        fft_mag = np.abs(np.fft.rfft(signal))
        top_fft = fft_mag[1:5]
        if len(top_fft) < 4:
            top_fft = np.pad(top_fft, (0, 4 - len(top_fft)))
        features.extend(
            [
                float(signal.mean()),
                float(signal.std()),
                float(signal.min()),
                float(signal.max()),
                float(np.sqrt(np.mean(signal**2))),
                float(np.mean(np.abs(np.diff(signal, prepend=signal[0])))),
                *top_fft.astype(float).tolist(),
            ]
        )
    return np.asarray(features, dtype=float)


def load_pamap2_windows(
    protocol_root: Path,
    config: PamapWindowConfig,
    subject_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    labels: list[str] = []
    subjects: list[str] = []
    sequence_ids: list[str] = []

    subject_paths = sorted(protocol_root.glob("subject10*.dat"))
    if subject_limit is not None:
        subject_paths = subject_paths[:subject_limit]

    use_columns = ["activity_id", *PAMAP2_SENSOR_COLUMNS]
    for subject_path in subject_paths:
        frame = pd.read_csv(
            subject_path,
            sep=r"\s+",
            header=None,
            names=PAMAP2_COLUMNS,
            usecols=use_columns,
            na_values="NaN",
        )
        frame = frame[frame["activity_id"].isin(PAMAP2_LABEL_MAP)].dropna().iloc[:: config.downsample].reset_index(drop=True)
        values = frame[PAMAP2_SENSOR_COLUMNS].to_numpy(dtype=float)
        activity_labels = frame["activity_id"].map(PAMAP2_LABEL_MAP).to_numpy(dtype=object)

        for start in range(0, len(frame) - config.window_size + 1, config.step_size):
            stop = start + config.window_size
            window = values[start:stop]
            window_labels = activity_labels[start:stop]
            majority_label = pd.Series(window_labels).mode().iat[0]
            feature_rows.append(extract_window_features(window))
            labels.append(str(majority_label))
            subjects.append(subject_path.stem)
            sequence_ids.append(f"{subject_path.stem}_{start}")

    return (
        np.vstack(feature_rows),
        np.array(labels, dtype=object),
        np.array(subjects, dtype=object),
        np.array(sequence_ids, dtype=object),
    )


def evaluate_logo_models(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    estimators = {
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearSVC(random_state=42, dual=False)),
            ]
        ),
    }
    logo = LeaveOneGroupOut()
    metrics: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Any] = {}

    for model_name, estimator in estimators.items():
        out_of_fold = np.empty(len(labels), dtype=object)
        fold_metrics: list[dict[str, float]] = []
        for fold_index, (train_idx, test_idx) in enumerate(logo.split(features, labels, groups), start=1):
            model = clone(estimator)
            model.fit(features[train_idx], labels[train_idx])
            predicted = model.predict(features[test_idx])
            out_of_fold[test_idx] = predicted
            fold_metrics.append(
                {
                    "fold": fold_index,
                    "accuracy": float(accuracy_score(labels[test_idx], predicted)),
                    "macro_f1": float(f1_score(labels[test_idx], predicted, average="macro", zero_division=0)),
                }
            )
        fitted_models[model_name] = clone(estimator).fit(features, labels)
        predictions[model_name] = out_of_fold
        metrics[model_name] = {
            "mean_accuracy": float(np.mean([item["accuracy"] for item in fold_metrics])),
            "mean_macro_f1": float(np.mean([item["macro_f1"] for item in fold_metrics])),
            "folds": fold_metrics,
            "classification_report": classification_report(labels, out_of_fold, output_dict=True, zero_division=0),
        }
    return metrics, predictions, fitted_models


class WorkflowHMM:
    def __init__(self) -> None:
        self.states_: list[str] = []
        self.state_index_: dict[str, int] = {}
        self.observation_index_: dict[str, int] = {}
        self.initial_: np.ndarray | None = None
        self.transition_: np.ndarray | None = None
        self.emission_: np.ndarray | None = None

    def fit(self, true_labels: np.ndarray, observed_labels: np.ndarray, sequence_ids: np.ndarray) -> "WorkflowHMM":
        self.states_ = sorted({str(item) for item in true_labels})
        observations = sorted({str(item) for item in observed_labels})
        self.state_index_ = {label: idx for idx, label in enumerate(self.states_)}
        self.observation_index_ = {label: idx for idx, label in enumerate(observations)}

        initial = np.ones(len(self.states_), dtype=float)
        transition = np.ones((len(self.states_), len(self.states_)), dtype=float)
        emission = np.ones((len(self.states_), len(observations)), dtype=float)

        frame = pd.DataFrame({"sequence_id": sequence_ids, "true": true_labels, "observed": observed_labels})
        for _, group in frame.groupby("sequence_id", sort=False):
            true_seq = [self.state_index_[str(value)] for value in group["true"]]
            observed_seq = [self.observation_index_[str(value)] for value in group["observed"]]
            initial[true_seq[0]] += 1.0
            for state, obs in zip(true_seq, observed_seq):
                emission[state, obs] += 1.0
            for current_state, next_state in zip(true_seq[:-1], true_seq[1:]):
                transition[current_state, next_state] += 1.0

        self.initial_ = initial / initial.sum()
        self.transition_ = transition / transition.sum(axis=1, keepdims=True)
        self.emission_ = emission / emission.sum(axis=1, keepdims=True)
        return self

    def decode(self, observations: np.ndarray) -> np.ndarray:
        if self.initial_ is None or self.transition_ is None or self.emission_ is None:
            raise RuntimeError("HMM must be fit before decoding.")

        encoded = [self.observation_index_.get(str(item), 0) for item in observations]
        log_initial = np.log(self.initial_)
        log_transition = np.log(self.transition_)
        log_emission = np.log(self.emission_)

        time_steps = len(encoded)
        state_count = len(self.states_)
        dp = np.full((time_steps, state_count), -np.inf)
        backpointer = np.zeros((time_steps, state_count), dtype=int)
        dp[0] = log_initial + log_emission[:, encoded[0]]

        for step in range(1, time_steps):
            for state in range(state_count):
                scores = dp[step - 1] + log_transition[:, state]
                best_previous = int(np.argmax(scores))
                dp[step, state] = scores[best_previous] + log_emission[state, encoded[step]]
                backpointer[step, state] = best_previous

        best_last = int(np.argmax(dp[-1]))
        path = [best_last]
        for step in range(time_steps - 1, 0, -1):
            best_last = int(backpointer[step, best_last])
            path.append(best_last)
        path.reverse()
        return np.array([self.states_[index] for index in path], dtype=object)


def evaluate_hmm(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sequence_ids: np.ndarray,
    estimator: Any,
) -> dict[str, Any]:
    logo = LeaveOneGroupOut()
    base_predictions = np.empty(len(labels), dtype=object)
    decoded_predictions = np.empty(len(labels), dtype=object)

    for train_idx, test_idx in logo.split(features, labels, groups):
        model = clone(estimator)
        model.fit(features[train_idx], labels[train_idx])
        train_pred = model.predict(features[train_idx])
        test_pred = model.predict(features[test_idx])
        hmm = WorkflowHMM().fit(labels[train_idx], train_pred, sequence_ids[train_idx])
        base_predictions[test_idx] = test_pred
        test_sequences = sequence_ids[test_idx]
        for sequence_id in pd.unique(test_sequences):
            local_mask = test_sequences == sequence_id
            positions = test_idx[local_mask]
            decoded_predictions[positions] = hmm.decode(test_pred[local_mask])

    return {
        "base_accuracy": float(accuracy_score(labels, base_predictions)),
        "decoded_accuracy": float(accuracy_score(labels, decoded_predictions)),
        "decoded_macro_f1": float(f1_score(labels, decoded_predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(labels, decoded_predictions, output_dict=True, zero_division=0),
    }


class ActivityLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.encoder(inputs)
        return self.head(outputs[:, -1, :])


def evaluate_uci_har_lstm(
    windows_train: np.ndarray,
    windows_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    epochs: int = 2,
) -> dict[str, Any]:
    if torch is None:
        return {"available": False, "reason": "PyTorch is not installed."}

    torch.manual_seed(42)
    torch.set_num_threads(1)

    classes = sorted(set(y_train).union(set(y_test)))
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    y_train_idx = np.array([label_to_index[label] for label in y_train], dtype=np.int64)
    y_test_idx = np.array([label_to_index[label] for label in y_test], dtype=np.int64)

    channel_mean = windows_train.mean(axis=(0, 1), keepdims=True)
    channel_std = windows_train.std(axis=(0, 1), keepdims=True) + 1e-6
    windows_train = (windows_train - channel_mean) / channel_std
    windows_test = (windows_test - channel_mean) / channel_std

    model = ActivityLSTM(input_size=windows_train.shape[2], hidden_size=48, num_classes=len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(
        torch.tensor(windows_train, dtype=torch.float32),
        torch.tensor(y_train_idx, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model.train()
    for _ in range(max(1, epochs)):
        for batch_windows, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_windows), batch_labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(windows_test, dtype=torch.float32))
    predicted = logits.argmax(dim=1).cpu().numpy()
    decoded = np.array([classes[index] for index in predicted], dtype=object)
    return {
        "available": True,
        "accuracy": float(accuracy_score(y_test, decoded)),
        "macro_f1": float(f1_score(y_test, decoded, average="macro", zero_division=0)),
        "classification_report": classification_report(y_test, decoded, output_dict=True, zero_division=0),
    }


def _plot_overview(uci_metrics: dict[str, Any], pamap2_metrics: dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    uci_models = ["random_forest", "svm"]
    uci_scores = [uci_metrics[name]["accuracy"] for name in uci_models]
    axes[0].bar(uci_models, uci_scores, color=["#457b9d", "#2a9d8f"])
    axes[0].set_title("UCI HAR Test Accuracy")
    axes[0].set_ylim(0, 1.0)

    pamap_models = ["random_forest", "svm", "hmm_decoded"]
    pamap_scores = [
        pamap2_metrics["classical_models"]["random_forest"]["mean_accuracy"],
        pamap2_metrics["classical_models"]["svm"]["mean_accuracy"],
        pamap2_metrics["workflow_hmm"]["decoded_accuracy"],
    ]
    axes[1].bar(pamap_models, pamap_scores, color=["#e76f51", "#f4a261", "#264653"])
    axes[1].set_title("PAMAP2 LOSO Accuracy")
    axes[1].set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_pipeline(
    project_root: Path,
    output_dir: Path,
    model_dir: Path,
    train_lstm: bool = False,
    pamap2_subject_limit: int | None = None,
) -> dict[str, Any]:
    uci_root = find_uci_har_root(project_root)
    pamap2_root = find_pamap2_root(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, subject_train, subject_test = load_uci_har_feature_split(uci_root)
    uci_metrics, uci_predictions, uci_models = evaluate_uci_har_models(X_train, X_test, y_train, y_test)

    lstm_metrics = {"available": False, "reason": "LSTM training skipped."}
    if train_lstm:
        inertial_train, inertial_test = load_uci_har_inertial_split(uci_root)
        lstm_metrics = evaluate_uci_har_lstm(inertial_train, inertial_test, y_train, y_test)

    uci_prediction_frame = pd.DataFrame(
        {
            "subject_id": subject_test,
            "true_activity": y_test,
            "random_forest_prediction": uci_predictions["random_forest"],
            "svm_prediction": uci_predictions["svm"],
        }
    )
    uci_prediction_frame.to_csv(output_dir / "uci_har_predictions.csv", index=False)

    pamap_config = PamapWindowConfig()
    pamap_features, pamap_labels, pamap_subjects, pamap_sequence_ids = load_pamap2_windows(
        pamap2_root,
        config=pamap_config,
        subject_limit=pamap2_subject_limit,
    )
    pamap_metrics, pamap_predictions, pamap_models = evaluate_logo_models(pamap_features, pamap_labels, pamap_subjects)
    best_name = max(pamap_metrics, key=lambda item: pamap_metrics[item]["mean_accuracy"])
    hmm_metrics = evaluate_hmm(
        pamap_features,
        pamap_labels,
        pamap_subjects,
        pamap_sequence_ids,
        estimator=pamap_models[best_name],
    )

    pamap_prediction_frame = pd.DataFrame(
        {
            "subject_id": pamap_subjects,
            "sequence_id": pamap_sequence_ids,
            "true_activity": pamap_labels,
            "random_forest_prediction": pamap_predictions["random_forest"],
            "svm_prediction": pamap_predictions["svm"],
        }
    )
    pamap_prediction_frame.to_csv(output_dir / "pamap2_predictions.csv", index=False)

    joblib.dump(uci_models[max(uci_metrics, key=lambda item: uci_metrics[item]["accuracy"])], model_dir / "uci_har_best_model.joblib")
    joblib.dump(pamap_models[best_name], model_dir / "pamap2_best_model.joblib")

    summary = {
        "uci_har": {
            "samples_train": int(len(X_train)),
            "samples_test": int(len(X_test)),
            "subjects_train": int(len(np.unique(subject_train))),
            "subjects_test": int(len(np.unique(subject_test))),
            "models": uci_metrics,
            "best_model": max(uci_metrics, key=lambda item: uci_metrics[item]["accuracy"]),
            "lstm": lstm_metrics,
        },
        "pamap2": {
            "windows": int(len(pamap_features)),
            "subjects": int(len(np.unique(pamap_subjects))),
            "window_config": {
                "window_size": pamap_config.window_size,
                "step_size": pamap_config.step_size,
                "downsample": pamap_config.downsample,
            },
            "classical_models": pamap_metrics,
            "best_model": best_name,
            "workflow_hmm": hmm_metrics,
        },
    }

    _plot_overview(uci_metrics, summary["pamap2"], output_dir / "performance_overview.png")
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
