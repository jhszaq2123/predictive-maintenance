from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from nasa_cmapss_preprocessing import (
    DEFAULT_SEQUENCE_LENGTH,
    ENGINE_ID_COLUMN,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_cmapss_split,
    preprocess_cmapss_frame,
)
from train_lstm_fd001 import MODEL_PATH, load_trained_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "nasa"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DASHBOARD_REPORTS_DIR = PROJECT_ROOT / "reports" / "dashboard"

FEATURE_METADATA_PATH = DATA_DIR / "fd001_feature_metadata.json"
SEQUENCE_METADATA_PATH = DATA_DIR / "fd001_train_sequence_metadata.json"

TEST_METRICS_PATH = METRICS_DIR / "lstm_fd001_test_metrics.json"
TEST_THESIS_SUMMARY_PATH = METRICS_DIR / "lstm_fd001_test_results_for_thesis.md"
TEST_PREDICTIONS_FIGURE_PATH = FIGURES_DIR / "lstm_test_predictions.png"
TEST_ERROR_DISTRIBUTION_FIGURE_PATH = FIGURES_DIR / "lstm_test_error_distribution.png"
DASHBOARD_PREDICTIONS_PATH = DASHBOARD_REPORTS_DIR / "fd001_predictions.csv"


def load_sequence_context() -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (FEATURE_METADATA_PATH, SEQUENCE_METADATA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required Sprint 5 metadata artifact not found: {path}")

    feature_metadata = json.loads(FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    sequence_metadata = json.loads(SEQUENCE_METADATA_PATH.read_text(encoding="utf-8"))
    return feature_metadata, sequence_metadata


def load_official_test_split(subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame]:
    test_frame_raw = load_cmapss_split(subset=subset, split="test")
    test_frame, _ = preprocess_cmapss_frame(test_frame_raw)
    rul_reference = load_cmapss_split(subset=subset, split="rul")
    return test_frame, rul_reference


def build_last_window_test_sequences(
    test_frame: pd.DataFrame,
    sequence_length: int,
    feature_columns: list[str] | tuple[str, ...],
    rul_reference: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped = list(test_frame.groupby(ENGINE_ID_COLUMN, sort=True))
    if len(grouped) != len(rul_reference):
        raise ValueError(
            "The number of FD001 test engines does not match the number of official RUL labels."
        )

    sequences: list[np.ndarray] = []
    engine_ids: list[int] = []
    end_cycles: list[int] = []

    for engine_id, engine_frame in grouped:
        if len(engine_frame) < sequence_length:
            raise ValueError(
                f"Engine {engine_id} has only {len(engine_frame)} cycles, "
                f"which is shorter than the required sequence length {sequence_length}."
            )
        ordered = engine_frame.sort_values("cycle", kind="stable")
        last_window = ordered.tail(sequence_length)
        sequences.append(last_window.loc[:, list(feature_columns)].to_numpy(dtype=np.float32, copy=True))
        engine_ids.append(int(engine_id))
        end_cycles.append(int(last_window["cycle"].iloc[-1]))

    X_test = np.stack(sequences).astype(np.float32)
    y_true = rul_reference[TARGET_COLUMN].to_numpy(dtype=np.float32, copy=True)
    return (
        X_test,
        y_true,
        np.asarray(engine_ids, dtype=np.int64),
        np.asarray(end_cycles, dtype=np.int64),
    )


def compute_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    return {"mae": mae, "rmse": rmse}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_predictions_plot(
    engine_ids: np.ndarray,
    y_true: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(engine_ids, y_true, marker="o", linewidth=1.5, label="Actual RUL")
    ax.plot(engine_ids, predictions, marker="s", linewidth=1.2, label="Predicted RUL")
    ax.set_xlabel("Engine ID")
    ax.set_ylabel("RUL")
    ax.set_title("LSTM FD001 official test set: predicted vs actual RUL")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_error_distribution_plot(errors: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=20, color="#4472C4", edgecolor="white", alpha=0.9)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction error (predicted RUL - actual RUL)")
    ax.set_ylabel("Number of engines")
    ax.set_title("LSTM FD001 official test set: prediction error distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_prediction_table(
    engine_ids: np.ndarray,
    predictions: np.ndarray,
    y_true: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "engine_id": engine_ids.astype(int),
            "predicted_rul": predictions.astype(float),
            "actual_rul": y_true.astype(float),
        }
    )
    frame["absolute_error"] = (frame["predicted_rul"] - frame["actual_rul"]).abs()
    frame["risk_level"] = frame["predicted_rul"].apply(risk_level_from_rul)
    return frame.sort_values("engine_id").reset_index(drop=True)


def risk_level_from_rul(rul: float) -> str:
    if rul >= 120:
        return "Low"
    if rul >= 60:
        return "Moderate"
    if rul >= 30:
        return "High"
    return "Critical"


def save_prediction_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def write_test_summary(metrics_payload: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# LSTM FD001 official test-set evaluation for thesis drafting",
        "",
        "This note documents the final evaluation of the trained LSTM baseline on the official NASA CMAPSS FD001 test set.",
        "",
        "## Evaluation protocol",
        "",
        "- The trained model from Sprint 6 was loaded from `models/lstm_fd001.keras`.",
        "- The official FD001 test trajectories were loaded from `test_FD001.txt`.",
        "- The official ground-truth RUL labels were loaded from `RUL_FD001.txt`.",
        "- For each engine, a single final inference window was created from the last available sequence of length 30, consistent with Sprint 5 sequence preparation.",
        "",
        "## Inference procedure",
        "",
        "- No retraining was performed.",
        "- No preprocessing logic was changed relative to Sprint 5.",
        "- The saved LSTM baseline produced one RUL prediction for each engine in the official test split.",
        "",
        "## Obtained metrics",
        "",
        f"- MAE: {metrics_payload['mae']:.4f}",
        f"- RMSE: {metrics_payload['rmse']:.4f}",
        "",
        "## Limitations",
        "",
        "- The evaluation concerns only the FD001 subset and only the baseline LSTM architecture accepted in Sprint 6.",
        "- No hyperparameter tuning or architectural comparison is included in this sprint.",
        "- The test protocol predicts one final RUL value per engine based on its last available observed window.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_test_evaluation() -> dict[str, Any]:
    feature_metadata, sequence_metadata = load_sequence_context()
    sequence_length = int(sequence_metadata["sequence_length"])
    feature_columns = sequence_metadata["feature_columns"]

    if sequence_length != DEFAULT_SEQUENCE_LENGTH:
        raise ValueError(
            f"Unexpected sequence length in Sprint 5 metadata: {sequence_length}. "
            f"Expected {DEFAULT_SEQUENCE_LENGTH}."
        )
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError("Sprint 5 feature columns do not match the expected NASA FD001 feature set.")
    if feature_metadata["feature_columns"] != feature_columns:
        raise ValueError("Feature metadata and sequence metadata disagree on feature column ordering.")

    test_frame, rul_reference = load_official_test_split("FD001")
    X_test, y_true, engine_ids, end_cycles = build_last_window_test_sequences(
        test_frame=test_frame,
        sequence_length=sequence_length,
        feature_columns=feature_columns,
        rul_reference=rul_reference,
    )

    model = load_trained_model(MODEL_PATH)
    predictions = model.predict(X_test, verbose=0).reshape(-1).astype(np.float32)
    errors = predictions - y_true
    metrics = compute_metrics(y_true, predictions)
    prediction_table = build_prediction_table(engine_ids, predictions, y_true)

    save_predictions_plot(engine_ids, y_true, predictions, TEST_PREDICTIONS_FIGURE_PATH)
    save_error_distribution_plot(errors, TEST_ERROR_DISTRIBUTION_FIGURE_PATH)
    save_prediction_table(prediction_table, DASHBOARD_PREDICTIONS_PATH)

    metrics_payload = {
        "dataset": "NASA CMAPSS",
        "subset": "FD001",
        "model_path": str(MODEL_PATH),
        "sequence_length": sequence_length,
        "feature_count": len(feature_columns),
        "engine_count": int(len(engine_ids)),
        "prediction_count": int(len(predictions)),
        "feature_columns": feature_columns,
        "engine_ids": engine_ids.tolist(),
        "end_cycles": end_cycles.tolist(),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "prediction_mean": float(predictions.mean()),
        "actual_mean": float(y_true.mean()),
        "error_mean": float(errors.mean()),
        "error_std": float(errors.std(ddof=0)),
        "dashboard_predictions_path": str(DASHBOARD_PREDICTIONS_PATH),
        "predictions_figure_path": str(TEST_PREDICTIONS_FIGURE_PATH),
        "error_distribution_figure_path": str(TEST_ERROR_DISTRIBUTION_FIGURE_PATH),
    }
    save_json(TEST_METRICS_PATH, metrics_payload)
    write_test_summary(metrics_payload, TEST_THESIS_SUMMARY_PATH)

    return {
        "metrics_path": str(TEST_METRICS_PATH),
        "thesis_summary_path": str(TEST_THESIS_SUMMARY_PATH),
        "dashboard_predictions_path": str(DASHBOARD_PREDICTIONS_PATH),
        "predictions_figure_path": str(TEST_PREDICTIONS_FIGURE_PATH),
        "error_distribution_figure_path": str(TEST_ERROR_DISTRIBUTION_FIGURE_PATH),
        "metrics": metrics_payload,
    }
