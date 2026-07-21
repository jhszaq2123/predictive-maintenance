from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "nasa"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "nasa_fd001_lstm"

SEQUENCE_PATH = DATA_DIR / "fd001_train_sequences.npz"
FEATURE_METADATA_PATH = DATA_DIR / "fd001_feature_metadata.json"
SEQUENCE_METADATA_PATH = DATA_DIR / "fd001_train_sequence_metadata.json"

MODEL_PATH = MODELS_DIR / "lstm_fd001.keras"
CHECKPOINT_PATH = MODELS_DIR / "lstm_fd001_checkpoint.keras"
METRICS_PATH = METRICS_DIR / "lstm_fd001_metrics.json"
THESIS_SUMMARY_PATH = METRICS_DIR / "lstm_fd001_results_for_thesis.md"
TRAINING_FIGURE_PATH = FIGURES_DIR / "lstm_training_loss.png"
TRAINING_HISTORY_PATH = EXPERIMENT_DIR / "training_history.json"
TRAINING_CONFIG_PATH = EXPERIMENT_DIR / "training_config.json"

DEFAULT_SEED = 42
DEFAULT_VAL_ENGINE_FRACTION = 0.2


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = DEFAULT_SEED
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 1e-3
    lstm_units: int = 32
    dense_units: int = 16
    patience: int = 4
    validation_engine_fraction: float = DEFAULT_VAL_ENGINE_FRACTION
    sequence_length: int = 30
    num_features: int = 24


def set_global_determinism(seed: int) -> None:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_sprint5_artifacts() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    for path in (SEQUENCE_PATH, FEATURE_METADATA_PATH, SEQUENCE_METADATA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required Sprint 5 artifact not found: {path}")

    sequence_payload = np.load(SEQUENCE_PATH)
    X = sequence_payload["X"].astype(np.float32)
    y = sequence_payload["y"].astype(np.float32)
    engine_ids = sequence_payload["engine_id"].astype(np.int64)
    end_cycles = sequence_payload["end_cycle"].astype(np.int64)

    feature_metadata = json.loads(FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    sequence_metadata = json.loads(SEQUENCE_METADATA_PATH.read_text(encoding="utf-8"))

    return X, y, engine_ids, end_cycles, feature_metadata, sequence_metadata


def split_sequences_by_engine(
    X: np.ndarray,
    y: np.ndarray,
    engine_ids: np.ndarray,
    end_cycles: np.ndarray,
    validation_engine_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    if not 0.0 < validation_engine_fraction < 1.0:
        raise ValueError("validation_engine_fraction must be between 0 and 1")

    unique_engines = np.unique(engine_ids)
    if len(unique_engines) < 2:
        raise ValueError("At least two engines are required for a train/validation split.")

    rng = np.random.default_rng(seed)
    shuffled_engines = unique_engines.copy()
    rng.shuffle(shuffled_engines)
    validation_engine_count = max(1, int(round(len(shuffled_engines) * validation_engine_fraction)))
    validation_engines = np.sort(shuffled_engines[:validation_engine_count])
    train_engines = np.sort(shuffled_engines[validation_engine_count:])
    if len(train_engines) == 0:
        raise ValueError("Validation split consumed all engines; adjust validation_engine_fraction.")

    train_mask = np.isin(engine_ids, train_engines)
    val_mask = np.isin(engine_ids, validation_engines)

    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(val_mask)

    shuffled_train_indices = train_indices[rng.permutation(len(train_indices))]

    return {
        "X_train": X[shuffled_train_indices],
        "y_train": y[shuffled_train_indices],
        "train_engine_ids": engine_ids[shuffled_train_indices],
        "train_end_cycles": end_cycles[shuffled_train_indices],
        "X_val": X[val_indices],
        "y_val": y[val_indices],
        "val_engine_ids": engine_ids[val_indices],
        "val_end_cycles": end_cycles[val_indices],
        "train_engines": train_engines,
        "validation_engines": validation_engines,
    }


def build_model(config: TrainingConfig, X_train: np.ndarray) -> tf.keras.Model:
    normalization = tf.keras.layers.Normalization(axis=-1, name="feature_normalization")
    normalization.adapt(X_train.reshape(-1, X_train.shape[-1]))

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(config.sequence_length, config.num_features)),
            normalization,
            tf.keras.layers.LSTM(config.lstm_units, name="lstm_encoder"),
            tf.keras.layers.Dense(config.dense_units, activation="relu", name="dense_projection"),
            tf.keras.layers.Dense(1, name="rul_output"),
        ],
        name="lstm_fd001_baseline",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def build_callbacks(config: TrainingConfig) -> list[tf.keras.callbacks.Callback]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CHECKPOINT_PATH),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": mae, "rmse": rmse}


def save_training_history(history: tf.keras.callbacks.History, output_path: Path) -> dict[str, list[float]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return serializable


def save_training_plot(history_payload: dict[str, list[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history_payload["loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history_payload["loss"], label="Training loss")
    if "val_loss" in history_payload:
        ax.plot(epochs, history_payload["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("LSTM FD001 training history")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def write_training_summary(metrics: dict[str, Any], config: TrainingConfig, output_path: Path) -> None:
    lines = [
        "# LSTM baseline results for thesis drafting",
        "",
        "This note documents the first reproducible LSTM baseline for Remaining Useful Life prediction on NASA CMAPSS FD001.",
        "",
        "## Model architecture",
        "",
        f"- Input sequences: length {config.sequence_length}, {config.num_features} features per timestep",
        f"- LSTM layer: {config.lstm_units} units",
        f"- Dense hidden layer: {config.dense_units} units with ReLU activation",
        "- Output layer: single scalar RUL prediction",
        "",
        "## Training configuration",
        "",
        f"- Random seed: {config.seed}",
        f"- Optimizer: Adam, learning rate {config.learning_rate}",
        f"- Batch size: {config.batch_size}",
        f"- Maximum epochs: {config.epochs}",
        f"- Validation split: engine-level holdout using {metrics['validation_engine_count']} out of {metrics['total_engine_count']} engines",
        f"- EarlyStopping patience: {config.patience}",
        "- ModelCheckpoint: best validation-loss model saved during training",
        "",
        "## Evaluation procedure",
        "",
        "- Training and validation data were split by engine identifier to avoid mixing overlapping windows from the same trajectory across both subsets.",
        "- The final reported metrics were computed on the held-out validation engines using the best checkpoint selected by validation loss.",
        "",
        "## Obtained metrics",
        "",
        f"- MAE: {metrics['mae']:.4f}",
        f"- RMSE: {metrics['rmse']:.4f}",
        "",
        "## Limitations",
        "",
        "- This is a baseline model without hyperparameter tuning, architectural search, or comparison to other sequential architectures.",
        "- The experiment uses only FD001 sequences prepared in Sprint 5 and therefore should not be generalized to other CMAPSS subsets.",
        "- The validation set is derived from the training split through engine-level holdout because Sprint 6 does not yet define a separate final test protocol.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def load_trained_model(path: Path = MODEL_PATH) -> tf.keras.Model:
    if not path.exists():
        raise FileNotFoundError(f"Trained LSTM model not found: {path}")
    return tf.keras.models.load_model(path)


def run_training(config: TrainingConfig | None = None) -> dict[str, Any]:
    training_config = config or TrainingConfig()
    set_global_determinism(training_config.seed)

    X, y, engine_ids, end_cycles, feature_metadata, sequence_metadata = load_sprint5_artifacts()
    if X.shape[1] != training_config.sequence_length or X.shape[2] != training_config.num_features:
        raise ValueError(
            "TrainingConfig does not match Sprint 5 sequence artifacts: "
            f"expected {(training_config.sequence_length, training_config.num_features)}, got {X.shape[1:]}"
        )

    split_payload = split_sequences_by_engine(
        X=X,
        y=y,
        engine_ids=engine_ids,
        end_cycles=end_cycles,
        validation_engine_fraction=training_config.validation_engine_fraction,
        seed=training_config.seed,
    )

    model = build_model(training_config, split_payload["X_train"])
    callbacks = build_callbacks(training_config)
    history = model.fit(
        split_payload["X_train"],
        split_payload["y_train"],
        validation_data=(split_payload["X_val"], split_payload["y_val"]),
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        callbacks=callbacks,
        shuffle=False,
        verbose=0,
    )

    best_model = tf.keras.models.load_model(CHECKPOINT_PATH)
    predictions = best_model.predict(split_payload["X_val"], verbose=0).reshape(-1)
    evaluation_metrics = evaluate_predictions(split_payload["y_val"], predictions)

    history_payload = save_training_history(history, TRAINING_HISTORY_PATH)
    save_training_plot(history_payload, TRAINING_FIGURE_PATH)

    best_model.save(MODEL_PATH)

    training_summary = {
        "dataset": "NASA CMAPSS",
        "subset": sequence_metadata["subset"],
        "split_used_from_sprint5": sequence_metadata["split"],
        "target_column": sequence_metadata["target_column"],
        "sequence_length": training_config.sequence_length,
        "num_features": training_config.num_features,
        "seed": training_config.seed,
        "epochs_requested": training_config.epochs,
        "epochs_completed": len(history_payload["loss"]),
        "stopped_early": len(history_payload["loss"]) < training_config.epochs,
        "batch_size": training_config.batch_size,
        "learning_rate": training_config.learning_rate,
        "patience": training_config.patience,
        "train_sequence_count": int(split_payload["X_train"].shape[0]),
        "validation_sequence_count": int(split_payload["X_val"].shape[0]),
        "validation_engine_count": int(len(split_payload["validation_engines"])),
        "total_engine_count": int(len(np.unique(engine_ids))),
        "validation_engines": split_payload["validation_engines"].tolist(),
        "feature_metadata": feature_metadata,
        "mae": evaluation_metrics["mae"],
        "rmse": evaluation_metrics["rmse"],
        "checkpoint_path": str(CHECKPOINT_PATH),
        "model_path": str(MODEL_PATH),
        "training_history_path": str(TRAINING_HISTORY_PATH),
        "training_figure_path": str(TRAINING_FIGURE_PATH),
    }

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_CONFIG_PATH.write_text(json.dumps(asdict(training_config), indent=2), encoding="utf-8")
    save_metrics(training_summary, METRICS_PATH)
    write_training_summary(training_summary, training_config, THESIS_SUMMARY_PATH)

    return {
        "model_path": str(MODEL_PATH),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "metrics_path": str(METRICS_PATH),
        "training_figure_path": str(TRAINING_FIGURE_PATH),
        "thesis_summary_path": str(THESIS_SUMMARY_PATH),
        "training_history_path": str(TRAINING_HISTORY_PATH),
        "training_config_path": str(TRAINING_CONFIG_PATH),
        "metrics": training_summary,
    }
