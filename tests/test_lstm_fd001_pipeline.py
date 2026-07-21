import json
import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train_lstm_fd001


def make_sequence_payload() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    engine_ids = np.repeat(np.arange(1, 6), 6)
    sequence_length = 5
    num_features = 3

    X = rng.normal(size=(len(engine_ids), sequence_length, num_features)).astype(np.float32)
    y = np.linspace(30, 1, num=len(engine_ids), dtype=np.float32)
    end_cycles = np.tile(np.arange(sequence_length, sequence_length + 6), 5).astype(np.int64)
    return X, y, engine_ids.astype(np.int64), end_cycles


def write_sprint5_like_artifacts(base_dir: Path) -> tuple[Path, Path, Path]:
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    X, y, engine_ids, end_cycles = make_sequence_payload()
    sequence_path = data_dir / "fd001_train_sequences.npz"
    np.savez_compressed(sequence_path, X=X, y=y, engine_id=engine_ids, end_cycle=end_cycles)

    feature_metadata_path = data_dir / "fd001_feature_metadata.json"
    feature_metadata_path.write_text(
        json.dumps(
            {
                "dataset": "NASA CMAPSS",
                "entity_id_column": "unit_id",
                "time_order_column": "cycle",
                "target_column": "RUL",
                "feature_columns": [f"feature_{index}" for index in range(3)],
                "operating_setting_columns": ["feature_0"],
                "sensor_columns": ["feature_1", "feature_2"],
                "normalization_applied": False,
            }
        ),
        encoding="utf-8",
    )

    sequence_metadata_path = data_dir / "fd001_train_sequence_metadata.json"
    sequence_metadata_path.write_text(
        json.dumps(
            {
                "dataset": "NASA CMAPSS",
                "subset": "FD001",
                "split": "train",
                "sequence_length": 5,
                "target_column": "RUL",
                "target_position": "last",
                "feature_columns": [f"feature_{index}" for index in range(3)],
                "num_sequences": int(X.shape[0]),
                "num_features": 3,
                "engine_count_in_sequences": 5,
                "feature_dtype": "float32",
                "target_dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    return sequence_path, feature_metadata_path, sequence_metadata_path


def test_lstm_runner_generates_loadable_model_and_artifacts(tmp_path, monkeypatch) -> None:
    sequence_path, feature_metadata_path, sequence_metadata_path = write_sprint5_like_artifacts(tmp_path)

    models_dir = tmp_path / "models"
    metrics_dir = tmp_path / "reports" / "metrics"
    figures_dir = tmp_path / "reports" / "figures"
    experiment_dir = tmp_path / "experiments" / "nasa_fd001_lstm"

    monkeypatch.setattr(train_lstm_fd001, "SEQUENCE_PATH", sequence_path)
    monkeypatch.setattr(train_lstm_fd001, "FEATURE_METADATA_PATH", feature_metadata_path)
    monkeypatch.setattr(train_lstm_fd001, "SEQUENCE_METADATA_PATH", sequence_metadata_path)
    monkeypatch.setattr(train_lstm_fd001, "MODELS_DIR", models_dir)
    monkeypatch.setattr(train_lstm_fd001, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(train_lstm_fd001, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(train_lstm_fd001, "EXPERIMENT_DIR", experiment_dir)
    monkeypatch.setattr(train_lstm_fd001, "MODEL_PATH", models_dir / "lstm_fd001.keras")
    monkeypatch.setattr(train_lstm_fd001, "CHECKPOINT_PATH", models_dir / "lstm_fd001_checkpoint.keras")
    monkeypatch.setattr(train_lstm_fd001, "METRICS_PATH", metrics_dir / "lstm_fd001_metrics.json")
    monkeypatch.setattr(
        train_lstm_fd001,
        "THESIS_SUMMARY_PATH",
        metrics_dir / "lstm_fd001_results_for_thesis.md",
    )
    monkeypatch.setattr(train_lstm_fd001, "TRAINING_FIGURE_PATH", figures_dir / "lstm_training_loss.png")
    monkeypatch.setattr(train_lstm_fd001, "TRAINING_HISTORY_PATH", experiment_dir / "training_history.json")
    monkeypatch.setattr(train_lstm_fd001, "TRAINING_CONFIG_PATH", experiment_dir / "training_config.json")

    config = train_lstm_fd001.TrainingConfig(
        seed=13,
        epochs=3,
        batch_size=4,
        learning_rate=1e-3,
        lstm_units=8,
        dense_units=4,
        patience=1,
        validation_engine_fraction=0.4,
        sequence_length=5,
        num_features=3,
    )

    result = train_lstm_fd001.run_training(config)

    model_path = Path(result["model_path"])
    checkpoint_path = Path(result["checkpoint_path"])
    metrics_path = Path(result["metrics_path"])
    figure_path = Path(result["training_figure_path"])
    summary_path = Path(result["thesis_summary_path"])
    history_path = Path(result["training_history_path"])
    config_path = Path(result["training_config_path"])

    assert model_path.exists()
    assert checkpoint_path.exists()
    assert metrics_path.exists()
    assert figure_path.exists()
    assert summary_path.exists()
    assert history_path.exists()
    assert config_path.exists()

    model = train_lstm_fd001.load_trained_model(model_path)
    X, _, _, _, _, _ = train_lstm_fd001.load_sprint5_artifacts()
    prediction = model.predict(X[:2], verbose=0)
    assert prediction.shape == (2, 1)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "checkpoint_path" in metrics
    assert metrics["train_sequence_count"] > 0
    assert metrics["validation_sequence_count"] > 0

    history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    assert "loss" in history_payload
    assert "val_loss" in history_payload
