import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate_lstm_fd001_test as test_eval
import train_lstm_fd001


def make_test_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for engine_id in (1, 2):
        for cycle in range(1, 6):
            row: dict[str, float | int] = {
                "unit_id": engine_id,
                "cycle": cycle,
            }
            for feature_index, column in enumerate(test_eval.FEATURE_COLUMNS, start=1):
                row[column] = float(engine_id * 10 + cycle + feature_index)
            rows.append(row)
    return pd.DataFrame(rows)[["unit_id", "cycle", *test_eval.FEATURE_COLUMNS]]


def write_cmapss_test_files(raw_dir: Path) -> None:
    frame = make_test_frame()
    frame.to_csv(raw_dir / "test_FD001.txt", sep=" ", header=False, index=False)
    pd.DataFrame({"RUL": [4, 7]}).to_csv(raw_dir / "RUL_FD001.txt", header=False, index=False)


def build_dummy_model(model_path: Path, sequence_length: int, num_features: int) -> None:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length, num_features)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.save(model_path)


def test_final_evaluation_generates_metrics_and_plots(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    metrics_dir = tmp_path / "reports" / "metrics"
    figures_dir = tmp_path / "reports" / "figures"
    dashboard_dir = tmp_path / "reports" / "dashboard"
    model_dir = tmp_path / "models"

    raw_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    write_cmapss_test_files(raw_dir)

    feature_metadata_path = data_dir / "fd001_feature_metadata.json"
    sequence_metadata_path = data_dir / "fd001_train_sequence_metadata.json"
    feature_metadata_path.write_text(
        json.dumps(
            {
                "dataset": "NASA CMAPSS",
                "entity_id_column": "unit_id",
                "time_order_column": "cycle",
                "target_column": "RUL",
                "feature_columns": test_eval.FEATURE_COLUMNS,
                "operating_setting_columns": test_eval.FEATURE_COLUMNS[:3],
                "sensor_columns": test_eval.FEATURE_COLUMNS[3:],
                "normalization_applied": False,
            }
        ),
        encoding="utf-8",
    )
    sequence_metadata_path.write_text(
        json.dumps(
            {
                "dataset": "NASA CMAPSS",
                "subset": "FD001",
                "split": "train",
                "sequence_length": 5,
                "target_column": "RUL",
                "target_position": "last",
                "feature_columns": test_eval.FEATURE_COLUMNS,
                "num_sequences": 2,
                "num_features": len(test_eval.FEATURE_COLUMNS),
                "engine_count_in_sequences": 2,
                "feature_dtype": "float64",
                "target_dtype": "float64",
            }
        ),
        encoding="utf-8",
    )

    model_path = model_dir / "lstm_fd001.keras"
    build_dummy_model(model_path, sequence_length=5, num_features=len(test_eval.FEATURE_COLUMNS))

    monkeypatch.setattr(test_eval, "FEATURE_METADATA_PATH", feature_metadata_path)
    monkeypatch.setattr(test_eval, "SEQUENCE_METADATA_PATH", sequence_metadata_path)
    monkeypatch.setattr(test_eval, "TEST_METRICS_PATH", metrics_dir / "lstm_fd001_test_metrics.json")
    monkeypatch.setattr(
        test_eval,
        "TEST_THESIS_SUMMARY_PATH",
        metrics_dir / "lstm_fd001_test_results_for_thesis.md",
    )
    monkeypatch.setattr(
        test_eval,
        "TEST_PREDICTIONS_FIGURE_PATH",
        figures_dir / "lstm_test_predictions.png",
    )
    monkeypatch.setattr(
        test_eval,
        "TEST_ERROR_DISTRIBUTION_FIGURE_PATH",
        figures_dir / "lstm_test_error_distribution.png",
    )
    monkeypatch.setattr(
        test_eval,
        "DASHBOARD_PREDICTIONS_PATH",
        dashboard_dir / "fd001_predictions.csv",
    )
    monkeypatch.setattr(test_eval, "MODEL_PATH", model_path)
    monkeypatch.setattr(train_lstm_fd001, "MODEL_PATH", model_path)
    monkeypatch.setattr(test_eval, "DEFAULT_SEQUENCE_LENGTH", 5)
    monkeypatch.setattr(
        test_eval,
        "load_official_test_split",
        lambda subset="FD001": (make_test_frame(), pd.DataFrame({"RUL": [4, 7]})),
    )

    result = test_eval.run_test_evaluation()

    metrics_path = Path(result["metrics_path"])
    thesis_path = Path(result["thesis_summary_path"])
    dashboard_predictions_path = Path(result["dashboard_predictions_path"])
    predictions_plot = Path(result["predictions_figure_path"])
    error_plot = Path(result["error_distribution_figure_path"])

    assert metrics_path.exists()
    assert thesis_path.exists()
    assert dashboard_predictions_path.exists()
    assert predictions_plot.exists()
    assert error_plot.exists()

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["prediction_count"] == metrics_payload["engine_count"] == 2
    assert metrics_payload["dashboard_predictions_path"] == str(dashboard_predictions_path)
    assert "mae" in metrics_payload
    assert "rmse" in metrics_payload

    dashboard_predictions = pd.read_csv(dashboard_predictions_path)
    assert list(dashboard_predictions.columns) == [
        "engine_id",
        "predicted_rul",
        "actual_rul",
        "absolute_error",
        "risk_level",
    ]
    assert dashboard_predictions["engine_id"].tolist() == [1, 2]
    assert (dashboard_predictions["absolute_error"] >= 0).all()
    assert set(dashboard_predictions["risk_level"].unique()).issubset(
        {"Low", "Moderate", "High", "Critical"}
    )

    model = train_lstm_fd001.load_trained_model(model_path)
    X_test, y_true, engine_ids, _ = test_eval.build_last_window_test_sequences(
        test_frame=make_test_frame(),
        sequence_length=5,
        feature_columns=test_eval.FEATURE_COLUMNS,
        rul_reference=pd.DataFrame({"RUL": [4, 7]}),
    )
    predictions = model.predict(X_test, verbose=0).reshape(-1)
    assert predictions.shape == (2,)
    assert y_true.shape == (2,)
    assert engine_ids.tolist() == [1, 2]
