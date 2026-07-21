import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate
import train_xgboost


PROCESSED_FILES_AVAILABLE = (
    train_xgboost.TRAIN_DATA_PATH.exists()
    and train_xgboost.PREPROCESSOR_PATH.exists()
    and train_xgboost.FEATURE_METADATA_PATH.exists()
    and evaluate.TEST_DATA_PATH.exists()
)


def make_training_frame(rows: int = 20) -> tuple[pd.DataFrame, pd.Series]:
    X_train = pd.DataFrame(
        {
            "numeric__Air temperature [K]": [300 + (idx % 3) for idx in range(rows)],
            "numeric__Process temperature [K]": [310 + (idx % 2) for idx in range(rows)],
            "numeric__Rotational speed [rpm]": [1400 + idx for idx in range(rows)],
            "numeric__Torque [Nm]": [40 + (idx % 5) for idx in range(rows)],
            "numeric__Tool wear [min]": [100 + idx for idx in range(rows)],
            "categorical__Type_H": [1 if idx % 3 == 0 else 0 for idx in range(rows)],
            "categorical__Type_L": [1 if idx % 3 == 1 else 0 for idx in range(rows)],
            "categorical__Type_M": [1 if idx % 3 == 2 else 0 for idx in range(rows)],
        }
    )
    y_train = pd.Series([0, 1] * (rows // 2), name=train_xgboost.TARGET_COLUMN)
    return X_train, y_train


def test_xgboost_training_bundle_can_be_saved(tmp_path, monkeypatch) -> None:
    X_train, y_train = make_training_frame()
    model = train_xgboost.train_model(X_train, y_train)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = processed_dir / "preprocessor.pkl"
    feature_metadata_path = processed_dir / "feature_metadata.json"

    with preprocessor_path.open("wb") as file:
        pickle.dump({"kind": "dummy-preprocessor"}, file)

    feature_metadata_path.write_text(
        json.dumps(
            {
                "target_column": train_xgboost.TARGET_COLUMN,
                "raw_feature_columns": ["Type"],
                "model_feature_columns": X_train.columns.tolist(),
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "models"
    output_path = output_dir / "xgboost.pkl"

    monkeypatch.setattr(train_xgboost, "PREPROCESSOR_PATH", preprocessor_path)
    monkeypatch.setattr(train_xgboost, "FEATURE_METADATA_PATH", feature_metadata_path)
    monkeypatch.setattr(train_xgboost, "MODEL_DIR", output_dir)
    monkeypatch.setattr(train_xgboost, "MODEL_PATH", output_path)

    train_xgboost.save_model_bundle(model)

    assert output_path.exists()

    with output_path.open("rb") as file:
        bundle = pickle.load(file)

    assert {"model", "preprocessor", "metadata", "input_format"} == set(bundle.keys())
    assert bundle["metadata"]["target_column"] == train_xgboost.TARGET_COLUMN
    assert bundle["input_format"] == "numpy"


def test_xgboost_evaluation_writes_isolated_artifacts_with_valid_metrics(
    tmp_path, monkeypatch
) -> None:
    X_train, y_train = make_training_frame(rows=24)
    model = train_xgboost.train_model(X_train, y_train)

    model_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"
    figures_dir = tmp_path / "figures"
    processed_dir = tmp_path / "processed"

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    xgb_model_path = model_dir / "xgboost.pkl"
    rf_metrics_path = metrics_dir / "random_forest_metrics.json"
    rf_figure_path = figures_dir / "random_forest_confusion_matrix.png"
    comparison_path = metrics_dir / "model_comparison.csv"
    test_data_path = processed_dir / "test_processed.csv"

    with xgb_model_path.open("wb") as file:
        pickle.dump({"model": model, "input_format": "numpy"}, file)

    rf_metrics_path.write_text("random-forest-metrics", encoding="utf-8")
    rf_figure_path.write_text("random-forest-figure", encoding="utf-8")
    comparison_path.write_text("comparison", encoding="utf-8")

    X_test = X_train.head(8).copy()
    y_test = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name=evaluate.TARGET_COLUMN)
    test_frame = X_test.copy()
    test_frame[evaluate.TARGET_COLUMN] = y_test
    test_frame.to_csv(test_data_path, index=False)

    xgb_metrics_path = metrics_dir / "xgboost_metrics.json"
    xgb_figure_path = figures_dir / "xgboost_confusion_matrix.png"

    monkeypatch.setattr(evaluate, "TEST_DATA_PATH", test_data_path)
    monkeypatch.setattr(evaluate, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(evaluate, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(evaluate, "COMPARISON_PATH", comparison_path)
    monkeypatch.setitem(evaluate.MODEL_ARTIFACTS["xgboost"], "model_path", xgb_model_path)
    monkeypatch.setitem(evaluate.MODEL_ARTIFACTS["xgboost"], "metrics_path", xgb_metrics_path)
    monkeypatch.setitem(evaluate.MODEL_ARTIFACTS["xgboost"], "figure_path", xgb_figure_path)
    monkeypatch.setattr(evaluate, "parse_args", lambda: type("Args", (), {"model": "xgboost"})())

    evaluate.main()

    assert xgb_metrics_path.exists()
    assert xgb_figure_path.exists()
    assert rf_metrics_path.read_text(encoding="utf-8") == "random-forest-metrics"
    assert rf_figure_path.read_text(encoding="utf-8") == "random-forest-figure"
    assert comparison_path.read_text(encoding="utf-8") == "comparison"

    metrics = json.loads(xgb_metrics_path.read_text(encoding="utf-8"))
    assert set(metrics.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "confusion_matrix",
    }
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= metrics[key] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2


@pytest.mark.skipif(
    not PROCESSED_FILES_AVAILABLE,
    reason="Processed AI4I artifacts are not available locally.",
)
def test_xgboost_official_runner_creates_sprint3_outputs(tmp_path, monkeypatch) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import run_ai4i_xgboost

    experiment_dir = tmp_path / "experiments" / "ai4i_xgboost"
    thesis_summary_path = tmp_path / "reports" / "metrics" / "xgboost_results_for_thesis.md"
    comparison_path = tmp_path / "reports" / "metrics" / "model_comparison.csv"
    model_dir = tmp_path / "models"
    xgb_model_path = model_dir / "xgboost.pkl"
    xgb_metrics_path = tmp_path / "reports" / "metrics" / "xgboost_metrics.json"
    xgb_figure_path = tmp_path / "reports" / "figures" / "xgboost_confusion_matrix.png"

    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text("legacy-comparison", encoding="utf-8")

    monkeypatch.setattr(run_ai4i_xgboost, "EXPERIMENT_DIR", experiment_dir)
    monkeypatch.setattr(run_ai4i_xgboost, "THESIS_SUMMARY_PATH", thesis_summary_path)
    monkeypatch.setattr(run_ai4i_xgboost.evaluate, "COMPARISON_PATH", comparison_path)
    monkeypatch.setattr(run_ai4i_xgboost.evaluate, "FIGURES_DIR", xgb_figure_path.parent)
    monkeypatch.setattr(run_ai4i_xgboost.evaluate, "METRICS_DIR", xgb_metrics_path.parent)
    monkeypatch.setattr(run_ai4i_xgboost.train_xgboost, "MODEL_DIR", model_dir)
    monkeypatch.setattr(run_ai4i_xgboost.train_xgboost, "MODEL_PATH", xgb_model_path)
    monkeypatch.setitem(
        run_ai4i_xgboost.evaluate.MODEL_ARTIFACTS["xgboost"], "model_path", xgb_model_path
    )
    monkeypatch.setitem(
        run_ai4i_xgboost.evaluate.MODEL_ARTIFACTS["xgboost"], "metrics_path", xgb_metrics_path
    )
    monkeypatch.setitem(
        run_ai4i_xgboost.evaluate.MODEL_ARTIFACTS["xgboost"], "figure_path", xgb_figure_path
    )

    result = run_ai4i_xgboost.run_experiment()

    payload_path = Path(result["payload_path"])
    summary_path = Path(result["summary_path"])
    top_features_path = Path(result["top_features_path"])

    assert payload_path.exists()
    assert summary_path.exists()
    assert top_features_path.exists()
    assert thesis_summary_path.exists()
    assert comparison_path.read_text(encoding="utf-8") == "legacy-comparison"
    assert "formal model comparison" in thesis_summary_path.read_text(encoding="utf-8")
