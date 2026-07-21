import json
import pickle
import sys
from hashlib import sha256
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import evaluate
import run_model_comparison


def file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def make_processed_frame() -> pd.DataFrame:
    X = pd.DataFrame(
        {
            "numeric__Air temperature [K]": [0.1, -0.4, 0.5, -0.6, 0.8, -0.2, 1.1, -0.9],
            "numeric__Process temperature [K]": [0.2, -0.3, 0.6, -0.5, 0.7, -0.1, 1.2, -0.8],
            "numeric__Rotational speed [rpm]": [0.0, 0.4, 0.8, -0.3, 1.0, -0.6, 1.2, -0.9],
            "numeric__Torque [Nm]": [0.3, -0.2, 0.7, -0.4, 0.9, -0.5, 1.3, -0.7],
            "numeric__Tool wear [min]": [0.4, -0.1, 0.9, -0.2, 1.1, -0.4, 1.4, -0.6],
            "categorical__Type_H": [1, 0, 0, 1, 0, 0, 1, 0],
            "categorical__Type_L": [0, 1, 0, 0, 1, 0, 0, 1],
            "categorical__Type_M": [0, 0, 1, 0, 0, 1, 0, 0],
        }
    )
    y = pd.Series([0, 0, 1, 0, 1, 0, 1, 1], name=evaluate.TARGET_COLUMN)
    frame = X.copy()
    frame[evaluate.TARGET_COLUMN] = y
    return frame


def create_model_bundles(rf_model_path: Path, xgb_model_path: Path) -> None:
    frame = make_processed_frame()
    X = frame.drop(columns=[evaluate.TARGET_COLUMN])
    y = frame[evaluate.TARGET_COLUMN]

    rf_model = RandomForestClassifier(n_estimators=20, random_state=7)
    rf_model.fit(X, y)

    xgb_like_model = LogisticRegression(random_state=7, max_iter=200)
    xgb_like_model.fit(X.to_numpy(), y.to_numpy())

    rf_model_path.parent.mkdir(parents=True, exist_ok=True)
    with rf_model_path.open("wb") as file:
        pickle.dump({"model": rf_model}, file)

    xgb_model_path.parent.mkdir(parents=True, exist_ok=True)
    with xgb_model_path.open("wb") as file:
        pickle.dump({"model": xgb_like_model, "input_format": "numpy"}, file)


def test_model_comparison_generates_outputs_and_preserves_baseline_artifacts(
    tmp_path, monkeypatch
) -> None:
    reports_dir = tmp_path / "reports"
    metrics_dir = reports_dir / "metrics"
    figures_dir = reports_dir / "figures"
    models_dir = tmp_path / "models"
    processed_dir = tmp_path / "processed"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    rf_model_path = models_dir / "random_forest.pkl"
    xgb_model_path = models_dir / "xgboost.pkl"
    create_model_bundles(rf_model_path, xgb_model_path)

    rf_metrics_path = metrics_dir / "random_forest_metrics.json"
    xgb_metrics_path = metrics_dir / "xgboost_metrics.json"
    rf_figure_path = figures_dir / "random_forest_confusion_matrix.png"
    xgb_figure_path = figures_dir / "xgboost_confusion_matrix.png"

    rf_metrics_path.write_text("rf-metrics", encoding="utf-8")
    xgb_metrics_path.write_text("xgb-metrics", encoding="utf-8")
    rf_figure_path.write_bytes(b"rf-figure")
    xgb_figure_path.write_bytes(b"xgb-figure")

    rf_model_digest_before = file_digest(rf_model_path)
    xgb_model_digest_before = file_digest(xgb_model_path)
    rf_metrics_digest_before = file_digest(rf_metrics_path)
    xgb_metrics_digest_before = file_digest(xgb_metrics_path)
    rf_figure_digest_before = file_digest(rf_figure_path)
    xgb_figure_digest_before = file_digest(xgb_figure_path)

    test_frame = make_processed_frame()
    test_data_path = processed_dir / "test_processed.csv"
    test_frame.to_csv(test_data_path, index=False)

    comparison_path = metrics_dir / "model_comparison.csv"
    comparison_figure_path = figures_dir / "model_comparison_metrics.png"
    thesis_summary_path = metrics_dir / "model_comparison_for_thesis.md"

    monkeypatch.setattr(evaluate, "TEST_DATA_PATH", test_data_path)
    monkeypatch.setitem(evaluate.MODEL_ARTIFACTS["random_forest"], "model_path", rf_model_path)
    monkeypatch.setitem(evaluate.MODEL_ARTIFACTS["xgboost"], "model_path", xgb_model_path)
    monkeypatch.setitem(evaluate.MODEL_ARTIFACTS["random_forest"], "metrics_path", rf_metrics_path)
    monkeypatch.setitem(evaluate.MODEL_ARTIFACTS["xgboost"], "metrics_path", xgb_metrics_path)
    monkeypatch.setitem(evaluate.MODEL_ARTIFACTS["random_forest"], "figure_path", rf_figure_path)
    monkeypatch.setitem(evaluate.MODEL_ARTIFACTS["xgboost"], "figure_path", xgb_figure_path)

    monkeypatch.setattr(run_model_comparison.evaluate, "TEST_DATA_PATH", test_data_path)
    monkeypatch.setattr(run_model_comparison, "COMPARISON_PATH", comparison_path)
    monkeypatch.setattr(run_model_comparison, "COMPARISON_FIGURE_PATH", comparison_figure_path)
    monkeypatch.setattr(run_model_comparison, "THESIS_SUMMARY_PATH", thesis_summary_path)
    monkeypatch.setitem(
        run_model_comparison.evaluate.MODEL_ARTIFACTS["random_forest"], "model_path", rf_model_path
    )
    monkeypatch.setitem(
        run_model_comparison.evaluate.MODEL_ARTIFACTS["xgboost"], "model_path", xgb_model_path
    )

    result = run_model_comparison.run_comparison()

    assert comparison_path.exists()
    assert comparison_figure_path.exists()
    assert thesis_summary_path.exists()
    assert Path(result["comparison_path"]) == comparison_path

    comparison_frame = pd.read_csv(comparison_path)
    assert comparison_frame["model"].tolist() == ["Random Forest", "XGBoost"]
    for column in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert column in comparison_frame.columns
        assert comparison_frame[column].between(0.0, 1.0).all()

    summary_text = thesis_summary_path.read_text(encoding="utf-8")
    assert "Random Forest" in summary_text
    assert "XGBoost" in summary_text
    assert "preferred model for the next experimental stages" in summary_text

    assert rf_model_digest_before == file_digest(rf_model_path)
    assert xgb_model_digest_before == file_digest(xgb_model_path)
    assert rf_metrics_digest_before == file_digest(rf_metrics_path)
    assert xgb_metrics_digest_before == file_digest(xgb_metrics_path)
    assert rf_figure_digest_before == file_digest(rf_figure_path)
    assert xgb_figure_digest_before == file_digest(xgb_figure_path)
