import argparse
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate
import train_random_forest


PROCESSED_FILES_AVAILABLE = (
    train_random_forest.TRAIN_DATA_PATH.exists()
    and train_random_forest.PREPROCESSOR_PATH.exists()
    and train_random_forest.FEATURE_METADATA_PATH.exists()
    and evaluate.TEST_DATA_PATH.exists()
)

pytestmark = pytest.mark.skipif(
    not PROCESSED_FILES_AVAILABLE,
    reason="Processed AI4I artifacts are not available locally.",
)


def test_random_forest_training_bundle_can_be_saved(tmp_path, monkeypatch) -> None:
    X_train, y_train = train_random_forest.load_training_data()
    model = train_random_forest.train_model(X_train.head(200), y_train.head(200))

    output_dir = tmp_path / "models"
    output_path = output_dir / "random_forest.pkl"
    monkeypatch.setattr(train_random_forest, "MODEL_DIR", output_dir)
    monkeypatch.setattr(train_random_forest, "MODEL_PATH", output_path)

    train_random_forest.save_model_bundle(model)

    assert output_path.exists()

    with output_path.open("rb") as file:
        bundle = pickle.load(file)

    assert {"model", "preprocessor", "metadata"} == set(bundle.keys())
    assert bundle["metadata"]["target_column"] == "Machine failure"


def test_random_forest_evaluation_writes_only_random_forest_artifacts(
    tmp_path, monkeypatch
) -> None:
    figures_dir = tmp_path / "figures"
    metrics_dir = tmp_path / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    legacy_confusion_path = figures_dir / "confusion_matrix.png"
    legacy_confusion_path.write_text("legacy", encoding="utf-8")
    comparison_path = metrics_dir / "model_comparison.csv"
    comparison_path.write_text("legacy", encoding="utf-8")

    rf_metrics_path = metrics_dir / "random_forest_metrics.json"
    rf_figure_path = figures_dir / "random_forest_confusion_matrix.png"

    monkeypatch.setattr(evaluate, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(evaluate, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(evaluate, "COMPARISON_PATH", comparison_path)
    monkeypatch.setitem(
        evaluate.MODEL_ARTIFACTS["random_forest"], "metrics_path", rf_metrics_path
    )
    monkeypatch.setitem(
        evaluate.MODEL_ARTIFACTS["random_forest"], "figure_path", rf_figure_path
    )
    monkeypatch.setattr(
        evaluate, "parse_args", lambda: argparse.Namespace(model="random_forest")
    )

    evaluate.main()

    assert rf_metrics_path.exists()
    assert rf_figure_path.exists()
    assert not legacy_confusion_path.exists()
    assert not comparison_path.exists()
