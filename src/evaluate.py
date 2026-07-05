from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_preprocessing import TARGET_COLUMN, TEST_DATA_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_ARTIFACTS = {
    "random_forest": {
        "label": "Random Forest",
        "model_path": MODELS_DIR / "random_forest.pkl",
        "metrics_path": METRICS_DIR / "random_forest_metrics.json",
        "figure_path": FIGURES_DIR / "random_forest_confusion_matrix.png",
    },
    "xgboost": {
        "label": "XGBoost",
        "model_path": MODELS_DIR / "xgboost.pkl",
        "metrics_path": METRICS_DIR / "xgboost_metrics.json",
        "figure_path": FIGURES_DIR / "xgboost_confusion_matrix.png",
    },
}
COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"


def load_model_bundle(model_name: str) -> dict[str, Any]:
    model_path = MODEL_ARTIFACTS[model_name]["model_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found for '{model_name}'. Train it first.")

    with model_path.open("rb") as file:
        return pickle.load(file)


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed test data not found. Run src/data_preprocessing.py first."
        )

    test_frame = pd.read_csv(TEST_DATA_PATH)
    X_test = test_frame.drop(columns=[TARGET_COLUMN])
    y_test = test_frame[TARGET_COLUMN]
    return X_test, y_test


def prepare_model_input(bundle: dict[str, Any], X_test: pd.DataFrame):
    if bundle.get("input_format") == "numpy":
        return X_test.to_numpy()
    return X_test


def evaluate_model(
    bundle: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float | list[list[int]]]:
    model = bundle["model"]
    model_input = prepare_model_input(bundle, X_test)
    y_pred = model.predict(model_input)
    y_prob = model.predict_proba(model_input)[:, 1]
    matrix = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": matrix.tolist(),
    }


def save_confusion_matrix(matrix, output_path: Path, title: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(matrix),
        display_labels=["No failure", "Failure"],
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_metrics(metrics: dict[str, float | list[list[int]]], output_path: Path) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_model_comparison(comparison_rows: list[dict[str, float | str]]) -> None:
    if not comparison_rows:
        return

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame.to_csv(COMPARISON_PATH, index=False)


def remove_stale_random_forest_artifacts() -> None:
    # Remove legacy artifacts from the earlier shared-evaluation workflow.
    legacy_paths = [
        FIGURES_DIR / "confusion_matrix.png",
        COMPARISON_PATH,
    ]

    for artifact_path in legacy_paths:
        if artifact_path.exists():
            artifact_path.unlink()


def evaluate_single_model(model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    bundle = load_model_bundle(model_name)
    metrics = evaluate_model(bundle, X_test, y_test)

    artifacts = MODEL_ARTIFACTS[model_name]
    save_metrics(metrics, artifacts["metrics_path"])
    save_confusion_matrix(
        metrics["confusion_matrix"],
        artifacts["figure_path"],
        f"{artifacts['label']} confusion matrix",
    )

    print(f"\n{artifacts['label']}")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to: {artifacts['metrics_path']}")
    print(f"Saved confusion matrix figure to: {artifacts['figure_path']}")

    return {
        "model": artifacts["label"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained predictive maintenance models.")
    parser.add_argument(
        "--model",
        choices=["random_forest", "xgboost", "all"],
        default="all",
        help="Model to evaluate. Default evaluates all available trained models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X_test, y_test = load_test_data()
    requested_models = (
        list(MODEL_ARTIFACTS.keys()) if args.model == "all" else [args.model]
    )

    if args.model == "random_forest":
        remove_stale_random_forest_artifacts()

    comparison_rows: list[dict[str, float | str]] = []
    missing_models: list[str] = []

    for model_name in requested_models:
        try:
            comparison_rows.append(evaluate_single_model(model_name, X_test, y_test))
        except FileNotFoundError:
            if args.model == "all":
                missing_models.append(model_name)
                continue
            raise

    if not comparison_rows:
        missing_list = ", ".join(missing_models) or args.model
        raise FileNotFoundError(f"No trained models available for evaluation: {missing_list}")

    if args.model == "all" and len(comparison_rows) > 1:
        save_model_comparison(comparison_rows)
        print(f"Saved model comparison to: {COMPARISON_PATH}")

    if missing_models:
        print(f"Skipped models without saved artifacts: {', '.join(missing_models)}")


if __name__ == "__main__":
    main()
