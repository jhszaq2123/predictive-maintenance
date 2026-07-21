from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import evaluate
import train_xgboost


EXPERIMENT_NAME = "ai4i_xgboost_baseline"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "ai4i_xgboost"
THESIS_SUMMARY_PATH = PROJECT_ROOT / "reports" / "metrics" / "xgboost_results_for_thesis.md"


def build_top_feature_importances(
    model, feature_names: list[str], limit: int = 10
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    return frame.sort_values("importance", ascending=False).head(limit).reset_index(drop=True)


def build_metrics_payload(
    metrics: dict[str, float | list[list[int]]],
    train_rows: int,
    test_rows: int,
    positive_rate_train: float,
    positive_rate_test: float,
) -> dict[str, object]:
    return {
        "experiment": EXPERIMENT_NAME,
        "dataset": "AI4I 2020",
        "target": train_xgboost.TARGET_COLUMN,
        "model": "XGBClassifier",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "positive_rate_train": positive_rate_train,
        "positive_rate_test": positive_rate_test,
    }


def write_experiment_summary(
    payload: dict[str, object], top_features: pd.DataFrame, output_path: Path
) -> None:
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)

    summary_lines = [
        "AI4I 2020 XGBoost baseline",
        f"Train rows: {payload['train_rows']}",
        f"Test rows: {payload['test_rows']}",
        f"Positive rate (train): {payload['positive_rate_train']:.4f}",
        f"Positive rate (test): {payload['positive_rate_test']:.4f}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1: {metrics['f1']:.4f}",
        f"ROC-AUC: {metrics['roc_auc']:.4f}",
        "",
        "Confusion matrix:",
        str(metrics["confusion_matrix"]),
        "",
        "Top feature importances:",
    ]
    summary_lines.extend(
        f"- {row.feature}: {row.importance:.4f}" for row in top_features.itertuples(index=False)
    )
    output_path.write_text("\n".join(summary_lines), encoding="utf-8")


def write_thesis_summary(payload: dict[str, object], output_path: Path) -> None:
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    confusion_matrix = metrics["confusion_matrix"]

    lines = [
        "# XGBoost baseline results for thesis drafting",
        "",
        "This note contains Sprint 3 baseline outputs reproduced through the official XGBoost runner.",
        "It documents only the XGBoost baseline and does not include any formal model comparison.",
        "",
        "## Experiment scope",
        "",
        "- Dataset: AI4I 2020",
        f"- Target: {payload['target']}",
        "- Model: XGBClassifier",
        f"- Training rows: {payload['train_rows']}",
        f"- Test rows: {payload['test_rows']}",
        f"- Positive rate in training split: {payload['positive_rate_train']:.4f}",
        f"- Positive rate in test split: {payload['positive_rate_test']:.4f}",
        "",
        "## Metrics",
        "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall: {metrics['recall']:.4f}",
        f"- F1 score: {metrics['f1']:.4f}",
        f"- ROC-AUC: {metrics['roc_auc']:.4f}",
        "",
        "## Confusion matrix",
        "",
        f"`{confusion_matrix}`",
        "",
        "## Interpretation draft",
        "",
        (
            "The reproduced XGBoost baseline achieved high overall classification quality on the AI4I 2020 "
            "dataset, with balanced accuracy-oriented performance and substantially stronger recall than the "
            "accepted Random Forest reference should be verified only in the formal comparison sprint."
        ),
        (
            "At this stage, the result should be interpreted strictly as an isolated Sprint 3 baseline "
            "obtained on the validated AI4I preprocessing workflow."
        ),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment() -> dict[str, object]:
    X_train, y_train = train_xgboost.load_training_data()
    model = train_xgboost.train_model(X_train, y_train)
    train_xgboost.save_model_bundle(model)

    X_test, y_test = evaluate.load_test_data()
    metrics = evaluate.evaluate_model(
        {"model": model, "input_format": "numpy"},
        X_test,
        y_test,
    )

    artifacts = evaluate.MODEL_ARTIFACTS["xgboost"]
    evaluate.save_metrics(metrics, artifacts["metrics_path"])
    evaluate.save_confusion_matrix(
        metrics["confusion_matrix"],
        artifacts["figure_path"],
        f"{artifacts['label']} confusion matrix",
    )

    top_features = build_top_feature_importances(model, X_train.columns.tolist(), limit=10)
    payload = build_metrics_payload(
        metrics=metrics,
        train_rows=len(X_train),
        test_rows=len(X_test),
        positive_rate_train=float(y_train.mean()),
        positive_rate_test=float(y_test.mean()),
    )

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    payload_path = EXPERIMENT_DIR / "metrics.json"
    top_features_path = EXPERIMENT_DIR / "top_feature_importances.csv"
    summary_path = EXPERIMENT_DIR / "summary.txt"

    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    top_features.to_csv(top_features_path, index=False)
    write_experiment_summary(payload, top_features, summary_path)

    THESIS_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_thesis_summary(payload, THESIS_SUMMARY_PATH)

    return {
        "payload_path": str(payload_path),
        "top_features_path": str(top_features_path),
        "summary_path": str(summary_path),
        "thesis_summary_path": str(THESIS_SUMMARY_PATH),
        "report_metrics_path": str(artifacts["metrics_path"]),
        "report_figure_path": str(artifacts["figure_path"]),
        "metrics": metrics,
    }


def main() -> None:
    result = run_experiment()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
