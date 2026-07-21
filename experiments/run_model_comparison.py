from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import evaluate


METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
COMPARISON_MODELS = ["random_forest", "xgboost"]
COMPARISON_PATH = PROJECT_ROOT / "reports" / "metrics" / "model_comparison.csv"
COMPARISON_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "model_comparison_metrics.png"
THESIS_SUMMARY_PATH = PROJECT_ROOT / "reports" / "metrics" / "model_comparison_for_thesis.md"


def collect_comparison_rows() -> list[dict[str, float | str]]:
    X_test, y_test = evaluate.load_test_data()
    comparison_rows: list[dict[str, float | str]] = []

    for model_name in COMPARISON_MODELS:
        bundle = evaluate.load_model_bundle(model_name)
        metrics = evaluate.evaluate_model(bundle, X_test, y_test)
        comparison_rows.append(
            {
                "model_key": model_name,
                "model": evaluate.MODEL_ARTIFACTS[model_name]["label"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )

    return comparison_rows


def build_comparison_frame(comparison_rows: list[dict[str, float | str]]) -> pd.DataFrame:
    comparison_frame = pd.DataFrame(comparison_rows)
    ordered_columns = ["model_key", "model", *METRIC_COLUMNS]
    return comparison_frame[ordered_columns]


def save_comparison_csv(comparison_frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_frame.to_csv(output_path, index=False)


def save_comparison_figure(comparison_frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = comparison_frame.melt(
        id_vars=["model"],
        value_vars=METRIC_COLUMNS,
        var_name="metric",
        value_name="value",
    )

    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
        "roc_auc": "ROC-AUC",
    }
    models = comparison_frame["model"].tolist()
    metrics = METRIC_COLUMNS
    bar_width = 0.35
    positions = range(len(metrics))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for index, model_label in enumerate(models):
        model_values = [
            float(
                plot_frame[
                    (plot_frame["model"] == model_label) & (plot_frame["metric"] == metric_name)
                ]["value"].iloc[0]
            )
            for metric_name in metrics
        ]
        shifted_positions = [position + (index - 0.5) * bar_width for position in positions]
        ax.bar(shifted_positions, model_values, width=bar_width, label=model_label)

    ax.set_xticks(list(positions))
    ax.set_xticklabels([metric_labels[metric_name] for metric_name in metrics])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Random Forest vs XGBoost baseline comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def select_preferred_model(comparison_frame: pd.DataFrame) -> pd.Series:
    ranking = comparison_frame.sort_values(
        by=["f1", "recall", "roc_auc", "accuracy", "precision"],
        ascending=False,
    ).reset_index(drop=True)
    return ranking.iloc[0]


def write_thesis_summary(
    comparison_frame: pd.DataFrame,
    preferred_model: pd.Series,
    output_path: Path,
) -> None:
    rf_row = comparison_frame.loc[comparison_frame["model_key"] == "random_forest"].iloc[0]
    xgb_row = comparison_frame.loc[comparison_frame["model_key"] == "xgboost"].iloc[0]

    lines = [
        "# Random Forest vs XGBoost baseline comparison",
        "",
        "This note summarizes the formal Sprint 4 comparison performed on the accepted AI4I 2020 baseline artifacts.",
        "Both models were assessed on the same processed dataset, with the same train/test split, preprocessing, and evaluation procedure.",
        "",
        "## Compared metrics",
        "",
        f"- Random Forest: accuracy {rf_row['accuracy']:.4f}, precision {rf_row['precision']:.4f}, recall {rf_row['recall']:.4f}, F1 {rf_row['f1']:.4f}, ROC-AUC {rf_row['roc_auc']:.4f}",
        f"- XGBoost: accuracy {xgb_row['accuracy']:.4f}, precision {xgb_row['precision']:.4f}, recall {xgb_row['recall']:.4f}, F1 {xgb_row['f1']:.4f}, ROC-AUC {xgb_row['roc_auc']:.4f}",
        "",
        "## Interpretation",
        "",
        "Random Forest reached slightly higher accuracy and precision, which means it generated fewer false alarms and achieved marginally better overall correctness on the strongly dominant negative class.",
        "At the same time, XGBoost obtained clearly higher recall, higher F1-score, and slightly better ROC-AUC, indicating stronger ability to recover failure cases under class imbalance.",
        "The main weakness of Random Forest in this comparison is lower recall, which means a larger share of actual failures remained undetected.",
        "The main weakness of XGBoost is lower precision and a small decrease in accuracy relative to Random Forest, which reflects a higher number of false positive predictions.",
        "",
        "## Model selected for further experiments",
        "",
        (
            f"The preferred model for the next experimental stages is {preferred_model['model']}. "
            "The selection is based primarily on F1-score and recall, with ROC-AUC treated as a supporting criterion, "
            "because the project focuses on reliable detection of failure cases in an imbalanced classification setting."
        ),
        "This choice should be interpreted as the outcome of the baseline comparison only and not as evidence of a universally best model beyond the evaluated AI4I 2020 workflow.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_comparison() -> dict[str, object]:
    comparison_rows = collect_comparison_rows()
    comparison_frame = build_comparison_frame(comparison_rows)
    preferred_model = select_preferred_model(comparison_frame)

    save_comparison_csv(comparison_frame, COMPARISON_PATH)
    save_comparison_figure(comparison_frame, COMPARISON_FIGURE_PATH)
    write_thesis_summary(comparison_frame, preferred_model, THESIS_SUMMARY_PATH)

    return {
        "comparison_path": str(COMPARISON_PATH),
        "comparison_figure_path": str(COMPARISON_FIGURE_PATH),
        "thesis_summary_path": str(THESIS_SUMMARY_PATH),
        "preferred_model": str(preferred_model["model"]),
        "results": comparison_rows,
    }


def main() -> None:
    result = run_comparison()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
