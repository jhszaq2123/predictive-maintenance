from __future__ import annotations

import json
from datetime import datetime, UTC

from predictive_maintenance.config import EXPERIMENTS_DIR
from predictive_maintenance.models.ai4i_random_forest import (
    top_feature_importances,
    train_ai4i_random_forest,
)


def main() -> None:
    model, metrics = train_ai4i_random_forest()
    top_features = top_feature_importances(model, limit=10)

    output_dir = EXPERIMENTS_DIR / "ai4i_random_forest"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "experiment": "ai4i_random_forest_baseline",
        "dataset": "AI4I 2020",
        "target": "Machine failure",
        "model": "RandomForestClassifier",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "metrics": metrics.to_dict(),
    }

    metrics_path = output_dir / "metrics.json"
    top_features_path = output_dir / "top_feature_importances.csv"
    summary_path = output_dir / "summary.txt"

    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    top_features.to_csv(top_features_path, index=False)

    summary_lines = [
        "AI4I 2020 Random Forest baseline",
        f"Train rows: {metrics.train_rows}",
        f"Test rows: {metrics.test_rows}",
        f"Accuracy: {metrics.accuracy:.4f}",
        f"Precision: {metrics.precision:.4f}",
        f"Recall: {metrics.recall:.4f}",
        f"F1: {metrics.f1:.4f}",
        f"ROC-AUC: {metrics.roc_auc:.4f}",
        "",
        "Top feature importances:",
    ]
    summary_lines.extend(
        f"- {row.feature}: {row.importance:.4f}" for row in top_features.itertuples(index=False)
    )
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
