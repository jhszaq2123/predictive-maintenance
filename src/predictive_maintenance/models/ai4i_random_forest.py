from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from predictive_maintenance.data.ai4i import build_ai4i_feature_frame, load_ai4i_dataset
from predictive_maintenance.features.ai4i import build_ai4i_preprocessor


@dataclass(frozen=True)
class Ai4iRandomForestResult:
    train_rows: int
    test_rows: int
    positive_rate_train: float
    positive_rate_test: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "positive_rate_train": self.positive_rate_train,
            "positive_rate_test": self.positive_rate_test,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }


def train_ai4i_random_forest(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, Ai4iRandomForestResult]:
    df = load_ai4i_dataset()
    X, y = build_ai4i_feature_frame(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_ai4i_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    result = Ai4iRandomForestResult(
        train_rows=len(X_train),
        test_rows=len(X_test),
        positive_rate_train=float(y_train.mean()),
        positive_rate_test=float(y_test.mean()),
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, y_proba)),
    )
    return pipeline, result


def top_feature_importances(model: Pipeline, limit: int = 10) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    frame = pd.DataFrame({"feature": feature_names, "importance": importances})
    return frame.sort_values("importance", ascending=False).head(limit).reset_index(drop=True)
