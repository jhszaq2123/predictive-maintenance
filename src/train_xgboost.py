from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from data_preprocessing import (
    FEATURE_METADATA_PATH,
    PREPROCESSOR_PATH,
    TARGET_COLUMN,
    TRAIN_DATA_PATH,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "xgboost.pkl"


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed training data not found. Run src/data_preprocessing.py first."
        )

    train_frame = pd.read_csv(TRAIN_DATA_PATH)
    X_train = train_frame.drop(columns=[TARGET_COLUMN])
    y_train = train_frame[TARGET_COLUMN]
    return X_train, y_train


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count if positive_count else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
    )
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    return model


def save_model_bundle(model: XGBClassifier) -> None:
    if not PREPROCESSOR_PATH.exists() or not FEATURE_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Preprocessing artifacts are missing. Run src/data_preprocessing.py first."
        )

    with PREPROCESSOR_PATH.open("rb") as file:
        preprocessor = pickle.load(file)

    metadata = json.loads(FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "metadata": metadata,
        "input_format": "numpy",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as file:
        pickle.dump(bundle, file)


def main() -> None:
    X_train, y_train = load_training_data()
    model = train_model(X_train, y_train)
    save_model_bundle(model)
    print(f"Saved model bundle to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
