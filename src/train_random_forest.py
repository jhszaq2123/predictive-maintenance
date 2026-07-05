from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import (
    FEATURE_METADATA_PATH,
    PREPROCESSOR_PATH,
    TARGET_COLUMN,
    TRAIN_DATA_PATH,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "random_forest.pkl"


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed training data not found. Run src/data_preprocessing.py first."
        )

    train_frame = pd.read_csv(TRAIN_DATA_PATH)
    X_train = train_frame.drop(columns=[TARGET_COLUMN])
    y_train = train_frame[TARGET_COLUMN]
    return X_train, y_train


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def save_model_bundle(model: RandomForestClassifier) -> None:
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
