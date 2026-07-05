from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.pkl"
FEATURE_METADATA_PATH = PROCESSED_DIR / "feature_metadata.json"
TRAIN_DATA_PATH = PROCESSED_DIR / "train_processed.csv"
TEST_DATA_PATH = PROCESSED_DIR / "test_processed.csv"
TARGET_COLUMN = "Machine failure"

IDENTIFIER_COLUMNS = ["UDI", "Product ID"]
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
CATEGORICAL_COLUMNS = ["Type"]
NUMERIC_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place AI4I 2020 CSV in data/raw/ai4i2020.csv."
        )
    return pd.read_csv(path)


def build_one_hot_encoder() -> OneHotEncoder:
    # Keep compatibility with multiple scikit-learn versions.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    # Numeric values are scaled, while the product type is one-hot encoded.
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("scaler", StandardScaler())]), NUMERIC_COLUMNS),
            (
                "categorical",
                build_one_hot_encoder(),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required_columns = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    cleaned = df.drop(columns=IDENTIFIER_COLUMNS + LEAKAGE_COLUMNS)
    X = cleaned.drop(columns=[TARGET_COLUMN]).copy()
    y = cleaned[TARGET_COLUMN].copy()
    return X, y


def save_processed_splits(
    X_train_transformed,
    X_test_transformed,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_frame = pd.DataFrame(X_train_transformed, columns=feature_names)
    train_frame[TARGET_COLUMN] = y_train.to_numpy()
    train_frame.to_csv(TRAIN_DATA_PATH, index=False)

    test_frame = pd.DataFrame(X_test_transformed, columns=feature_names)
    test_frame[TARGET_COLUMN] = y_test.to_numpy()
    test_frame.to_csv(TEST_DATA_PATH, index=False)


def save_preprocessing_artifacts(preprocessor: ColumnTransformer, raw_columns: list[str]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with PREPROCESSOR_PATH.open("wb") as file:
        pickle.dump(preprocessor, file)

    metadata = {
        "raw_feature_columns": raw_columns,
        "model_feature_columns": preprocessor.get_feature_names_out().tolist(),
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "target_column": TARGET_COLUMN,
    }
    FEATURE_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    df = load_raw_data()
    X, y = split_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()

    save_processed_splits(
        X_train_transformed=X_train_transformed,
        X_test_transformed=X_test_transformed,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
    )
    save_preprocessing_artifacts(preprocessor=preprocessor, raw_columns=X.columns.tolist())

    print(f"Saved processed train data to: {TRAIN_DATA_PATH}")
    print(f"Saved processed test data to: {TEST_DATA_PATH}")
    print(f"Saved preprocessor to: {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    main()
