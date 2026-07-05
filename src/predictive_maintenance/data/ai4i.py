from __future__ import annotations

import pandas as pd

from predictive_maintenance.data.loaders import load_csv_dataset


TARGET_COLUMN = "Machine failure"
DROP_COLUMNS = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def load_ai4i_dataset() -> pd.DataFrame:
    return load_csv_dataset("ai4i2020", "ai4i2020.csv")


def build_ai4i_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN]).copy()
    features["Type"] = features["Type"].astype("category")
    target = df[TARGET_COLUMN].copy()
    return features, target
