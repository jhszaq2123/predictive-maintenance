from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_ai4i_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    categorical_features = ["Type"]

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("scaler", StandardScaler())]), numeric_features),
            (
                "categorical",
                Pipeline(
                    [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
                ),
                categorical_features,
            ),
        ]
    )
