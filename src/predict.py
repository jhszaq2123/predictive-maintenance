from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from train_random_forest import MODEL_PATH


def load_model_bundle(model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Train the model before running predictions.")

    with model_path.open("rb") as file:
        return pickle.load(file)


def predict_failure(machine_data: dict) -> dict[str, float | int]:
    bundle = load_model_bundle()
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    metadata = bundle["metadata"]

    frame = pd.DataFrame([machine_data], columns=metadata["raw_feature_columns"])
    transformed = preprocessor.transform(frame)
    transformed_frame = pd.DataFrame(
        transformed,
        columns=metadata["model_feature_columns"],
    )
    probability = float(model.predict_proba(transformed_frame)[0, 1])
    prediction = int(model.predict(transformed_frame)[0])

    return {
        "failure_probability": probability,
        "prediction": prediction,
    }


def main() -> None:
    sample_machine = {
        "Type": "M",
        "Air temperature [K]": 298.5,
        "Process temperature [K]": 309.0,
        "Rotational speed [rpm]": 1450,
        "Torque [Nm]": 48.0,
        "Tool wear [min]": 120,
    }
    result = predict_failure(sample_machine)
    print(result)


if __name__ == "__main__":
    main()
