from dataclasses import dataclass
from pathlib import Path

from predictive_maintenance.config import RAW_DATA_DIR


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    raw_dir: Path
    description: str
    primary_task: str


DATASETS: dict[str, DatasetSpec] = {
    "ai4i2020": DatasetSpec(
        name="AI4I 2020",
        raw_dir=RAW_DATA_DIR / "ai4i2020",
        description="Tabular dataset for predictive maintenance classification experiments.",
        primary_task="classification",
    ),
    "c_mapss": DatasetSpec(
        name="NASA CMAPSS",
        raw_dir=RAW_DATA_DIR / "c_mapss",
        description="Time-series turbofan degradation dataset for RUL prediction.",
        primary_task="regression_rul",
    ),
    "secom": DatasetSpec(
        name="SECOM",
        raw_dir=RAW_DATA_DIR / "secom",
        description="Semiconductor manufacturing dataset for anomaly-related experiments.",
        primary_task="anomaly_detection",
    ),
}


def get_dataset_spec(dataset_key: str) -> DatasetSpec:
    try:
        return DATASETS[dataset_key]
    except KeyError as exc:
        available = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset '{dataset_key}'. Available: {available}") from exc
