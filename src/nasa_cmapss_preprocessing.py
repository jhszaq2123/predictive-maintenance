from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "c_mapss" / "CMAPSSData"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "nasa"
REPORTS_DIR = PROJECT_ROOT / "reports" / "nasa"

SUPPORTED_SUBSETS = ("FD001", "FD002", "FD003", "FD004")
DEFAULT_SUBSET = "FD001"
DEFAULT_SPLIT = "train"
DEFAULT_SEQUENCE_LENGTH = 30

ENGINE_ID_COLUMN = "unit_id"
CYCLE_COLUMN = "cycle"
TARGET_COLUMN = "RUL"
OPERATING_SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{index:02d}" for index in range(1, 22)]
CMAPSS_COLUMNS = [ENGINE_ID_COLUMN, CYCLE_COLUMN, *OPERATING_SETTING_COLUMNS, *SENSOR_COLUMNS]
FEATURE_COLUMNS = [*OPERATING_SETTING_COLUMNS, *SENSOR_COLUMNS]


@dataclass(frozen=True)
class SequenceArtifact:
    features: np.ndarray
    targets: np.ndarray
    engine_ids: np.ndarray
    end_cycles: np.ndarray


def subset_file_path(subset: str, split: str) -> Path:
    normalized_subset = subset.upper()
    normalized_split = split.lower()
    if normalized_subset not in SUPPORTED_SUBSETS:
        supported = ", ".join(SUPPORTED_SUBSETS)
        raise ValueError(f"Unsupported CMAPSS subset '{subset}'. Supported values: {supported}")
    if normalized_split not in {"train", "test", "rul"}:
        raise ValueError("Split must be one of: train, test, rul")

    filename = {
        "train": f"train_{normalized_subset}.txt",
        "test": f"test_{normalized_subset}.txt",
        "rul": f"RUL_{normalized_subset}.txt",
    }[normalized_split]
    return RAW_DATA_DIR / filename


def load_cmapss_split(subset: str = DEFAULT_SUBSET, split: str = DEFAULT_SPLIT) -> pd.DataFrame:
    path = subset_file_path(subset, split)
    if not path.exists():
        raise FileNotFoundError(f"NASA CMAPSS file not found: {path}")

    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    frame = frame.dropna(axis=1, how="all")

    if split.lower() == "rul":
        frame.columns = [TARGET_COLUMN]
        return frame

    if frame.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(
            f"Unexpected number of columns in {path.name}: {frame.shape[1]}. "
            f"Expected {len(CMAPSS_COLUMNS)} after removing empty columns."
        )

    frame.columns = CMAPSS_COLUMNS
    return frame


def preprocess_cmapss_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing_columns = [column for column in CMAPSS_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"CMAPSS frame is missing required columns: {missing_columns}")

    processed = frame.copy()
    processed[ENGINE_ID_COLUMN] = processed[ENGINE_ID_COLUMN].astype("int64")
    processed[CYCLE_COLUMN] = processed[CYCLE_COLUMN].astype("int64")

    float_columns = [column for column in processed.columns if column not in {ENGINE_ID_COLUMN, CYCLE_COLUMN}]
    processed[float_columns] = processed[float_columns].astype("float64")

    duplicate_rows = int(processed.duplicated().sum())
    missing_values_total = int(processed.isna().sum().sum())

    processed = processed.sort_values([ENGINE_ID_COLUMN, CYCLE_COLUMN], kind="stable").reset_index(drop=True)

    verification = {
        "row_count": int(len(processed)),
        "engine_count": int(processed[ENGINE_ID_COLUMN].nunique()),
        "duplicate_rows": duplicate_rows,
        "missing_values_total": missing_values_total,
        "dtype_summary": {column: str(dtype) for column, dtype in processed.dtypes.items()},
        "is_sorted_by_engine_and_cycle": bool(
            processed[[ENGINE_ID_COLUMN, CYCLE_COLUMN]].equals(
                processed[[ENGINE_ID_COLUMN, CYCLE_COLUMN]]
                .sort_values([ENGINE_ID_COLUMN, CYCLE_COLUMN], kind="stable")
                .reset_index(drop=True)
            )
        ),
    }
    return processed, verification


def add_rul_column(frame: pd.DataFrame) -> pd.DataFrame:
    processed = frame.copy()
    max_cycles = processed.groupby(ENGINE_ID_COLUMN)[CYCLE_COLUMN].transform("max")
    processed[TARGET_COLUMN] = (max_cycles - processed[CYCLE_COLUMN]).astype("int64")
    return processed


def build_engine_cycle_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(ENGINE_ID_COLUMN, sort=True)
        .agg(
            start_cycle=(CYCLE_COLUMN, "min"),
            end_cycle=(CYCLE_COLUMN, "max"),
            cycle_count=(CYCLE_COLUMN, "count"),
            min_rul=(TARGET_COLUMN, "min"),
            max_rul=(TARGET_COLUMN, "max"),
        )
        .reset_index()
    )
    return summary


def build_dataset_summary(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    rul_reference: pd.DataFrame,
    verification: dict[str, Any],
    subset: str,
) -> dict[str, Any]:
    train_cycle_summary = build_engine_cycle_summary(train_frame)
    return {
        "dataset": "NASA CMAPSS",
        "subset": subset.upper(),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_engine_count": int(train_frame[ENGINE_ID_COLUMN].nunique()),
        "test_engine_count": int(test_frame[ENGINE_ID_COLUMN].nunique()),
        "train_cycle_min": int(train_frame[CYCLE_COLUMN].min()),
        "train_cycle_max": int(train_frame[CYCLE_COLUMN].max()),
        "test_cycle_min": int(test_frame[CYCLE_COLUMN].min()),
        "test_cycle_max": int(test_frame[CYCLE_COLUMN].max()),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "verification": verification,
        "rul_statistics": {
            "min": int(train_frame[TARGET_COLUMN].min()),
            "max": int(train_frame[TARGET_COLUMN].max()),
            "mean": float(train_frame[TARGET_COLUMN].mean()),
            "median": float(train_frame[TARGET_COLUMN].median()),
            "std": float(train_frame[TARGET_COLUMN].std(ddof=0)),
        },
        "engine_cycle_statistics": {
            "min_cycles_per_engine": int(train_cycle_summary["cycle_count"].min()),
            "max_cycles_per_engine": int(train_cycle_summary["cycle_count"].max()),
            "mean_cycles_per_engine": float(train_cycle_summary["cycle_count"].mean()),
        },
        "test_rul_reference_rows": int(len(rul_reference)),
    }


def create_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...] = FEATURE_COLUMNS,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    target_column: str = TARGET_COLUMN,
    target_position: str = "last",
) -> SequenceArtifact:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer")
    if target_position not in {"last", "next"}:
        raise ValueError("target_position must be either 'last' or 'next'")

    sorted_frame = frame.sort_values([ENGINE_ID_COLUMN, CYCLE_COLUMN], kind="stable").reset_index(drop=True)

    sequences: list[np.ndarray] = []
    targets: list[float] = []
    engine_ids: list[int] = []
    end_cycles: list[int] = []

    for engine_id, engine_frame in sorted_frame.groupby(ENGINE_ID_COLUMN, sort=True):
        engine_features = engine_frame.loc[:, list(feature_columns)].to_numpy(dtype=np.float64, copy=True)
        engine_targets = engine_frame.loc[:, target_column].to_numpy(copy=True)
        engine_cycles = engine_frame.loc[:, CYCLE_COLUMN].to_numpy(copy=True)

        if target_position == "last":
            max_start = len(engine_frame) - sequence_length + 1
        else:
            max_start = len(engine_frame) - sequence_length

        if max_start <= 0:
            continue

        for start_index in range(max_start):
            end_index = start_index + sequence_length
            sequences.append(engine_features[start_index:end_index])
            if target_position == "last":
                target_index = end_index - 1
            else:
                target_index = end_index
            targets.append(float(engine_targets[target_index]))
            engine_ids.append(int(engine_id))
            end_cycles.append(int(engine_cycles[end_index - 1]))

    if sequences:
        sequence_array = np.stack(sequences).astype(np.float64)
        target_array = np.asarray(targets, dtype=np.float64)
        engine_array = np.asarray(engine_ids, dtype=np.int64)
        end_cycle_array = np.asarray(end_cycles, dtype=np.int64)
    else:
        sequence_array = np.empty((0, sequence_length, len(feature_columns)), dtype=np.float64)
        target_array = np.empty((0,), dtype=np.float64)
        engine_array = np.empty((0,), dtype=np.int64)
        end_cycle_array = np.empty((0,), dtype=np.int64)

    return SequenceArtifact(
        features=sequence_array,
        targets=target_array,
        engine_ids=engine_array,
        end_cycles=end_cycle_array,
    )


def build_feature_metadata() -> dict[str, Any]:
    return {
        "dataset": "NASA CMAPSS",
        "entity_id_column": ENGINE_ID_COLUMN,
        "time_order_column": CYCLE_COLUMN,
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "operating_setting_columns": OPERATING_SETTING_COLUMNS,
        "sensor_columns": SENSOR_COLUMNS,
        "normalization_applied": False,
    }


def build_sequence_metadata(
    subset: str,
    split: str,
    sequence_length: int,
    target_position: str,
    artifact: SequenceArtifact,
) -> dict[str, Any]:
    return {
        "dataset": "NASA CMAPSS",
        "subset": subset.upper(),
        "split": split.lower(),
        "sequence_length": int(sequence_length),
        "target_column": TARGET_COLUMN,
        "target_position": target_position,
        "feature_columns": FEATURE_COLUMNS,
        "num_sequences": int(artifact.features.shape[0]),
        "num_features": int(artifact.features.shape[2]) if artifact.features.ndim == 3 else len(FEATURE_COLUMNS),
        "engine_count_in_sequences": int(len(np.unique(artifact.engine_ids))) if len(artifact.engine_ids) else 0,
        "feature_dtype": str(artifact.features.dtype),
        "target_dtype": str(artifact.targets.dtype),
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_preprocessing_summary(
    path: Path,
    subset: str,
    split: str,
    sequence_length: int,
    summary: dict[str, Any],
    sequence_metadata: dict[str, Any],
) -> None:
    lines = [
        "# NASA CMAPSS preprocessing summary",
        "",
        f"- Dataset: NASA CMAPSS {subset.upper()}",
        f"- Prepared split: {split.lower()}",
        f"- Rows after preprocessing: {summary['train_rows'] if split.lower() == 'train' else summary['test_rows']}",
        f"- Number of engines in prepared split: {summary['train_engine_count'] if split.lower() == 'train' else summary['test_engine_count']}",
        f"- Sequence length: {sequence_length}",
        "",
        "## Preprocessing",
        "",
        "- Applied explicit CMAPSS column naming for unit identifier, cycle number, operating settings, and sensor channels.",
        "- Verified numeric dtypes, missing values, duplicate rows, and sorted order by engine and operating cycle.",
        "- No normalization or scaling was applied at this stage to keep Sprint 5 limited to reproducible data preparation.",
        "",
        "## RUL generation",
        "",
        "- Computed Remaining Useful Life for each prepared observation using `RUL = max_cycle(engine) - current_cycle`.",
        f"- RUL statistics for the prepared training split: min {summary['rul_statistics']['min']}, max {summary['rul_statistics']['max']}, mean {summary['rul_statistics']['mean']:.4f}.",
        "",
        "## Sequence generation",
        "",
        "- Used sliding windows built separately inside each engine trajectory.",
        "- No sequence crosses engine boundaries, and no engine mixing is allowed within a single window.",
        f"- Exported {sequence_metadata['num_sequences']} sequences with {sequence_metadata['num_features']} features per timestep.",
        "",
        "## Assumptions",
        "",
        "- Sprint 5 prepares the FD001 training split for later sequence models and does not train any neural architecture.",
        "- The official exported target is the RUL value aligned with the last timestep of each sequence.",
        "- Test-set RUL reference files are inspected for dataset completeness but are not transformed into training sequences in this sprint.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_preprocessing(
    subset: str = DEFAULT_SUBSET,
    split: str = DEFAULT_SPLIT,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    target_position: str = "last",
) -> dict[str, str | int]:
    train_raw = load_cmapss_split(subset=subset, split="train")
    test_raw = load_cmapss_split(subset=subset, split="test")
    rul_reference = load_cmapss_split(subset=subset, split="rul")

    prepared_split_raw = train_raw if split.lower() == "train" else test_raw
    prepared_split, verification = preprocess_cmapss_frame(prepared_split_raw)
    prepared_with_rul = add_rul_column(prepared_split)

    summary = build_dataset_summary(
        train_frame=add_rul_column(preprocess_cmapss_frame(train_raw)[0]),
        test_frame=preprocess_cmapss_frame(test_raw)[0],
        rul_reference=rul_reference,
        verification=verification,
        subset=subset,
    )
    engine_cycle_summary = build_engine_cycle_summary(prepared_with_rul)
    artifact = create_sequences(
        frame=prepared_with_rul,
        feature_columns=FEATURE_COLUMNS,
        sequence_length=sequence_length,
        target_column=TARGET_COLUMN,
        target_position=target_position,
    )
    feature_metadata = build_feature_metadata()
    sequence_metadata = build_sequence_metadata(
        subset=subset,
        split=split,
        sequence_length=sequence_length,
        target_position=target_position,
        artifact=artifact,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    processed_dataset_path = PROCESSED_DIR / f"{subset.lower()}_{split.lower()}_processed.csv"
    sequence_path = PROCESSED_DIR / f"{subset.lower()}_{split.lower()}_sequences.npz"
    sequence_metadata_path = PROCESSED_DIR / f"{subset.lower()}_{split.lower()}_sequence_metadata.json"
    feature_metadata_path = PROCESSED_DIR / f"{subset.lower()}_feature_metadata.json"

    prepared_with_rul.to_csv(processed_dataset_path, index=False)
    np.savez_compressed(
        sequence_path,
        X=artifact.features,
        y=artifact.targets,
        engine_id=artifact.engine_ids,
        end_cycle=artifact.end_cycles,
    )
    save_json(sequence_metadata_path, sequence_metadata)
    save_json(feature_metadata_path, feature_metadata)

    dataset_summary_path = REPORTS_DIR / f"{subset.lower()}_dataset_summary.json"
    engine_cycle_summary_path = REPORTS_DIR / f"{subset.lower()}_{split.lower()}_engine_cycle_summary.csv"
    feature_list_path = REPORTS_DIR / f"{subset.lower()}_feature_list.json"
    rul_statistics_path = REPORTS_DIR / f"{subset.lower()}_rul_statistics.json"
    preprocessing_summary_path = REPORTS_DIR / "nasa_preprocessing_summary.md"

    save_json(dataset_summary_path, summary)
    engine_cycle_summary.to_csv(engine_cycle_summary_path, index=False)
    save_json(feature_list_path, feature_metadata)
    save_json(rul_statistics_path, summary["rul_statistics"])
    write_preprocessing_summary(
        path=preprocessing_summary_path,
        subset=subset,
        split=split,
        sequence_length=sequence_length,
        summary=summary,
        sequence_metadata=sequence_metadata,
    )

    return {
        "processed_dataset_path": str(processed_dataset_path),
        "sequence_path": str(sequence_path),
        "sequence_metadata_path": str(sequence_metadata_path),
        "feature_metadata_path": str(feature_metadata_path),
        "dataset_summary_path": str(dataset_summary_path),
        "engine_cycle_summary_path": str(engine_cycle_summary_path),
        "feature_list_path": str(feature_list_path),
        "rul_statistics_path": str(rul_statistics_path),
        "preprocessing_summary_path": str(preprocessing_summary_path),
        "num_sequences": int(artifact.features.shape[0]),
    }
