import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import nasa_cmapss_preprocessing as nasa


def make_cmapss_like_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for engine_id, cycles in [(1, 4), (2, 3)]:
        for cycle in range(1, cycles + 1):
            row: dict[str, float | int] = {
                nasa.ENGINE_ID_COLUMN: engine_id,
                nasa.CYCLE_COLUMN: cycle,
            }
            for index, column in enumerate(nasa.OPERATING_SETTING_COLUMNS, start=1):
                row[column] = float(engine_id * 10 + cycle + index)
            for sensor_index, column in enumerate(nasa.SENSOR_COLUMNS, start=1):
                row[column] = float(engine_id * 100 + cycle + sensor_index)
            rows.append(row)
    frame = pd.DataFrame(rows)
    return frame[nasa.CMAPSS_COLUMNS]


def test_add_rul_column_computes_remaining_cycles_per_engine() -> None:
    frame = make_cmapss_like_frame()

    result = nasa.add_rul_column(frame)

    engine_1_rul = result.loc[result[nasa.ENGINE_ID_COLUMN] == 1, nasa.TARGET_COLUMN].tolist()
    engine_2_rul = result.loc[result[nasa.ENGINE_ID_COLUMN] == 2, nasa.TARGET_COLUMN].tolist()

    assert engine_1_rul == [3, 2, 1, 0]
    assert engine_2_rul == [2, 1, 0]


def test_create_sequences_preserves_engine_boundaries_and_length() -> None:
    frame = nasa.add_rul_column(make_cmapss_like_frame())

    artifact = nasa.create_sequences(frame, sequence_length=2, target_position="last")

    assert artifact.features.shape == (5, 2, len(nasa.FEATURE_COLUMNS))
    assert artifact.targets.tolist() == [2.0, 1.0, 0.0, 1.0, 0.0]
    assert artifact.engine_ids.tolist() == [1, 1, 1, 2, 2]
    assert artifact.end_cycles.tolist() == [2, 3, 4, 2, 3]


def test_create_sequences_can_align_target_to_next_timestep() -> None:
    frame = nasa.add_rul_column(make_cmapss_like_frame())

    artifact = nasa.create_sequences(frame, sequence_length=2, target_position="next")

    assert artifact.features.shape == (3, 2, len(nasa.FEATURE_COLUMNS))
    assert artifact.targets.tolist() == [1.0, 0.0, 0.0]
    assert artifact.engine_ids.tolist() == [1, 1, 2]


def test_create_sequences_is_reproducible() -> None:
    frame = nasa.add_rul_column(make_cmapss_like_frame())

    artifact_a = nasa.create_sequences(frame, sequence_length=3, target_position="last")
    artifact_b = nasa.create_sequences(frame, sequence_length=3, target_position="last")

    assert np.array_equal(artifact_a.features, artifact_b.features)
    assert np.array_equal(artifact_a.targets, artifact_b.targets)
    assert np.array_equal(artifact_a.engine_ids, artifact_b.engine_ids)
    assert np.array_equal(artifact_a.end_cycles, artifact_b.end_cycles)


def test_run_preprocessing_writes_expected_artifacts(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_frame = make_cmapss_like_frame()
    test_frame = make_cmapss_like_frame().loc[lambda df: df[nasa.ENGINE_ID_COLUMN] == 1].copy()
    rul_frame = pd.DataFrame({nasa.TARGET_COLUMN: [10]})

    train_path = raw_dir / "train_FD001.txt"
    test_path = raw_dir / "test_FD001.txt"
    rul_path = raw_dir / "RUL_FD001.txt"

    train_frame.to_csv(train_path, sep=" ", header=False, index=False)
    test_frame.to_csv(test_path, sep=" ", header=False, index=False)
    rul_frame.to_csv(rul_path, header=False, index=False)

    monkeypatch.setattr(nasa, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(nasa, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(nasa, "REPORTS_DIR", reports_dir)

    result = nasa.run_preprocessing(subset="FD001", split="train", sequence_length=2)

    processed_dataset_path = Path(result["processed_dataset_path"])
    sequence_path = Path(result["sequence_path"])
    sequence_metadata_path = Path(result["sequence_metadata_path"])
    feature_metadata_path = Path(result["feature_metadata_path"])
    preprocessing_summary_path = Path(result["preprocessing_summary_path"])
    dataset_summary_path = Path(result["dataset_summary_path"])

    assert processed_dataset_path.exists()
    assert sequence_path.exists()
    assert sequence_metadata_path.exists()
    assert feature_metadata_path.exists()
    assert preprocessing_summary_path.exists()
    assert dataset_summary_path.exists()

    processed_frame = pd.read_csv(processed_dataset_path)
    assert nasa.TARGET_COLUMN in processed_frame.columns
    assert processed_frame[nasa.TARGET_COLUMN].tolist() == [3, 2, 1, 0, 2, 1, 0]

    npz_payload = np.load(sequence_path)
    assert npz_payload["X"].shape == (5, 2, len(nasa.FEATURE_COLUMNS))
    assert npz_payload["y"].tolist() == [2.0, 1.0, 0.0, 1.0, 0.0]

    sequence_metadata = json.loads(sequence_metadata_path.read_text(encoding="utf-8"))
    assert sequence_metadata["sequence_length"] == 2
    assert sequence_metadata["num_sequences"] == 5

    feature_metadata = json.loads(feature_metadata_path.read_text(encoding="utf-8"))
    assert feature_metadata["normalization_applied"] is False

    summary_text = preprocessing_summary_path.read_text(encoding="utf-8")
    assert "NASA CMAPSS preprocessing summary" in summary_text
    assert "No normalization or scaling was applied" in summary_text
