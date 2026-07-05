import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predictive_maintenance.data.datasets import DATASETS, get_dataset_spec


def test_dataset_specs_are_registered() -> None:
    assert {"ai4i2020", "c_mapss", "secom"}.issubset(DATASETS.keys())


def test_get_dataset_spec_returns_expected_dataset() -> None:
    dataset = get_dataset_spec("ai4i2020")
    assert dataset.name == "AI4I 2020"
    assert dataset.primary_task == "classification"
