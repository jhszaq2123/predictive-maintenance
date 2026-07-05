from pathlib import Path

import pandas as pd

from predictive_maintenance.data.datasets import get_dataset_spec


def load_csv_dataset(dataset_key: str, filename: str, **kwargs) -> pd.DataFrame:
    dataset = get_dataset_spec(dataset_key)
    path = dataset.raw_dir / filename
    return pd.read_csv(path, **kwargs)


def dataset_file_path(dataset_key: str, filename: str) -> Path:
    dataset = get_dataset_spec(dataset_key)
    return dataset.raw_dir / filename
