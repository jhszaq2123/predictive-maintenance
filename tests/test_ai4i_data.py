import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predictive_maintenance.data.ai4i import build_ai4i_feature_frame, load_ai4i_dataset
from predictive_maintenance.data.loaders import dataset_file_path


DATASET_PATH = dataset_file_path("ai4i2020", "ai4i2020.csv")

pytestmark = pytest.mark.skipif(
    not DATASET_PATH.exists(),
    reason="AI4I raw dataset is not available locally.",
)


def test_ai4i_feature_frame_has_expected_target_shape() -> None:
    df = load_ai4i_dataset()
    X, y = build_ai4i_feature_frame(df)

    assert len(X) == len(y) == len(df)
    assert "Machine failure" not in X.columns
    assert "Type" in X.columns
