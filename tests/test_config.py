import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predictive_maintenance.config import DATA_DIR, MODELS_DIR, PROJECT_ROOT


def test_project_paths_are_relative_to_repo_root() -> None:
    assert PROJECT_ROOT.name == "Predictive Maintenance"
    assert DATA_DIR.parent == PROJECT_ROOT
    assert MODELS_DIR.parent == PROJECT_ROOT
