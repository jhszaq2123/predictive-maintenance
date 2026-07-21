from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_artifacts_exist() -> None:
    required_paths = [
        PROJECT_ROOT / "dashboard" / "app.py",
        PROJECT_ROOT / "dashboard" / "pages" / "1_Project_Overview.py",
        PROJECT_ROOT / "dashboard" / "pages" / "2_Machine_View.py",
        PROJECT_ROOT / "dashboard" / "pages" / "3_Model_Results.py",
        PROJECT_ROOT / "dashboard" / "pages" / "4_Predictions.py",
        PROJECT_ROOT / "dashboard" / "assets" / "machine_diagram.svg",
        PROJECT_ROOT / "dashboard" / "assets" / "industrial_drive",
        PROJECT_ROOT / "dashboard" / "assets" / "centrifugal_pump",
        PROJECT_ROOT / "dashboard" / "assets" / "nasa_turbofan",
        PROJECT_ROOT / "dashboard" / "assets" / "icons",
        PROJECT_ROOT / "dashboard" / "assets" / "shared",
        PROJECT_ROOT / "reports" / "metrics" / "lstm_fd001_test_metrics.json",
        PROJECT_ROOT / "reports" / "dashboard" / "fd001_predictions.csv",
        PROJECT_ROOT / "reports" / "figures" / "lstm_training_loss.png",
        PROJECT_ROOT / "reports" / "figures" / "lstm_test_predictions.png",
    ]
    for path in required_paths:
        assert path.exists(), f"Missing required dashboard artifact: {path}"


def test_dashboard_app_starts() -> None:
    at = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"))
    at.run(timeout=30)
    assert not at.exception
    assert any("Industrial Predictive Maintenance System" in element.value for element in at.markdown)


def test_dashboard_pages_load() -> None:
    page_paths = [
        PROJECT_ROOT / "dashboard" / "pages" / "1_Project_Overview.py",
        PROJECT_ROOT / "dashboard" / "pages" / "2_Machine_View.py",
        PROJECT_ROOT / "dashboard" / "pages" / "3_Model_Results.py",
        PROJECT_ROOT / "dashboard" / "pages" / "4_Predictions.py",
    ]
    for page_path in page_paths:
        at = AppTest.from_file(str(page_path))
        at.run(timeout=30)
        assert not at.exception, f"Page failed to load: {page_path.name}"


def test_dashboard_predictions_artifact_schema() -> None:
    frame = pd.read_csv(PROJECT_ROOT / "reports" / "dashboard" / "fd001_predictions.csv")
    assert list(frame.columns) == [
        "engine_id",
        "predicted_rul",
        "actual_rul",
        "absolute_error",
        "risk_level",
    ]
    assert not frame.empty
    assert frame["engine_id"].is_monotonic_increasing


def test_dashboard_code_is_presentation_only() -> None:
    dashboard_sources = [
        *sorted((PROJECT_ROOT / "dashboard").glob("*.py")),
        *sorted((PROJECT_ROOT / "dashboard" / "components").glob("*.py")),
        *sorted((PROJECT_ROOT / "dashboard" / "pages").glob("*.py")),
    ]
    combined_source = "\n".join(path.read_text(encoding="utf-8") for path in dashboard_sources)

    forbidden_fragments = [
        "tensorflow",
        "keras",
        "model.predict",
        "load_trained_model",
        "build_last_window_test_sequences",
        "load_official_test_split",
        "preprocess_cmapss_frame",
        "pickle.load(",
        "joblib.load(",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in combined_source


def test_machine_mode_configuration_is_complete() -> None:
    from dashboard.machine_config import MACHINE_MODES

    assert set(MACHINE_MODES) == {"industrial_drive", "centrifugal_pump", "nasa_turbofan"}
    for mode_config in MACHINE_MODES.values():
        for field in [
            "display_name",
            "description",
            "component_names",
            "asset_path",
            "dataset_context",
            "disclaimer",
        ]:
            assert field in mode_config
        assert mode_config["component_names"]


def test_data_loaders_return_expected_artifacts() -> None:
    from dashboard.components.data_loaders import load_fd001_prediction_frame, load_metric_artifacts

    prediction_frame = load_fd001_prediction_frame()
    assert list(prediction_frame.columns) == [
        "engine_id",
        "predicted_rul",
        "actual_rul",
        "absolute_error",
        "risk_level",
    ]

    artifacts = load_metric_artifacts()
    assert {"random_forest", "xgboost", "lstm_validation", "lstm_test", "comparison"} == set(artifacts)


def test_dashboard_pages_can_be_imported() -> None:
    page_paths = [
        PROJECT_ROOT / "dashboard" / "app.py",
        PROJECT_ROOT / "dashboard" / "pages" / "1_Project_Overview.py",
        PROJECT_ROOT / "dashboard" / "pages" / "2_Machine_View.py",
        PROJECT_ROOT / "dashboard" / "pages" / "3_Model_Results.py",
        PROJECT_ROOT / "dashboard" / "pages" / "4_Predictions.py",
    ]
    for page_path in page_paths:
        spec = importlib.util.spec_from_file_location(page_path.stem, page_path)
        assert spec is not None and spec.loader is not None
