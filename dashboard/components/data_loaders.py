from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
REPORTS_DASHBOARD_DIR = PROJECT_ROOT / "reports" / "dashboard"
FD001_PREDICTIONS_PATH = REPORTS_DASHBOARD_DIR / "fd001_predictions.csv"

REQUIRED_FD001_COLUMNS = [
    "engine_id",
    "predicted_rul",
    "actual_rul",
    "absolute_error",
    "risk_level",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_required_columns(frame: pd.DataFrame, required_columns: list[str], artifact_name: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{artifact_name} is missing required columns: {missing}")


@st.cache_data(show_spinner=False)
def load_metric_artifacts() -> dict[str, Any]:
    return {
        "random_forest": read_json(REPORTS_METRICS_DIR / "random_forest_metrics.json"),
        "xgboost": read_json(REPORTS_METRICS_DIR / "xgboost_metrics.json"),
        "lstm_validation": read_json(REPORTS_METRICS_DIR / "lstm_fd001_metrics.json"),
        "lstm_test": read_json(REPORTS_METRICS_DIR / "lstm_fd001_test_metrics.json"),
        "comparison": read_csv(REPORTS_METRICS_DIR / "model_comparison.csv"),
    }


@st.cache_data(show_spinner=False)
def load_fd001_prediction_frame() -> pd.DataFrame:
    frame = read_csv(FD001_PREDICTIONS_PATH)
    validate_required_columns(
        frame,
        REQUIRED_FD001_COLUMNS,
        artifact_name="reports/dashboard/fd001_predictions.csv",
    )
    frame["engine_id"] = frame["engine_id"].astype(int)
    frame["predicted_rul"] = frame["predicted_rul"].astype(float)
    frame["actual_rul"] = frame["actual_rul"].astype(float)
    frame["absolute_error"] = frame["absolute_error"].astype(float)
    frame["risk_level"] = frame["risk_level"].astype(str)
    return frame.sort_values("engine_id").reset_index(drop=True)
