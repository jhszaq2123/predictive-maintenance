from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.data_loaders import load_metric_artifacts
from dashboard.components.layout import render_section_title, render_system_header
from dashboard.components.navigation import render_sidebar
from dashboard.components.styles import inject_global_styles
from dashboard.shared import REPORTS_FIGURES_DIR


st.set_page_config(page_title="Model Results", layout="wide")
inject_global_styles()
render_sidebar("Model Results")
render_system_header()

st.title("Model Results")
metrics = load_metric_artifacts()

rf = metrics["random_forest"]
xgb = metrics["xgboost"]
lstm_val = metrics["lstm_validation"]
lstm_test = metrics["lstm_test"]
comparison = metrics["comparison"]

render_section_title("Tabular Baselines")
st.dataframe(
    comparison.rename(
        columns={
            "model": "Model",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-score",
            "roc_auc": "ROC-AUC",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

metrics_cols = st.columns(4)
metrics_cols[0].metric("RF Accuracy", f"{rf['accuracy']:.4f}")
metrics_cols[1].metric("XGB Accuracy", f"{xgb['accuracy']:.4f}")
metrics_cols[2].metric("LSTM Validation MAE", f"{lstm_val['mae']:.4f}")
metrics_cols[3].metric("LSTM Test MAE", f"{lstm_test['mae']:.4f}")

render_section_title("Existing Figures")
fig1, fig2 = st.columns(2, gap="large")
with fig1:
    st.image(str(REPORTS_FIGURES_DIR / "model_comparison_metrics.png"), caption="RF vs XGBoost comparison")
    st.image(str(REPORTS_FIGURES_DIR / "lstm_training_loss.png"), caption="LSTM training loss")
with fig2:
    st.image(str(REPORTS_FIGURES_DIR / "lstm_test_predictions.png"), caption="Predicted vs Actual RUL")
    st.image(
        str(REPORTS_FIGURES_DIR / "lstm_test_error_distribution.png"),
        caption="Prediction error distribution",
    )
