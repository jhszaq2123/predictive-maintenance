from __future__ import annotations

import streamlit as st

from dashboard.components.data_loaders import load_metric_artifacts


PAGE_LINKS = [
    ("app.py", "Home"),
    ("pages/1_Project_Overview.py", "Project Overview"),
    ("pages/2_Machine_View.py", "Machine View"),
    ("pages/3_Model_Results.py", "Model Results"),
    ("pages/4_Predictions.py", "Predictions"),
]


def render_sidebar(active_label: str) -> None:
    with st.sidebar:
        st.title("Industrial PM System")
        st.caption("Presentation-only SCADA/HMI demonstrator")
        st.markdown("**Navigation**")
        for page_path, label in PAGE_LINKS:
            try:
                st.page_link(page_path, label=label, disabled=(label == active_label))
            except Exception:
                current_suffix = " (current)" if label == active_label else ""
                st.markdown(f"- {label}{current_suffix}")

        st.divider()
        st.markdown("**Global status**")
        render_sidebar_metrics()


def render_sidebar_metrics() -> None:
    metrics = load_metric_artifacts()
    lstm_test = metrics["lstm_test"]
    st.metric("LSTM Test MAE", f"{lstm_test['mae']:.2f}")
    st.metric("LSTM Test RMSE", f"{lstm_test['rmse']:.2f}")
