from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.charts import build_gauge_chart
from dashboard.components.data_loaders import load_fd001_prediction_frame
from dashboard.components.layout import render_section_title, render_system_header
from dashboard.components.navigation import render_sidebar
from dashboard.components.styles import inject_global_styles


st.set_page_config(page_title="Predictions", layout="wide")
inject_global_styles()
render_sidebar("Predictions")
render_system_header()

st.title("Predictions")
st.caption("Official NASA CMAPSS FD001 test-set predictions using the accepted LSTM baseline.")

prediction_frame = load_fd001_prediction_frame()
engine_ids = prediction_frame["engine_id"].tolist()

selected_engine = st.selectbox("Engine ID", engine_ids, index=0)
row = prediction_frame.loc[prediction_frame["engine_id"] == selected_engine].iloc[0]
actual_rul_text = "N/A" if pd.isna(row["actual_rul"]) else f"{row['actual_rul']:.2f}"
absolute_error_text = "N/A" if pd.isna(row["actual_rul"]) else f"{row['absolute_error']:.2f}"

render_section_title("Whole-engine prediction summary")
top = st.columns(5)
top[0].metric("Engine ID", int(row["engine_id"]))
top[1].metric("Predicted RUL", f"{row['predicted_rul']:.2f}")
top[2].metric("Actual RUL", actual_rul_text)
top[3].metric("Absolute Error", absolute_error_text)
top[4].metric("Risk Level", str(row["risk_level"]))

left, right = st.columns([1, 1], gap="large")
with left:
    render_section_title("RUL Gauge")
    st.plotly_chart(build_gauge_chart(float(row["predicted_rul"])), use_container_width=True)
with right:
    render_section_title("Saved prediction table")
    st.dataframe(
        prediction_frame[["engine_id", "predicted_rul", "actual_rul", "absolute_error", "risk_level"]]
        .rename(
            columns={
                "engine_id": "Engine ID",
                "predicted_rul": "Predicted RUL",
                "actual_rul": "Actual RUL",
                "absolute_error": "Absolute Error",
                "risk_level": "Risk Level",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
