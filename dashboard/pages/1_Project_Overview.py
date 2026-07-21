from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.charts import build_pipeline_figure
from dashboard.components.layout import render_panel, render_section_title, render_system_header
from dashboard.components.navigation import render_sidebar
from dashboard.components.styles import inject_global_styles
from dashboard.shared import PROJECT_TITLE


st.set_page_config(page_title="Project Overview", layout="wide")
inject_global_styles()
render_sidebar("Project Overview")
render_system_header()

st.title("Project Overview")
st.caption(PROJECT_TITLE)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    render_panel(
        "Supported dataset contexts",
        "The current demonstrator covers two industrial data settings: AI4I 2020 for tabular "
        "machine-failure classification and NASA CMAPSS FD001 for sequence-based Remaining Useful Life prediction.",
    )
    st.markdown("- `Random Forest`\n- `XGBoost`\n- `LSTM`")

with col2:
    render_panel(
        "Presentation scope",
        "The dashboard remains a presentation-only layer. It does not retrain models, regenerate artifacts, "
        "or infer new predictions at runtime. NASA FD001 values are displayed as whole-engine indicators.",
    )

render_section_title("Pipeline Diagram")
st.plotly_chart(build_pipeline_figure(), use_container_width=True)
