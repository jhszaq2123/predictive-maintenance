from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.charts import build_pipeline_figure
from dashboard.components.layout import render_bullet_overview, render_panel, render_system_header
from dashboard.components.navigation import render_sidebar
from dashboard.components.styles import inject_global_styles
from dashboard.shared import PROJECT_TITLE


st.set_page_config(
    page_title="Industrial Predictive Maintenance System",
    page_icon="dashboard/assets/machine_diagram.svg",
    layout="wide",
)

inject_global_styles()
render_sidebar("Home")
render_system_header()

st.subheader(PROJECT_TITLE)

left, right = st.columns([1.2, 1], gap="large")
with left:
    render_panel(
        "Dashboard scope",
        "This interface presents accepted artifacts from the Predictive Maintenance project. "
        "It visualizes saved results for AI4I and NASA CMAPSS FD001 without retraining models, "
        "without running inference, and without regenerating preprocessing outputs.",
    )
    render_bullet_overview(
        [
            "`Project Overview`: datasets, implemented models, and the end-to-end pipeline.",
            "`Machine View`: a machine-mode selector with an illustrative whole-unit status view.",
            "`Model Results`: existing metrics and saved result figures from validated workflows.",
            "`Predictions`: engine-level FD001 test-set predictions loaded from a saved CSV artifact.",
        ]
    )

with right:
    st.plotly_chart(build_pipeline_figure(), use_container_width=True)

st.info(
    "Presentation Mode: all values shown in this dashboard come from saved artifacts in reports/. "
    "No ML model is loaded and no runtime inference is executed."
)
