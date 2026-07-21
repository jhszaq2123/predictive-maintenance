from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.charts import (
    build_prediction_summary_chart,
    build_rul_profile_chart,
)
from dashboard.components.data_loaders import load_fd001_prediction_frame
from dashboard.components.layout import render_disclaimer, render_html_block, render_machine_selector
from dashboard.components.machine_view import (
    COMPONENT_ORDER,
    alert_message_for_risk,
    build_interactive_machine_figure,
    machine_status_for_risk,
    presentation_risk_level,
    risk_tone_for_label,
    selected_component_from_event,
)
from dashboard.components.styles import inject_global_styles
from dashboard.machine_config import MACHINE_MODES


def component_description_for_mode(mode_config: dict[str, object], selected_component: str) -> str:
    descriptions = mode_config.get("component_descriptions", {})
    if isinstance(descriptions, dict):
        description = descriptions.get(selected_component)
        if isinstance(description, str):
            return description
    return "Illustrative machine component for presentation mode."


def reset_component_selection_for_mode(selected_mode: str) -> str:
    return str(MACHINE_MODES[selected_mode]["component_names"][0])


def component_status_label(selected_component: str, risk_label: str) -> str:
    if selected_component in {"Gearbox", "Pump"} and risk_label != "LOW":
        return "Warning"
    if risk_label == "HIGH":
        return "Watch"
    return "Nominal"


def build_prediction_table(prediction_frame: pd.DataFrame, selected_engine: int) -> pd.DataFrame:
    table = prediction_frame.copy()
    table["scope"] = table["engine_id"].apply(
        lambda engine_id: "Selected" if int(engine_id) == int(selected_engine) else "Reference"
    )
    return table[["scope", "engine_id", "predicted_rul", "actual_rul", "absolute_error", "risk_level"]]


def build_table_preview(prediction_frame: pd.DataFrame, selected_engine: int) -> pd.DataFrame:
    selected_index = prediction_frame.index[prediction_frame["engine_id"] == selected_engine][0]
    start_index = max(0, int(selected_index) - 2)
    end_index = min(len(prediction_frame), int(selected_index) + 3)
    return build_prediction_table(prediction_frame.iloc[start_index:end_index].copy(), selected_engine)


def render_header_tile(label: str, value: str, tone: str = "neutral") -> None:
    render_html_block(
        f"""
        <section class="machine-header-tile {tone}">
            <div class="machine-header-label">{label}</div>
            <div class="machine-header-value">{value}</div>
        </section>
        """
    )


st.set_page_config(page_title="Machine View", layout="wide")
inject_global_styles()
st.markdown(
    """
    <style>
    .stApp:has(.machine-view-layout) [data-testid="stMainBlockContainer"] {
        padding-top: 0.75rem !important;
        padding-right: 1rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1rem !important;
    }

    .stApp:has(.machine-view-layout) [data-testid="stSidebarContent"] {
        padding-top: 0.6rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 0.7rem !important;
        padding-right: 0.7rem !important;
    }

    .stApp:has(.machine-view-layout) [data-testid="stSidebarHeader"] {
        min-height: 1.85rem !important;
        height: 1.85rem !important;
        padding-top: 0.15rem !important;
        padding-bottom: 0.05rem !important;
        padding-left: 0.25rem !important;
        padding-right: 0.25rem !important;
        margin-bottom: 0 !important;
    }

    .stApp:has(.machine-view-layout) [data-testid="stSidebarCollapseButton"] {
        margin-left: auto !important;
        margin-right: 0 !important;
    }

    .stApp:has(.machine-view-layout) [data-testid="stSidebarUserContent"] {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    .stApp:has(.machine-view-layout) [data-testid="stSidebarNav"] {
        margin-top: 0 !important;
        margin-bottom: 0.45rem !important;
    }

    .stApp:has(.machine-view-layout) [data-testid="stSidebarUserContent"] h1 {
        margin-top: 0.15rem !important;
        margin-bottom: 0.2rem !important;
        font-size: 1.18rem !important;
        line-height: 1.15 !important;
    }

    .stApp:has(.machine-view-layout) [data-testid="stSidebarUserContent"] p {
        margin-top: 0.1rem !important;
        margin-bottom: 0.28rem !important;
        font-size: 0.74rem !important;
        line-height: 1.25 !important;
    }

    .stApp:has(.machine-view-layout) [data-testid="stSidebarUserContent"] .stMarkdown p strong {
        font-size: 0.84rem !important;
    }

    .stApp:has(.machine-view-layout) [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] {
        gap: 0.32rem !important;
    }

    .stApp:has(.machine-view-layout) [data-testid="stSidebarUserContent"] .element-container {
        margin-bottom: 0.12rem !important;
    }

    .stApp:has(.machine-view-layout) [data-testid="stSidebarUserContent"] label[data-testid="stWidgetLabel"] p {
        font-size: 0.72rem !important;
        margin-bottom: 0.1rem !important;
    }

    .stApp:has(.machine-view-layout) [data-testid="stSidebarUserContent"] [data-baseweb="select"] > div,
    .stApp:has(.machine-view-layout) [data-testid="stSidebarUserContent"] [data-baseweb="select"] input {
        min-height: 2rem !important;
        font-size: 0.78rem !important;
    }

    .stApp:has(.machine-view-layout) [data-testid="stSidebarUserContent"] [data-baseweb="slider"] {
        padding-top: 0.02rem !important;
        padding-bottom: 0 !important;
    }

    .stApp:has(.machine-view-layout) [data-testid="stSidebarUserContent"] .stButton button {
        min-height: 1.75rem !important;
        padding-top: 0.08rem !important;
        padding-bottom: 0.08rem !important;
        font-size: 0.76rem !important;
    }

    .stApp:has(.machine-view-layout) [data-testid="stSidebarUserContent"] [data-testid="column"] {
        gap: 0.28rem !important;
    }

    .stApp:has(.machine-view-layout) .sidebar-cards-start {
        margin-top: 0.9rem !important;
    }

    .stApp:has(.machine-view-layout) .sidebar-description {
        margin-top: 0.12rem !important;
        margin-bottom: 0.5rem !important;
        font-size: 0.74rem !important;
        line-height: 1.28 !important;
        color: var(--scada-muted) !important;
    }

    .stApp:has(.machine-view-layout) .sidebar-section-title {
        margin-top: 0.08rem !important;
        margin-bottom: 0.2rem !important;
        color: var(--scada-text) !important;
        font-size: 0.84rem !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
    }

    .stApp:has(.machine-view-layout) .sidebar-mode-block {
        margin-bottom: 0.4rem !important;
    }

    .stApp:has(.machine-view-layout) .sidebar-engine-block {
        margin-top: 0.08rem !important;
    }

    .stApp:has(.machine-view-layout) .sidebar-engine-row {
        margin-top: 0.02rem !important;
        margin-bottom: 0.12rem !important;
    }

    .stApp:has(.machine-view-layout) .scada-sidebar-engine-label {
        padding-top: 0.2rem;
        padding-bottom: 0.02rem;
        font-size: 0.78rem;
        line-height: 1.1;
        white-space: nowrap;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

prediction_frame = load_fd001_prediction_frame()
engine_ids = [int(engine_id) for engine_id in prediction_frame["engine_id"].tolist()]
engine_id_to_index = {engine_id: index for index, engine_id in enumerate(engine_ids)}

with st.sidebar:
    st.markdown('<div class="machine-sidebar-wrap">', unsafe_allow_html=True)
    st.title("Industrial PM System")
    st.markdown('<div class="sidebar-description">Machine View | presentation-only</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Machine View</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-mode-block">', unsafe_allow_html=True)
    selected_mode = render_machine_selector(
        "Machine Mode",
        list(MACHINE_MODES.keys()),
        index=0,
        key="machine_mode_sidebar",
    )
    mode_config = MACHINE_MODES[selected_mode]
    if st.session_state.get("machine_view_mode") != selected_mode:
        st.session_state["machine_view_mode"] = selected_mode
        st.session_state["machine_view_component"] = reset_component_selection_for_mode(selected_mode)
    st.markdown("</div>", unsafe_allow_html=True)

    if "engine_slider_index" not in st.session_state:
        st.session_state["engine_slider_index"] = 0
    if "engine_slider_value" not in st.session_state:
        st.session_state["engine_slider_value"] = engine_ids[0]

    current_engine_from_state = int(st.session_state.get("engine_slider_value", engine_ids[0]))
    if current_engine_from_state in engine_id_to_index:
        st.session_state["engine_slider_index"] = engine_id_to_index[current_engine_from_state]

    st.markdown('<div class="sidebar-section-title sidebar-engine-block">Engine</div>', unsafe_allow_html=True)
    current_engine = int(st.session_state.get("engine_slider_value", engine_ids[0]))
    current_index = engine_id_to_index.get(current_engine, 0)
    st.markdown('<div class="sidebar-engine-row">', unsafe_allow_html=True)
    engine_prev, engine_label, engine_next = st.columns([0.95, 1.45, 0.95], gap="small")
    with engine_prev:
        if st.button("◀", use_container_width=True, key="engine_prev_button") and current_index > 0:
            st.session_state["engine_slider_value"] = engine_ids[current_index - 1]
            st.session_state["engine_slider_index"] = current_index - 1
            st.rerun()
    with engine_label:
        st.markdown(
            f'<div class="scada-sidebar-engine-label">Engine {int(st.session_state["engine_slider_value"])}</div>',
            unsafe_allow_html=True,
        )
    with engine_next:
        if st.button("▶", use_container_width=True, key="engine_next_button") and current_index < len(engine_ids) - 1:
            st.session_state["engine_slider_value"] = engine_ids[current_index + 1]
            st.session_state["engine_slider_index"] = current_index + 1
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    selected_engine_index = st.slider(
        "Engine slider",
        min_value=0,
        max_value=len(engine_ids) - 1,
        key="engine_slider_index",
        label_visibility="collapsed",
    )
    selected_engine = engine_ids[int(selected_engine_index)]
    st.session_state["engine_slider_value"] = selected_engine

    st.markdown('<div class="sidebar-cards-start"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-info-card">
            <div class="sidebar-info-title">Dataset</div>
            <div><span>Dataset</span><strong>NASA CMAPSS FD001</strong></div>
            <div><span>Rows</span><strong>100 prediction records</strong></div>
            <div><span>Task</span><strong>Whole-engine RUL</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-info-card">
            <div class="sidebar-info-title">System</div>
            <div><span>Architecture</span><strong>Presentation-only</strong></div>
            <div><span>Inference</span><strong>Offline artifacts</strong></div>
            <div><span>Interaction</span><strong>Clickable SVG enabled</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

selected_component = st.session_state.get(
    "machine_view_component",
    reset_component_selection_for_mode(selected_mode),
)

engine_row = prediction_frame.loc[prediction_frame["engine_id"] == selected_engine].iloc[0]
risk_label = presentation_risk_level(str(engine_row["risk_level"]))
risk_tone = risk_tone_for_label(risk_label)
alert_message = alert_message_for_risk(str(engine_row["risk_level"]))
status_label = machine_status_for_risk(risk_label)
component_description = component_description_for_mode(mode_config, selected_component)
component_status = component_status_label(selected_component, risk_label)

render_html_block('<div class="machine-view-layout">')

header_name, header_rul, header_actual, header_error, header_risk = st.columns(
    [2.25, 1, 1, 1, 1],
    gap="small",
)
with header_name:
    render_html_block(
        f"""
        <section class="machine-title-shell">
            <div class="machine-title-path">Machines / {mode_config['display_name']}</div>
            <div class="machine-title-row">
                <h2>{mode_config['display_name']}</h2>
                <span class="machine-inline-badge {risk_tone}">{status_label}</span>
            </div>
        </section>
        """
    )
with header_rul:
    render_header_tile("Predicted RUL", f"{engine_row['predicted_rul']:.0f}", "neutral")
with header_actual:
    render_header_tile("Actual RUL", f"{engine_row['actual_rul']:.0f}", "low")
with header_error:
    render_header_tile("Absolute Error", f"{engine_row['absolute_error']:.1f}", "warning")
with header_risk:
    render_header_tile("Risk Level", risk_label.title(), risk_tone)

main_left, main_right = st.columns([3.25, 1.65], gap="medium")

with main_left:
    render_html_block(
        """
        <section class="machine-stage-panel">
            <div class="machine-stage-toolbar">
                <span class="machine-inline-badge neutral">Interactive machine canvas</span>
                <span class="machine-inline-hint">Click a component to view details</span>
            </div>
        </section>
        """
    )
    if selected_mode == "industrial_drive":
        selection_event = st.plotly_chart(
            build_interactive_machine_figure(selected_component, float(engine_row["predicted_rul"])),
            use_container_width=True,
            config={"displayModeBar": False, "scrollZoom": False},
            key="industrial_drive_hmi_view",
            on_select="rerun",
            selection_mode=("points",),
        )
        clicked_component = selected_component_from_event(selection_event)
        if clicked_component and clicked_component in COMPONENT_ORDER:
            st.session_state["machine_view_component"] = clicked_component
            selected_component = clicked_component
            component_description = component_description_for_mode(mode_config, selected_component)
            component_status = component_status_label(selected_component, risk_label)
    else:
        fallback_component = st.segmented_control(
            "Component focus",
            mode_config["component_names"],
            key="placeholder_component_selector",
            default=mode_config["component_names"][0],
        )
        if fallback_component:
            selected_component = str(fallback_component)
            st.session_state["machine_view_component"] = selected_component
            component_description = component_description_for_mode(mode_config, selected_component)
            component_status = component_status_label(selected_component, risk_label)
        st.info("Interactive component clicking is currently available for the Industrial Drive System view.")

    bottom_left, bottom_mid, bottom_table = st.columns([1.2, 1.2, 1.55], gap="medium")
    with bottom_left:
        render_html_block('<div class="machine-panel-title compact">Prediction Chart</div>')
        summary_fig = build_prediction_summary_chart(engine_row)
        summary_fig.update_layout(height=170, margin=dict(l=8, r=8, t=12, b=8), showlegend=False)
        st.plotly_chart(summary_fig, use_container_width=True, config={"displayModeBar": False})
    with bottom_mid:
        render_html_block('<div class="machine-panel-title compact">RUL Comparison</div>')
        rul_fig = build_rul_profile_chart(prediction_frame, int(selected_engine))
        rul_fig.update_layout(
            height=170,
            margin=dict(l=8, r=8, t=12, b=8),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=9)),
        )
        st.plotly_chart(rul_fig, use_container_width=True, config={"displayModeBar": False})
    with bottom_table:
        render_html_block('<div class="machine-panel-title compact">Prediction Table</div>')
        st.dataframe(
            build_table_preview(prediction_frame, int(selected_engine)).rename(
                columns={
                    "scope": "Scope",
                    "engine_id": "Engine",
                    "predicted_rul": "Pred.",
                    "actual_rul": "Actual",
                    "absolute_error": "Error",
                    "risk_level": "Risk",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=170,
        )

with main_right:
    render_html_block(
        f"""
        <section class="machine-side-panel">
            <div class="machine-panel-title">Selected Component</div>
            <div class="machine-component-name">{selected_component}</div>
            <div class="machine-component-status">
                <span>Status</span>
                <strong>{component_status}</strong>
            </div>
            <div class="machine-component-copy">{component_description}</div>
        </section>

        <section class="machine-side-panel">
            <div class="machine-panel-title">Prediction Summary</div>
            <div class="machine-kv-list">
                <div><span>Predicted RUL</span><strong>{engine_row['predicted_rul']:.2f}</strong></div>
                <div><span>Actual RUL</span><strong>{engine_row['actual_rul']:.2f}</strong></div>
                <div><span>Absolute Error</span><strong>{engine_row['absolute_error']:.2f}</strong></div>
                <div><span>Risk Level</span><strong>{risk_label}</strong></div>
            </div>
        </section>

        <section class="machine-side-panel">
            <div class="machine-panel-title">Alerts</div>
            <div class="machine-alert-card {risk_tone}">
                <div class="machine-alert-status">{status_label}</div>
                <div class="machine-alert-copy">{alert_message}</div>
            </div>
        </section>

        <section class="machine-side-panel">
            <div class="machine-panel-title">Information</div>
            <div class="machine-kv-list">
                <div><span>Engine ID</span><strong>{int(engine_row['engine_id'])}</strong></div>
                <div><span>Dataset</span><strong>NASA CMAPSS FD001</strong></div>
                <div><span>Model</span><strong>LSTM</strong></div>
                <div><span>Artifact Source</span><strong>reports/dashboard/fd001_predictions.csv</strong></div>
            </div>
            <div class="machine-side-note">
                Component selection is for visualization only. The ML prediction applies to the complete engine.
            </div>
        </section>
        """
    )

render_disclaimer(
    "Visualization is provided for demonstration purposes. Component highlighting does not represent real sensor placement."
)
render_html_block("</div>")
