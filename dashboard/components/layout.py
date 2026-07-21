from __future__ import annotations

from typing import Iterable

import streamlit as st


SYSTEM_NAME = "Industrial Predictive Maintenance System"
SYSTEM_SUBTITLE = "AI-powered Condition Monitoring Platform"


def render_system_header(
    system_mode: str = "Presentation Mode",
    data_source_note: str = "Data source: saved artifacts from reports/",
    compact: bool = False,
) -> None:
    shell_class = "scada-shell scada-shell-compact" if compact else "scada-shell"
    st.markdown(
        f"""
        <section class="{shell_class}">
            <div class="scada-header">
                <div>
                    <div class="scada-kicker">Industrial Predictive Maintenance</div>
                    <h1 class="scada-title">{SYSTEM_NAME}</h1>
                    <p class="scada-subtitle">{SYSTEM_SUBTITLE}</p>
                </div>
                <div class="scada-grid">
                    <span class="scada-badge neutral">System mode: {system_mode}</span>
                    <span class="scada-badge neutral">{data_source_note}</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(label: str, tone: str = "neutral") -> None:
    st.markdown(f'<span class="scada-badge {tone}">{label}</span>', unsafe_allow_html=True)


def render_metric_card(label: str, value: str, note: str | None = None) -> None:
    note_markup = f'<div class="scada-metric-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="scada-metric-card">
            <div class="scada-metric-label">{label}</div>
            <div class="scada-metric-value">{value}</div>
            {note_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel(title: str, body: str) -> None:
    st.markdown(
        f"""
        <section class="scada-panel">
            <h3>{title}</h3>
            <p>{body}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer(text: str) -> None:
    st.markdown(f'<div class="scada-disclaimer">{text}</div>', unsafe_allow_html=True)


def render_section_title(title: str) -> None:
    st.markdown(f'<div class="scada-section-title">{title}</div>', unsafe_allow_html=True)


def render_bullet_overview(lines: Iterable[str]) -> None:
    formatted = "\n".join(f"- {line}" for line in lines)
    st.markdown(formatted)


def render_machine_selector(label: str, options: list[str], index: int = 0, key: str | None = None) -> str:
    return st.selectbox(label, options, index=index, key=key)


def render_html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)
