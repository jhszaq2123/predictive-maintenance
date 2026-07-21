from __future__ import annotations

import streamlit as st


SCADA_THEME = {
    "bg": "#0F1720",
    "panel": "#17232D",
    "panel_alt": "#1D2B36",
    "border": "#355061",
    "text": "#E8EEF2",
    "muted": "#9DB0BD",
    "low": "#2F9E73",
    "warning": "#D9A441",
    "critical": "#D94F4F",
    "neutral": "#4D8CA8",
}


def inject_global_styles() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --scada-bg: {SCADA_THEME["bg"]};
            --scada-panel: {SCADA_THEME["panel"]};
            --scada-panel-alt: {SCADA_THEME["panel_alt"]};
            --scada-border: {SCADA_THEME["border"]};
            --scada-text: {SCADA_THEME["text"]};
            --scada-muted: {SCADA_THEME["muted"]};
            --scada-low: {SCADA_THEME["low"]};
            --scada-warning: {SCADA_THEME["warning"]};
            --scada-critical: {SCADA_THEME["critical"]};
            --scada-neutral: {SCADA_THEME["neutral"]};
        }}

        .stApp {{
            background:
                radial-gradient(circle at top right, rgba(77, 140, 168, 0.12), transparent 28%),
                linear-gradient(180deg, #0D141B 0%, var(--scada-bg) 100%);
            color: var(--scada-text);
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #101922 0%, #152330 100%);
            border-right: 1px solid rgba(53, 80, 97, 0.8);
        }}

        section[data-testid="stSidebar"][aria-expanded="true"] {{
            min-width: clamp(180px, 12vw, 205px) !important;
            max-width: clamp(180px, 12vw, 205px) !important;
        }}

        section[data-testid="stSidebar"][aria-expanded="false"] {{
            min-width: 0 !important;
            max-width: 0 !important;
        }}

        [data-testid="stSidebar"] .stButton button {{
            min-height: 1.9rem;
            padding-top: 0.16rem;
            padding-bottom: 0.16rem;
        }}

        .scada-sidebar-engine-label {{
            text-align: center;
            color: var(--scada-text);
            font-size: 0.8rem;
            font-weight: 700;
            padding-top: 0.35rem;
        }}

        .scada-shell {{
            background: rgba(23, 35, 45, 0.82);
            border: 1px solid rgba(53, 80, 97, 0.9);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 16px 42px rgba(3, 10, 16, 0.28);
            margin-bottom: 1rem;
        }}

        .scada-shell.scada-shell-compact {{
            padding: 0.78rem 0.95rem;
            margin-bottom: 0.65rem;
        }}

        .scada-header {{
            display: flex;
            flex-wrap: wrap;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
        }}

        .scada-kicker {{
            font-size: 0.85rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--scada-neutral);
            margin-bottom: 0.35rem;
        }}

        .scada-title {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--scada-text);
            margin: 0;
        }}

        .scada-subtitle {{
            font-size: 1rem;
            color: var(--scada-muted);
            margin-top: 0.35rem;
        }}

        .scada-grid {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}

        .scada-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.4rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            color: var(--scada-text);
            font-size: 0.82rem;
        }}

        .scada-badge.low {{ border-color: rgba(47, 158, 115, 0.6); color: var(--scada-low); }}
        .scada-badge.warning {{ border-color: rgba(217, 164, 65, 0.6); color: var(--scada-warning); }}
        .scada-badge.critical {{ border-color: rgba(217, 79, 79, 0.7); color: var(--scada-critical); }}
        .scada-badge.neutral {{ border-color: rgba(77, 140, 168, 0.7); color: var(--scada-neutral); }}

        .scada-panel {{
            background: rgba(23, 35, 45, 0.72);
            border: 1px solid rgba(53, 80, 97, 0.78);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin-bottom: 1rem;
        }}

        .scada-operator-grid {{
            display: grid;
            grid-template-columns: minmax(0, 2.2fr) minmax(320px, 1fr);
            gap: 1rem;
            margin-bottom: 1rem;
        }}

        .scada-stack {{
            display: grid;
            gap: 1rem;
        }}

        .scada-visual-shell {{
            background:
                linear-gradient(180deg, rgba(18, 30, 39, 0.95), rgba(13, 22, 29, 0.98));
            border: 1px solid rgba(66, 96, 114, 0.84);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 18px 36px rgba(3, 10, 16, 0.26);
        }}

        .scada-visual-meta {{
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: center;
            margin-bottom: 0.75rem;
            flex-wrap: wrap;
        }}

        .scada-meta-block {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(66, 96, 114, 0.58);
            border-radius: 12px;
            padding: 0.7rem 0.85rem;
        }}

        .scada-meta-label {{
            color: var(--scada-muted);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .scada-meta-value {{
            color: var(--scada-text);
            font-size: 1rem;
            font-weight: 600;
            margin-top: 0.2rem;
        }}

        .scada-panel h3, .scada-panel h4, .scada-panel p {{
            margin-top: 0;
        }}

        .scada-machine-hero {{
            min-height: 100%;
        }}

        .scada-machine-breadcrumb {{
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--scada-neutral);
            margin-bottom: 0.6rem;
        }}

        .scada-machine-title {{
            margin: 0;
            font-size: 2rem;
            color: var(--scada-text);
        }}

        .scada-machine-subtitle {{
            color: var(--scada-muted);
            font-size: 1rem;
            margin-top: 0.5rem;
        }}

        .scada-machine-callout {{
            margin-top: 1rem;
            padding: 0.8rem 0.95rem;
            border-radius: 12px;
            border: 1px solid rgba(77, 140, 168, 0.45);
            background: rgba(77, 140, 168, 0.09);
            color: var(--scada-text);
            line-height: 1.45;
        }}

        .machine-view-compact {{
            padding-bottom: 0.2rem;
        }}

        .machine-view-layout {{
            padding-bottom: 0.2rem;
            width: 100%;
        }}

        [data-testid="stAppViewBlockContainer"]:has(.machine-view-layout) {{
            padding-top: 0.55rem;
            padding-bottom: 0.45rem;
        }}

        .machine-view-layout [data-testid="stHorizontalBlock"] {{
            gap: 0.52rem;
        }}

        .machine-view-layout [data-testid="stVerticalBlock"] {{
            gap: 0.22rem;
        }}

        .machine-view-layout .element-container {{
            margin-bottom: 0.08rem;
        }}

        .machine-view-layout .stPlotlyChart,
        .machine-view-layout .js-plotly-plot {{
            margin-bottom: 0 !important;
        }}

        .machine-view-layout .stDataFrame {{
            border-radius: 12px;
            overflow: hidden;
        }}

        .machine-sidebar-wrap {{
            display: grid;
            gap: 0.22rem;
        }}

        .sidebar-info-card {{
            border: 1px solid rgba(53, 80, 97, 0.72);
            border-radius: 12px;
            padding: 0.45rem 0.55rem;
            background: rgba(23, 35, 45, 0.68);
            display: grid;
            gap: 0.24rem;
            margin-bottom: 0.5rem;
        }}

        .sidebar-info-card div {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.06rem;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            padding-bottom: 0.14rem;
        }}

        .sidebar-info-card span {{
            color: var(--scada-muted);
            font-size: 0.76rem;
            line-height: 1.15;
        }}

        .sidebar-info-card strong {{
            color: var(--scada-text);
            font-size: 0.77rem;
            text-align: left;
            line-height: 1.2;
            white-space: normal;
            overflow-wrap: anywhere;
        }}

        .sidebar-info-card .sidebar-info-title {{
            display: block;
            color: var(--scada-neutral);
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.04rem;
        }}

        .machine-title-shell {{
            background: rgba(23, 35, 45, 0.82);
            border: 1px solid rgba(53, 80, 97, 0.84);
            border-radius: 14px;
            padding: 0.62rem 0.78rem;
            min-height: 100%;
        }}

        .machine-title-path {{
            color: var(--scada-neutral);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.28rem;
        }}

        .machine-title-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.55rem;
        }}

        .machine-title-row h2 {{
            margin: 0;
            color: var(--scada-text);
            font-size: 1.42rem;
            line-height: 1.08;
        }}

        .machine-inline-badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.28rem 0.52rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            color: var(--scada-text);
            font-size: 0.74rem;
            white-space: nowrap;
        }}

        .machine-inline-badge.low {{ border-color: rgba(47, 158, 115, 0.6); color: var(--scada-low); }}
        .machine-inline-badge.warning {{ border-color: rgba(217, 164, 65, 0.6); color: var(--scada-warning); }}
        .machine-inline-badge.critical {{ border-color: rgba(217, 79, 79, 0.7); color: var(--scada-critical); }}
        .machine-inline-badge.neutral {{ border-color: rgba(77, 140, 168, 0.7); color: var(--scada-neutral); }}

        .machine-header-tile {{
            background: linear-gradient(180deg, rgba(25, 36, 46, 0.96), rgba(15, 24, 32, 0.96));
            border: 1px solid rgba(53, 80, 97, 0.72);
            border-radius: 14px;
            padding: 0.56rem 0.68rem;
            min-height: 100%;
        }}

        .machine-header-tile.low {{ border-color: rgba(47, 158, 115, 0.54); }}
        .machine-header-tile.warning {{ border-color: rgba(217, 164, 65, 0.54); }}
        .machine-header-tile.critical {{ border-color: rgba(217, 79, 79, 0.58); }}
        .machine-header-tile.neutral {{ border-color: rgba(77, 140, 168, 0.6); }}

        .machine-header-label {{
            color: var(--scada-muted);
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.16rem;
        }}

        .machine-header-value {{
            color: var(--scada-text);
            font-size: 1.08rem;
            font-weight: 700;
            line-height: 1.04;
        }}

        .machine-stage-panel {{
            background:
                linear-gradient(180deg, rgba(18, 30, 39, 0.95), rgba(13, 22, 29, 0.98));
            border: 1px solid rgba(66, 96, 114, 0.84);
            border-radius: 16px;
            padding: 0.54rem 0.74rem;
            margin-bottom: 0.02rem;
        }}

        .machine-stage-toolbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.55rem;
            flex-wrap: wrap;
        }}

        .machine-inline-hint {{
            color: var(--scada-muted);
            font-size: 0.78rem;
        }}

        .machine-side-panel {{
            background: rgba(23, 35, 45, 0.72);
            border: 1px solid rgba(53, 80, 97, 0.78);
            border-radius: 14px;
            padding: 0.62rem 0.72rem;
            margin-bottom: 0.26rem;
        }}

        .machine-panel-title {{
            color: var(--scada-neutral);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.22rem;
        }}

        .machine-panel-title.compact {{
            margin-bottom: 0.22rem;
        }}

        .machine-component-name {{
            color: var(--scada-text);
            font-size: 1rem;
            font-weight: 700;
        }}

        .machine-component-status {{
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
            margin-top: 0.24rem;
            margin-bottom: 0.24rem;
        }}

        .machine-component-status span {{
            color: var(--scada-muted);
            font-size: 0.78rem;
        }}

        .machine-component-status strong {{
            color: var(--scada-text);
            font-size: 0.8rem;
        }}

        .machine-component-copy {{
            color: var(--scada-muted);
            font-size: 0.81rem;
            line-height: 1.34;
        }}

        .machine-kv-list {{
            display: grid;
            gap: 0.2rem;
        }}

        .machine-kv-list div {{
            display: flex;
            justify-content: space-between;
            gap: 0.55rem;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            padding-bottom: 0.18rem;
        }}

        .machine-kv-list span {{
            color: var(--scada-muted);
            font-size: 0.78rem;
        }}

        .machine-kv-list strong {{
            color: var(--scada-text);
            font-size: 0.8rem;
            text-align: right;
        }}

        .machine-alert-card {{
            border-left: 4px solid var(--scada-warning);
            background: linear-gradient(180deg, rgba(29, 43, 54, 0.96), rgba(18, 29, 38, 0.96));
            border-radius: 12px;
            padding: 0.52rem 0.66rem;
        }}

        .machine-alert-card.low {{ border-left-color: var(--scada-low); }}
        .machine-alert-card.warning {{ border-left-color: var(--scada-warning); }}
        .machine-alert-card.critical {{ border-left-color: var(--scada-critical); }}

        .machine-alert-status {{
            color: var(--scada-text);
            font-size: 0.88rem;
            font-weight: 700;
            margin-bottom: 0.14rem;
        }}

        .machine-alert-copy {{
            color: var(--scada-muted);
            font-size: 0.8rem;
        }}

        .machine-side-note {{
            margin-top: 0.38rem;
            padding: 0.42rem 0.5rem;
            border-radius: 10px;
            border: 1px solid rgba(77, 140, 168, 0.42);
            background: rgba(77, 140, 168, 0.08);
            color: var(--scada-text);
            font-size: 0.76rem;
            line-height: 1.28;
        }}

        [data-testid="stAppViewBlockContainer"]:has(.machine-view-compact) {{
            padding-top: 0.55rem;
            padding-bottom: 0.45rem;
        }}

        .machine-view-compact [data-testid="stHorizontalBlock"] {{
            gap: 0.65rem;
        }}

        .machine-view-compact [data-testid="stVerticalBlock"] {{
            gap: 0.35rem;
        }}

        .machine-view-compact .element-container {{
            margin-bottom: 0.12rem;
        }}

        .machine-view-compact .scada-section-title {{
            margin-bottom: 0.35rem;
        }}

        .machine-view-compact .scada-machine-hero.compact {{
            padding-top: 0.8rem;
            padding-bottom: 0.8rem;
            margin-bottom: 0.45rem;
        }}

        .machine-view-compact .scada-machine-title {{
            font-size: 1.5rem;
            line-height: 1.08;
        }}

        .machine-view-compact .scada-machine-subtitle {{
            font-size: 0.88rem;
            margin-top: 0.25rem;
        }}

        .machine-view-compact .scada-machine-callout {{
            margin-top: 0.65rem;
            padding: 0.58rem 0.72rem;
            font-size: 0.82rem;
            line-height: 1.3;
        }}

        .machine-view-compact .scada-overview-card.compact {{
            padding: 0.68rem 0.8rem;
        }}

        .machine-view-compact .scada-overview-label {{
            font-size: 0.72rem;
            margin-bottom: 0.2rem;
        }}

        .machine-view-compact .scada-overview-value {{
            font-size: 1.18rem;
            line-height: 1.04;
        }}

        .machine-view-compact .scada-visual-shell.compact {{
            padding: 0.78rem 0.85rem;
            margin-bottom: 0.32rem;
        }}

        .machine-view-compact .scada-panel.compact {{
            padding: 0.72rem 0.84rem;
            margin-bottom: 0.45rem;
        }}

        .machine-view-compact .scada-right-panel {{
            margin-bottom: 0.45rem;
        }}

        .machine-view-compact .scada-right-title {{
            font-size: 1rem;
        }}

        .machine-view-compact .scada-right-copy {{
            margin-top: 0.3rem;
            font-size: 0.82rem;
            line-height: 1.34;
        }}

        .machine-view-compact .compact-grid {{
            margin-top: 0.55rem;
        }}

        .machine-view-compact .scada-meta-block {{
            padding: 0.55rem 0.68rem;
        }}

        .machine-view-compact .scada-meta-value {{
            font-size: 0.9rem;
        }}

        .machine-view-compact .scada-data-list.compact {{
            gap: 0.35rem;
        }}

        .machine-view-compact .scada-data-list.compact div {{
            padding-bottom: 0.28rem;
        }}

        .machine-view-compact .scada-data-list.compact span,
        .machine-view-compact .scada-data-list.compact strong {{
            font-size: 0.82rem;
        }}

        .machine-view-compact .scada-alert-panel.compact {{
            padding: 0.72rem 0.84rem;
            margin-top: 0.35rem;
            margin-bottom: 0.45rem;
        }}

        .machine-view-compact .compact-note-panel {{
            min-height: 205px;
        }}

        .machine-view-compact .stTabs [data-baseweb="tab-list"] {{
            gap: 0.25rem;
            margin-bottom: 0.25rem;
        }}

        .machine-view-compact .stTabs [data-baseweb="tab"] {{
            padding-top: 0.32rem;
            padding-bottom: 0.32rem;
            font-size: 0.8rem;
        }}

        .machine-view-compact .stDataFrame {{
            border-radius: 12px;
            overflow: hidden;
        }}

        .machine-view-compact .js-plotly-plot,
        .machine-view-compact [data-testid="stPlotlyChart"] {{
            margin-bottom: 0 !important;
        }}

        .scada-metric-card {{
            background: linear-gradient(180deg, rgba(29, 43, 54, 0.95), rgba(20, 33, 43, 0.95));
            border: 1px solid rgba(53, 80, 97, 0.72);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            min-height: 100%;
        }}

        .scada-metric-label {{
            color: var(--scada-muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }}

        .scada-metric-value {{
            color: var(--scada-text);
            font-size: 1.35rem;
            font-weight: 700;
        }}

        .scada-metric-note {{
            color: var(--scada-muted);
            font-size: 0.84rem;
            margin-top: 0.25rem;
        }}

        .scada-kpi-grid {{
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.5rem 0 1rem 0;
        }}

        .scada-overview-card {{
            background: linear-gradient(180deg, rgba(25, 36, 46, 0.96), rgba(15, 24, 32, 0.96));
            border: 1px solid rgba(53, 80, 97, 0.72);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            min-height: 100%;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
        }}

        .scada-overview-card.low {{ border-color: rgba(47, 158, 115, 0.54); }}
        .scada-overview-card.warning {{ border-color: rgba(217, 164, 65, 0.54); }}
        .scada-overview-card.critical {{ border-color: rgba(217, 79, 79, 0.58); }}
        .scada-overview-card.neutral {{ border-color: rgba(77, 140, 168, 0.6); }}

        .scada-overview-label {{
            color: var(--scada-muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }}

        .scada-overview-value {{
            color: var(--scada-text);
            font-size: 1.45rem;
            font-weight: 700;
        }}

        .scada-info-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
        }}

        .scada-list {{
            margin: 0;
            padding-left: 1rem;
            color: var(--scada-text);
        }}

        .scada-list li {{
            margin-bottom: 0.35rem;
        }}

        .scada-section-title {{
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--scada-neutral);
            margin: 0 0 0.75rem 0;
        }}

        .scada-disclaimer {{
            border-left: 3px solid var(--scada-warning);
            padding: 0.75rem 0.9rem;
            background: rgba(217, 164, 65, 0.08);
            color: var(--scada-text);
            border-radius: 0 10px 10px 0;
            margin-top: 0.75rem;
        }}

        .scada-alert-panel {{
            border-left: 4px solid var(--scada-warning);
            background: linear-gradient(180deg, rgba(29, 43, 54, 0.96), rgba(18, 29, 38, 0.96));
            border-radius: 14px;
            padding: 0.95rem 1rem;
            margin-top: 0.75rem;
        }}

        .scada-alert-panel.low {{ border-left-color: var(--scada-low); }}
        .scada-alert-panel.warning {{ border-left-color: var(--scada-warning); }}
        .scada-alert-panel.critical {{ border-left-color: var(--scada-critical); }}

        .scada-alert-title {{
            color: var(--scada-text);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }}

        .scada-alert-text {{
            color: var(--scada-muted);
            font-size: 0.95rem;
        }}

        .scada-right-panel {{
            margin-bottom: 0.8rem;
        }}

        .scada-right-title {{
            font-size: 1.12rem;
            font-weight: 700;
            color: var(--scada-text);
        }}

        .scada-right-copy {{
            margin-top: 0.45rem;
            color: var(--scada-muted);
            line-height: 1.52;
        }}

        .scada-data-list {{
            display: grid;
            gap: 0.55rem;
        }}

        .scada-data-list div {{
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            padding-bottom: 0.5rem;
        }}

        .scada-data-list span {{
            color: var(--scada-muted);
            font-size: 0.9rem;
        }}

        .scada-data-list strong {{
            color: var(--scada-text);
            font-size: 0.92rem;
            text-align: right;
        }}

        .scada-visual-toolbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            flex-wrap: wrap;
        }}

        .scada-toolbar-group {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}

        .scada-chart-note {{
            color: var(--scada-muted);
            font-size: 0.88rem;
            margin: -0.1rem 0 0.55rem 0;
        }}

        @media (max-width: 900px) {{
            .scada-title {{ font-size: 1.55rem; }}
            .scada-shell {{ padding: 1rem; }}
            .scada-operator-grid {{ grid-template-columns: 1fr; }}
            .scada-kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            .scada-info-grid {{ grid-template-columns: 1fr; }}
            .scada-machine-title {{ font-size: 1.55rem; }}
            .scada-data-list div {{ flex-direction: column; align-items: flex-start; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
