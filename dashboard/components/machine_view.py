from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import plotly.graph_objects as go

from dashboard.machine_config import MACHINE_MODES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "dashboard" / "assets"
DEFAULT_MACHINE_ASSET = ASSETS_DIR / "machine_diagram.svg"

RISK_LEVELS = [
    {"name": "Low", "color": "#2F9E73", "threshold": 120, "status": "Normal"},
    {"name": "Moderate", "color": "#D9A441", "threshold": 60, "status": "Warning"},
    {"name": "High", "color": "#D66A3D", "threshold": 30, "status": "Critical watch"},
    {"name": "Critical", "color": "#D94F4F", "threshold": -1, "status": "Critical"},
]

RISK_DISPLAY_MAP = {
    "Low": "LOW",
    "Moderate": "MODERATE",
    "High": "HIGH",
    "Critical": "HIGH",
}

ALERT_MESSAGES = {
    "LOW": "Normal operating conditions",
    "MODERATE": "Monitor component condition",
    "HIGH": "Maintenance recommended",
}

COMPONENT_ORDER = [
    "Electric Motor",
    "Coupling",
    "Gearbox",
    "Bearings",
    "Shaft",
    "Pump",
]

COMPONENT_HOTSPOTS: dict[str, list[tuple[float, float]]] = {
    "Electric Motor": [(190, 292), (222, 286), (250, 280), (282, 272), (312, 266)],
    "Coupling": [(396, 282), (404, 272), (412, 282)],
    "Gearbox": [(522, 286), (564, 274), (604, 262), (628, 286)],
    "Bearings": [(752, 286), (760, 278), (820, 286), (828, 278)],
    "Shaft": [(458, 278), (520, 278), (680, 278), (868, 278)],
    "Pump": [(982, 294), (1012, 284), (1046, 292), (1080, 304)],
}


def risk_info_from_rul(rul: float) -> dict[str, str | float]:
    for level in RISK_LEVELS:
        if rul >= level["threshold"]:
            return level
    return RISK_LEVELS[-1]


def risk_level_from_rul(rul: float) -> str:
    return str(risk_info_from_rul(rul)["name"])


def status_from_rul(rul: float) -> str:
    return str(risk_info_from_rul(rul)["status"])


def color_from_rul(rul: float) -> str:
    return str(risk_info_from_rul(rul)["color"])


def normalize_component_key(component_name: str) -> str:
    return component_name.lower().replace(" ", "_").replace("-", "_")


def presentation_risk_level(risk_name: str) -> str:
    return RISK_DISPLAY_MAP.get(risk_name, "HIGH")


def alert_message_for_risk(risk_name: str) -> str:
    return ALERT_MESSAGES[presentation_risk_level(risk_name)]


def risk_tone_for_label(risk_label: str) -> str:
    if risk_label == "LOW":
        return "low"
    if risk_label == "MODERATE":
        return "warning"
    return "critical"


def machine_status_for_risk(risk_label: str) -> str:
    if risk_label == "LOW":
        return "Stable"
    if risk_label == "MODERATE":
        return "Observe"
    return "Action required"


def health_index_from_rul(rul: float, maximum_rul: float = 160.0) -> int:
    bounded = max(0.0, min(rul, maximum_rul))
    return int(round((bounded / maximum_rul) * 100))


def trend_label_from_risk(risk_label: str) -> str:
    if risk_label == "LOW":
        return "Stable"
    if risk_label == "MODERATE":
        return "Watching"
    return "Degrading"


def trend_delta_from_error(absolute_error: float) -> str:
    if absolute_error <= 5:
        return "Tight fit"
    if absolute_error <= 20:
        return "Moderate spread"
    return "Wide spread"


def build_component_highlight_palette(selected_component: str) -> dict[str, dict[str, str]]:
    component_keys = [
        "electric_motor",
        "coupling",
        "gearbox",
        "bearings",
        "shaft",
        "pump",
    ]
    selected_key = normalize_component_key(selected_component)
    palette: dict[str, dict[str, str]] = {}
    for key in component_keys:
        is_selected = key == selected_key
        palette[key] = {
            "fill": "#29404F" if is_selected else "#223542",
            "top_fill": "#476A80" if is_selected else "#304755",
            "side_fill": "#1A2A34" if is_selected else "#18252F",
            "stroke": "#F4F8FA" if is_selected else "#5F7888",
            "stroke_width": "4.8" if is_selected else "2.2",
        }
    palette["pump"]["nozzle_fill"] = "#49728A" if selected_key == "pump" else "#233947"
    return palette


def render_industrial_drive_svg(selected_component: str) -> str:
    template = (ASSETS_DIR / "industrial_drive" / "industrial_drive_system.svg").read_text(encoding="utf-8")
    palette = build_component_highlight_palette(selected_component)
    replacements = {
        "{{ELECTRIC_MOTOR_FILL}}": palette["electric_motor"]["fill"],
        "{{ELECTRIC_MOTOR_TOP_FILL}}": palette["electric_motor"]["top_fill"],
        "{{ELECTRIC_MOTOR_SIDE_FILL}}": palette["electric_motor"]["side_fill"],
        "{{ELECTRIC_MOTOR_STROKE}}": palette["electric_motor"]["stroke"],
        "{{ELECTRIC_MOTOR_STROKE_WIDTH}}": palette["electric_motor"]["stroke_width"],
        "{{COUPLING_FILL}}": palette["coupling"]["fill"],
        "{{COUPLING_TOP_FILL}}": palette["coupling"]["top_fill"],
        "{{COUPLING_STROKE}}": palette["coupling"]["stroke"],
        "{{COUPLING_STROKE_WIDTH}}": palette["coupling"]["stroke_width"],
        "{{GEARBOX_FILL}}": palette["gearbox"]["fill"],
        "{{GEARBOX_TOP_FILL}}": palette["gearbox"]["top_fill"],
        "{{GEARBOX_SIDE_FILL}}": palette["gearbox"]["side_fill"],
        "{{GEARBOX_STROKE}}": palette["gearbox"]["stroke"],
        "{{GEARBOX_STROKE_WIDTH}}": palette["gearbox"]["stroke_width"],
        "{{BEARINGS_FILL}}": palette["bearings"]["fill"],
        "{{BEARINGS_TOP_FILL}}": palette["bearings"]["top_fill"],
        "{{BEARINGS_STROKE}}": palette["bearings"]["stroke"],
        "{{BEARINGS_STROKE_WIDTH}}": palette["bearings"]["stroke_width"],
        "{{SHAFT_FILL}}": palette["shaft"]["fill"],
        "{{SHAFT_TOP_FILL}}": palette["shaft"]["top_fill"],
        "{{SHAFT_STROKE}}": palette["shaft"]["stroke"],
        "{{SHAFT_STROKE_WIDTH}}": palette["shaft"]["stroke_width"],
        "{{PUMP_FILL}}": palette["pump"]["fill"],
        "{{PUMP_TOP_FILL}}": palette["pump"]["top_fill"],
        "{{PUMP_NOZZLE_FILL}}": palette["pump"]["nozzle_fill"],
        "{{PUMP_STROKE}}": palette["pump"]["stroke"],
        "{{PUMP_STROKE_WIDTH}}": palette["pump"]["stroke_width"],
    }
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def render_svg_machine(selected_component: str, rul: float, asset_path: Path | None = None) -> str:
    template_path = asset_path or DEFAULT_MACHINE_ASSET
    if template_path == MACHINE_MODES["industrial_drive"]["asset_path"]:
        return render_industrial_drive_svg(selected_component)

    template = template_path.read_text(encoding="utf-8")
    machine_color = color_from_rul(rul)
    replacements = {
        "{{MACHINE_FILL}}": machine_color,
        "{{ENGINE_STROKE}}": "#E8EEF2" if selected_component == "Engine" else "#6D7A84",
        "{{PUMP_STROKE}}": "#E8EEF2" if selected_component == "Pump" else "#6D7A84",
        "{{BEARINGS_STROKE}}": "#E8EEF2" if selected_component == "Bearings" else "#6D7A84",
        "{{SENSORS_STROKE}}": "#E8EEF2" if selected_component == "Sensors" else "#6D7A84",
        "{{COOLING_STROKE}}": "#E8EEF2" if selected_component == "Cooling System" else "#6D7A84",
        "{{ENGINE_WIDTH}}": "4" if selected_component == "Engine" else "2",
        "{{PUMP_WIDTH}}": "4" if selected_component == "Pump" else "2",
        "{{BEARINGS_WIDTH}}": "4" if selected_component == "Bearings" else "2",
        "{{SENSORS_WIDTH}}": "4" if selected_component == "Sensors" else "2",
        "{{COOLING_WIDTH}}": "4" if selected_component == "Cooling System" else "2",
    }
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def build_interactive_machine_figure(selected_component: str, rul: float) -> go.Figure:
    svg_markup = render_industrial_drive_svg(selected_component)
    image_uri = "data:image/svg+xml;utf8," + quote(svg_markup)
    risk_color = color_from_rul(rul)

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=image_uri,
            xref="x",
            yref="y",
            x=0,
            y=520,
            sizex=1200,
            sizey=520,
            sizing="stretch",
            layer="below",
            opacity=1.0,
        )
    )

    for component_name in COMPONENT_ORDER:
        coordinates = COMPONENT_HOTSPOTS[component_name]
        is_selected = component_name == selected_component
        x_values = [point[0] for point in coordinates]
        y_values = [520 - point[1] for point in coordinates]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=component_name,
                customdata=[[component_name]] * len(coordinates),
                marker={
                    "size": 28 if is_selected else 24,
                    "color": risk_color if is_selected else "#E8EEF2",
                    "opacity": 0.24 if is_selected else 0.06,
                    "line": {
                        "color": "#F4F8FA" if is_selected else "#A8BAC6",
                        "width": 1.6 if is_selected else 0.8,
                    },
                    "symbol": "circle",
                },
                hovertemplate=f"{component_name}<br>Click to inspect<extra></extra>",
                selected={"marker": {"opacity": 0.34, "size": 30}},
                unselected={"marker": {"opacity": 0.06}},
                showlegend=False,
            )
        )

    fig.update_xaxes(visible=False, range=[0, 1200], fixedrange=True)
    fig.update_yaxes(visible=False, range=[0, 520], fixedrange=True)
    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
        clickmode="event+select",
    )
    return fig


def selected_component_from_event(selection_event: object) -> str | None:
    if selection_event is None:
        return None

    points = None
    if hasattr(selection_event, "selection"):
        selection = getattr(selection_event, "selection")
        points = getattr(selection, "points", None)
    if points is None and isinstance(selection_event, dict):
        selection_payload = selection_event.get("selection", {})
        if isinstance(selection_payload, dict):
            points = selection_payload.get("points")

    if not points:
        return None

    first_point = points[0]
    customdata = None
    if hasattr(first_point, "customdata"):
        customdata = getattr(first_point, "customdata")
    elif isinstance(first_point, dict):
        customdata = first_point.get("customdata")

    if isinstance(customdata, (list, tuple)) and customdata:
        value = customdata[0]
        return str(value)
    if isinstance(customdata, str):
        return customdata
    return None
