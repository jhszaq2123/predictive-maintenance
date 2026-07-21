from __future__ import annotations

from pathlib import Path

from dashboard.components.charts import build_gauge_chart, build_pipeline_figure
from dashboard.components.data_loaders import (
    FD001_PREDICTIONS_PATH,
    REPORTS_DASHBOARD_DIR,
    REPORTS_METRICS_DIR,
    load_fd001_prediction_frame,
    load_metric_artifacts,
    read_json as _read_json,
)
from dashboard.components.layout import (
    SYSTEM_NAME,
    SYSTEM_SUBTITLE,
    render_disclaimer,
    render_machine_selector,
    render_metric_card,
    render_panel,
    render_section_title,
    render_status_badge,
    render_system_header,
)
from dashboard.components.machine_view import (
    color_from_rul,
    render_svg_machine,
    risk_info_from_rul,
    risk_level_from_rul,
    status_from_rul,
)
from dashboard.components.navigation import PAGE_LINKS, render_sidebar
from dashboard.components.styles import SCADA_THEME, inject_global_styles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
ASSETS_DIR = DASHBOARD_DIR / "assets"
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

PROJECT_TITLE = (
    "Analiza efektywnosci wybranych modeli AI dla potrzeb budowy systemu "
    "do predykcji awarii maszyn przemyslowych na podstawie danych czujnikowych"
)

COMPONENT_DETAILS = {
    "Engine": {
        "description": "Glowny element wizualizacji. Predykcja RUL dotyczy calego silnika, a nie pojedynczego podzespolu.",
        "status_hint": "Core monitored asset",
    },
    "Pump": {
        "description": "Element pogladowy pokazujacy przeplyw medium w maszynie. Nie jest oceniany oddzielnym modelem.",
        "status_hint": "Auxiliary subsystem",
    },
    "Bearings": {
        "description": "Pogladowy punkt odniesienia dla elementow wirujacych. Dashboard nie wykonuje predykcji komponentowej.",
        "status_hint": "Rotating support",
    },
    "Sensors": {
        "description": "Symboliczna reprezentacja sygnalow pomiarowych wykorzystywanych przez pipeline danych.",
        "status_hint": "Data source layer",
    },
    "Cooling System": {
        "description": "Wizualizacja pomocniczego ukladu chlodzenia. Kolor maszyny nadal odnosi sie do calego silnika.",
        "status_hint": "Thermal support",
    },
}


def artifact_exists(path: Path) -> bool:
    return path.exists()
