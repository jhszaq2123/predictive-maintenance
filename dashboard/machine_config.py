from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "dashboard" / "assets"

MACHINE_MODES = {
    "industrial_drive": {
        "display_name": "Industrial Drive System",
        "description": "Operator view for a motor-driven industrial powertrain with a centrifugal pump load.",
        "component_names": [
            "Electric Motor",
            "Coupling",
            "Gearbox",
            "Bearings",
            "Shaft",
            "Pump",
        ],
        "asset_path": ASSETS_DIR / "industrial_drive" / "industrial_drive_system.svg",
        "dataset_context": "Dataset context: NASA CMAPSS FD001 offline prediction artifacts.",
        "disclaimer": (
            "This visualization is demonstrative. The displayed risk and RUL values still come from saved "
            "project artifacts and do not represent component-level sensor assignments."
        ),
        "visualization_type": "Industrial drive SCADA/HMI schematic",
        "component_descriptions": {
            "Electric Motor": "Primary electrical drive responsible for converting electrical energy into rotational motion.",
            "Coupling": "Mechanical connector transferring torque from the motor shaft to the gearbox input.",
            "Gearbox": "Transmission stage used to adapt rotational speed and torque for downstream equipment.",
            "Bearings": "Rotating support elements stabilizing the shaft line and reducing friction.",
            "Shaft": "Rotating mechanical axis transferring power across the drive train.",
            "Pump": "Centrifugal pump stage representing the driven process equipment in the demonstration layout.",
        },
    },
    "centrifugal_pump": {
        "display_name": "Centrifugal Pump System",
        "description": "Pump-oriented demonstration layout for future SCADA/HMI expansion.",
        "component_names": [
            "Motor",
            "Shaft",
            "Bearings",
            "Mechanical Seal",
            "Impeller",
            "Pump Housing",
        ],
        "asset_path": ASSETS_DIR / "machine_diagram.svg",
        "dataset_context": "Dataset context: presentation mode using saved offline project artifacts.",
        "disclaimer": (
            "This visualization is demonstrative. The predictive values still originate from saved artifacts, "
            "and the components do not imply a real mapping between sensors and physical parts."
        ),
        "visualization_type": "Demonstration placeholder layout",
        "component_descriptions": {
            "Motor": "Illustrative motor component for the planned pump-system view.",
            "Shaft": "Illustrative shaft component for the planned pump-system view.",
            "Bearings": "Illustrative bearing support for the planned pump-system view.",
            "Mechanical Seal": "Illustrative sealing assembly for the planned pump-system view.",
            "Impeller": "Illustrative hydraulic impeller for the planned pump-system view.",
            "Pump Housing": "Illustrative housing for the planned pump-system view.",
        },
    },
    "nasa_turbofan": {
        "display_name": "NASA Turbofan",
        "description": "Illustrative whole-engine view for the NASA CMAPSS FD001 baseline workflow.",
        "component_names": [
            "Fan",
            "Low Pressure Compressor",
            "High Pressure Compressor",
            "Combustor",
            "High Pressure Turbine",
            "Low Pressure Turbine",
            "Nozzle",
        ],
        "asset_path": ASSETS_DIR / "machine_diagram.svg",
        "dataset_context": "Dataset context: NASA CMAPSS FD001 saved evaluation artifacts.",
        "disclaimer": (
            "NASA CMAPSS sensors remain anonymous in the source dataset. This component view is illustrative "
            "only, and RUL still applies to the whole engine rather than to individual parts."
        ),
        "visualization_type": "Illustrative turbofan placeholder layout",
        "component_descriptions": {
            "Fan": "Illustrative inlet fan stage for the turbofan presentation mode.",
            "Low Pressure Compressor": "Illustrative compression stage for the turbofan presentation mode.",
            "High Pressure Compressor": "Illustrative high-pressure compression stage for the turbofan presentation mode.",
            "Combustor": "Illustrative combustor section for the turbofan presentation mode.",
            "High Pressure Turbine": "Illustrative high-pressure turbine stage for the turbofan presentation mode.",
            "Low Pressure Turbine": "Illustrative low-pressure turbine stage for the turbofan presentation mode.",
            "Nozzle": "Illustrative exhaust nozzle for the turbofan presentation mode.",
        },
    },
}
