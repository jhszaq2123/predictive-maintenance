from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import predict_failure


app = FastAPI(
    title="Predictive Maintenance AI API",
    description="Simple Random Forest API for AI4I 2020 machine failure prediction.",
)


class MachineInput(BaseModel):
    Type: str = Field(..., examples=["L", "M", "H"])
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(machine: MachineInput) -> dict[str, float | int]:
    payload = {
        "Type": machine.Type,
        "Air temperature [K]": machine.air_temperature_k,
        "Process temperature [K]": machine.process_temperature_k,
        "Rotational speed [rpm]": machine.rotational_speed_rpm,
        "Torque [Nm]": machine.torque_nm,
        "Tool wear [min]": machine.tool_wear_min,
    }
    return predict_failure(payload)
