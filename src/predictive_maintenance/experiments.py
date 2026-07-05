from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentDefinition:
    name: str
    dataset_key: str
    task_type: str
    target_column: str | None
    notes: str


EXPERIMENTS: dict[str, ExperimentDefinition] = {
    "ai4i_classification_baseline": ExperimentDefinition(
        name="AI4I classification baseline",
        dataset_key="ai4i2020",
        task_type="classification",
        target_column="Machine failure",
        notes="First baseline for failure classification on tabular sensor data.",
    ),
    "c_mapss_rul_baseline": ExperimentDefinition(
        name="NASA CMAPSS RUL baseline",
        dataset_key="c_mapss",
        task_type="regression_rul",
        target_column="RUL",
        notes="Primary experiment track for remaining useful life prediction.",
    ),
    "secom_anomaly_baseline": ExperimentDefinition(
        name="SECOM anomaly baseline",
        dataset_key="secom",
        task_type="anomaly_detection",
        target_column=None,
        notes="Candidate track for anomaly detection and reconstruction-error analysis.",
    ),
}
