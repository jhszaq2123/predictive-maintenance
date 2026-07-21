# Predictive Maintenance AI

An end-to-end machine learning pipeline for predictive maintenance using industrial sensor data.

The repository currently provides a validated tabular pipeline built on the AI4I 2020 dataset:

```text
AI4I 2020 Dataset
        │
        ▼
Preprocessing
        │
        ▼
Random Forest
        │
        ▼
Evaluation
        │
        ▼
Model Serialization
        │
        ▼
Prediction API
```

The codebase also includes early scaffolding for additional datasets and models, but those parts are not yet presented here as completed baselines.

## Features

### ✅ Implemented

- AI4I 2020 preprocessing pipeline with explicit target preparation
- Leakage-aware feature selection for machine failure classification
- One-hot encoding for categorical features
- Standard scaling for numeric features
- Stratified train/test split
- Random Forest baseline training
- Random Forest evaluation with accuracy, precision, recall, F1, ROC-AUC, and confusion matrix
- Serialized model bundle for local inference
- Prediction utility for single machine records
- FastAPI prototype with `/health` and `/predict`
- Basic automated tests for dataset configuration and Random Forest pipeline behavior

### 🚧 In Progress

- XGBoost baseline scripts and artifacts are present in the repository, but the baseline is still under formal validation
- Public release hardening and repository cleanup for broader reproducibility
- Separation of validated baselines from future experiment scaffolding

### 📅 Planned

- Formal XGBoost baseline completion
- Side-by-side Random Forest vs. XGBoost comparison
- NASA CMAPSS integration for RUL prediction
- Sequence construction for time-series modeling
- LSTM experiments
- GRU experiments
- SECOM experiments for missing-data-heavy industrial settings
- FastAPI improvements and a more complete demonstration layer

## Project Structure

```text
predictive-maintenance-ai/
├── api/                # FastAPI application
├── data/               # Local raw, interim, and processed datasets
├── docs/               # Supporting project notes and dataset references
├── experiments/        # Experiment scripts and lightweight experiment outputs
├── models/             # Serialized models generated locally
├── notebooks/          # Exploratory notebooks
├── reports/            # Figures and metrics generated locally
├── scripts/            # Utility scripts, including dataset download helpers
├── src/                # Training, preprocessing, evaluation, and prediction code
├── tests/              # Automated tests
├── pyproject.toml      # Pytest configuration and project metadata
├── requirements.txt    # Python dependencies
└── README.md
```

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- FastAPI
- Uvicorn
- pytest
- Jupyter Notebook

## Datasets

### AI4I 2020

Implemented in the current baseline workflow.

This is the only dataset fully wired into preprocessing, model training, evaluation, and prediction. It is treated as a binary classification task with `Machine failure` as the target.

Expected local file:

```text
data/raw/ai4i2020.csv
```

### NASA CMAPSS

Planned.

The repository already contains dataset metadata and analysis-related scaffolding, but CMAPSS is not yet part of the validated experimental pipeline. It is intended for future Remaining Useful Life (RUL) experiments with sequence models.

### SECOM

Planned.

The repository includes references and exploratory support for SECOM, but it is not yet integrated into a validated modeling workflow. It is intended for later experiments involving high-dimensional industrial data with substantial missingness.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org-or-username>/predictive-maintenance.git
cd predictive-maintenance
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download datasets

Use the provided helper script to download the datasets referenced by the project:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_datasets.ps1
```

For the validated AI4I workflow, make sure this file exists after download:

```text
data/raw/ai4i2020.csv
```

### 5. Prepare the local environment

This repository does not ship raw datasets, processed datasets, trained models, or generated reports. Those artifacts are expected to be created locally.

## Usage

### Preprocess AI4I 2020

```powershell
python .\src\data_preprocessing.py
```

This step:

- loads the raw CSV
- removes identifier columns and known leakage columns
- encodes `Type`
- scales numeric features
- creates a stratified train/test split
- writes processed artifacts to `data/processed/`

### Train the Random Forest baseline

```powershell
python .\src\train_random_forest.py
```

This writes the trained model bundle to:

```text
models/random_forest.pkl
```

### Evaluate the Random Forest baseline

```powershell
python .\src\evaluate.py --model random_forest
```

This generates:

- `reports/metrics/random_forest_metrics.json`
- `reports/figures/random_forest_confusion_matrix.png`

### Run a local prediction

```powershell
python .\src\predict.py
```

### Start the FastAPI server

```powershell
uvicorn api.main:app --reload
```

Available endpoints:

- `GET /health`
- `POST /predict`

Example request payload:

```json
{
  "Type": "M",
  "air_temperature_k": 298.5,
  "process_temperature_k": 309.0,
  "rotational_speed_rpm": 1450,
  "torque_nm": 48.0,
  "tool_wear_min": 120
}
```

## Current Results

Current validated baseline: `RandomForestClassifier` on AI4I 2020.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9805 |
| Precision | 0.7959 |
| Recall | 0.5735 |
| F1 Score | 0.6667 |
| ROC-AUC | 0.9701 |

The current Random Forest baseline shows strong overall classification performance, especially in accuracy and ROC-AUC. At the same time, recall remains noticeably lower than the other headline metrics, which suggests that failure cases are still harder to capture reliably. This makes the baseline useful as a reference point while also motivating a careful comparison with XGBoost in the next stage.

Confusion matrix:

```text
[[1922, 10],
 [  29, 39]]
```

## Roadmap

- [ ] XGBoost baseline
- [ ] Random Forest vs. XGBoost comparison
- [ ] NASA CMAPSS integration
- [ ] Sequence generation
- [ ] LSTM experiments
- [ ] GRU experiments
- [ ] SECOM integration
- [ ] FastAPI improvements

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
