# Streamlit Dashboard

The dashboard is a presentation-only demonstrator for the Predictive Maintenance project.
It reads saved artifacts from `reports/` and does not load ML models, does not rebuild sequences, and does not execute runtime inference.

## Presentation-only architecture

- no TensorFlow or Keras imports inside dashboard pages or components
- no model loading
- no `model.predict(...)`
- no preprocessing or scaling
- no runtime prediction generation
- all displayed values come from previously generated files in `reports/`

## Component structure

```text
dashboard/
├─ app.py
├─ machine_config.py
├─ shared.py
├─ components/
│  ├─ data_loaders.py
│  ├─ charts.py
│  ├─ machine_view.py
│  ├─ navigation.py
│  ├─ styles.py
│  └─ layout.py
├─ pages/
└─ assets/
```

### Responsibilities

- `components/data_loaders.py`
  Loads JSON and CSV dashboard artifacts and validates required schemas.
- `components/charts.py`
  Stores Plotly chart builders such as the pipeline figure and RUL gauge.
- `components/machine_view.py`
  Stores risk-to-style mapping and SVG rendering helpers.
- `components/navigation.py`
  Stores sidebar navigation and global sidebar metrics.
- `components/styles.py`
  Injects shared SCADA/HMI-inspired CSS.
- `components/layout.py`
  Stores reusable UI helpers for headers, panels, cards, and disclaimers.
- `machine_config.py`
  Stores the three future machine-mode definitions.
- `shared.py`
  Thin compatibility layer re-exporting commonly used dashboard symbols.

## Machine modes

Sprint SCADA 0 prepares configuration for three demonstration modes:

1. `industrial_drive`
2. `centrifugal_pump`
3. `nasa_turbofan`

These modes currently share existing saved prediction artifacts.
Their component layouts are illustrative only and do not claim real sensor-to-component mapping.

## Assets structure

```text
dashboard/assets/
├─ industrial_drive/
├─ centrifugal_pump/
├─ nasa_turbofan/
├─ icons/
├─ shared/
└─ machine_diagram.svg
```

The legacy `machine_diagram.svg` remains in place in this sprint to preserve backward compatibility.

## Run

```powershell
streamlit run dashboard/app.py
```

## Current pages

- `Project Overview`
- `Machine View`
- `Model Results`
- `Predictions`

## Artifact inputs

The dashboard currently reads:

- `reports/metrics/random_forest_metrics.json`
- `reports/metrics/xgboost_metrics.json`
- `reports/metrics/lstm_fd001_metrics.json`
- `reports/metrics/lstm_fd001_test_metrics.json`
- `reports/metrics/model_comparison.csv`
- `reports/dashboard/fd001_predictions.csv`
- selected figures from `reports/figures/`

## Visual scope

Sprint SCADA 0 introduces only a safe structural refactor and a lightweight SCADA/HMI-inspired visual system:

- dark technical theme
- reusable panels and metric cards
- shared system header
- machine-mode selector prepared for future visuals

It does not yet include:

- final SVG machine graphics
- Three.js
- live telemetry
- runtime inference
- new models or ML pipeline changes

## Next steps

- finalize the SCADA/HMI landing layout
- add the first dedicated `Industrial Drive System` illustration
- introduce component highlighting states
- keep the dashboard strictly presentation-only
