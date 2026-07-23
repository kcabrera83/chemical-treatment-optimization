# Chemical Treatment Optimization

ML-based oilfield chemical treatment analysis and dosage optimization system.

## Overview

This project uses machine learning to optimize oilfield chemical treatments including scale inhibitors, corrosion inhibitors, demulsifiers, and paraffin inhibitors. It provides:

- **Dosage Optimizer**: GradientBoostingRegressor that predicts optimal chemical dosage (ppm) based on operating conditions
- **Effectiveness Predictor**: RandomForestClassifier that classifies treatment effectiveness (poor/fair/good/excellent)

## Project Structure

```
chemical-treatment-optimization/
├── chemical_treatment/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dosage_optimizer.py
│   │   └── effectiveness_predictor.py
│   └── utils/
│       ├── __init__.py
│       └── preprocessor.py
├── templates/
│   └── index.html
├── outputs/
│   └── models/
├── train.py
├── app.py
├── test_api.py
├── requirements.txt
├── setup.py
└── .github/
    └── workflows/
        └── ci.yml
```

## Setup

```bash
pip install -r requirements.txt
```

## Train Models

```bash
python train.py
```

## Run API

```bash
python app.py
```

Server starts on http://localhost:5009

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Web UI dashboard |
| GET | /api/health | Health check |
| GET | /api/models | Model information |
| POST | /api/optimize | Optimal dosage prediction |
| POST | /api/predict | Effectiveness classification |

### POST /api/optimize

```json
{
    "treatment_type": "scale_inhibitor",
    "temperature_c": 65,
    "ph": 7.2,
    "water_hardness": 350
}
```

### POST /api/predict

```json
{
    "treatment_type": "corrosion_inhibitor",
    "dosage_ppm": 100,
    "temperature_c": 50,
    "ph": 6.5,
    "water_hardness": 200
}
```

## Run Tests

```bash
python test_api.py
```

## Features

- Synthetic data generation with realistic oilfield chemistry relationships
- GradientBoostingRegressor for continuous dosage optimization
- RandomForestClassifier for effectiveness category prediction
- Flask REST API with JSON endpoints
- Dark-theme web dashboard with Chart.js visualization
- GitHub Actions CI pipeline

## Treatment Types

- Scale Inhibitor
- Corrosion Inhibitor
- Demulsifier
- Paraffin Inhibitor

## Effectiveness Categories

- poor (score < 0.25)
- fair (0.25 <= score < 0.50)
- good (0.50 <= score < 0.75)
- excellent (score >= 0.75)

---

Elaborado por Ing. Kelvin Cabrera
