# Chemical Treatment Optimization

ML-based oilfield chemical treatment analysis and dosage optimization system using Bayesian methods and hyperparameter tuning.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Bayesian Inference | **PyMC3** - probabilistic programming |
| Hyperparameter Tuning | **Optuna** - Bayesian optimization |
| Data Processing | pandas, numpy, joblib |
| Web Server | **FastAPI** + uvicorn |
| Monitoring | prometheus-fastapi-instrumentator |
| Validation | pydantic v2 |
| Visualization | matplotlib, seaborn |

### Key Libraries
- PyMC3 - Bayesian statistical modeling
- Optuna - Automatic hyperparameter optimization
- FastAPI - Modern async web framework
- pandas / numpy - Data processing

## Overview

This project uses machine learning to optimize oilfield chemical treatments including scale inhibitors, corrosion inhibitors, demulsifiers, and paraffin inhibitors. It provides:

- **Dosage Optimizer**: PyMC3 Bayesian model that predicts optimal chemical dosage (ppm) based on operating conditions
- **Effectiveness Predictor**: Optuna-tuned classifier that classifies treatment effectiveness (poor/fair/good/excellent)

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
