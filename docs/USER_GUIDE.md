# User Guide - Chemical Treatment Optimization

## Overview

The Chemical Treatment Optimization system uses machine learning to optimize oilfield chemical treatments including scale inhibitors, corrosion inhibitors, demulsifiers, and paraffin inhibitors. It predicts optimal dosages and classifies treatment effectiveness.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
cd chemical-treatment-optimization
pip install -r requirements.txt
```

### Train Models

```bash
python train.py
```

This generates 2,000 synthetic samples and trains:
- Dosage Optimizer (GradientBoostingRegressor)
- Effectiveness Predictor (RandomForestClassifier)

### Run the Server

```bash
python app.py
```

Open `http://localhost:5009` in your browser.

## Dashboard Features

- **Dosage Optimization Panel** - Enter treatment conditions and get optimal dosage (ppm)
- **Effectiveness Prediction Panel** - Enter dosage and conditions to predict treatment effectiveness
- **Chart.js Visualizations** - Visual breakdown of treatment parameters
- **Dark Theme UI** - Modern dark-themed dashboard

## API Usage

### Optimize Dosage (curl)

```bash
curl -X POST http://localhost:5009/api/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "treatment_type": "scale_inhibitor",
    "temperature_c": 65,
    "ph": 7.2,
    "water_hardness": 350
  }'
```

### Optimize Dosage (Python)

```python
import requests

response = requests.post("http://localhost:5009/api/optimize", json={
    "treatment_type": "scale_inhibitor",
    "temperature_c": 65,
    "ph": 7.2,
    "water_hardness": 350
})
result = response.json()
print(f"Optimal dosage: {result['optimal_dosage_ppm']} ppm")
```

### Predict Effectiveness (curl)

```bash
curl -X POST http://localhost:5009/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "treatment_type": "corrosion_inhibitor",
    "dosage_ppm": 100,
    "temperature_c": 50,
    "ph": 6.5,
    "water_hardness": 200
  }'
```

### Predict Effectiveness (Python)

```python
import requests

response = requests.post("http://localhost:5009/api/predict", json={
    "treatment_type": "corrosion_inhibitor",
    "dosage_ppm": 100,
    "temperature_c": 50,
    "ph": 6.5,
    "water_hardness": 200
})
result = response.json()
print(f"Effectiveness: {result['predicted_effectiveness']}")
```

### Check Health

```bash
curl http://localhost:5009/api/health
```

### Get Model Info

```bash
curl http://localhost:5009/api/models
```

## Typical Workflow

1. Use `/api/optimize` to get the optimal dosage for your conditions
2. Use `/api/predict` with the recommended dosage to verify expected effectiveness
3. Adjust conditions and re-optimize if needed

## Running Tests

```bash
python test_api.py
```

## Troubleshooting

- **Invalid treatment_type error**: Use one of: scale_inhibitor, corrosion_inhibitor, demulsifier, paraffin_inhibitor
- **Models not loaded**: Run `python train.py` first
- **Port in use**: Change port in `app.py`

---

*Elaborado por Ing. Kelvin Cabrera*
