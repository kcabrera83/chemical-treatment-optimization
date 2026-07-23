# Architecture - Chemical Treatment Optimization

## System Overview

```
+------------------+     +-------------------+     +------------------+
|   Data Layer     | --> |   Model Layer     | --> |   API Layer      |
| (Data Generator) |     | (ML Models)       |     | (Flask REST)     |
+------------------+     +-------------------+     +------------------+
                                                          |
                                                          v
                                                 +------------------+
                                                 | Dashboard Layer  |
                                                 | (HTML/CSS/JS)    |
                                                 +------------------+
```

## Components

### Data Layer

- **Source**: Synthetic data generator (`data_generator.py`)
- **Samples**: 2,000 treatment records
- **Treatment types**: scale_inhibitor, corrosion_inhibitor, demulsifier, paraffin_inhibitor
- **Features**: treatment_type, temperature_c, ph, water_hardness, dosage_ppm
- **Targets**: dosage_ppm (regression), effectiveness_category (classification)

### Model Layer

#### Dosage Optimizer
- **Algorithm**: GradientBoostingRegressor
- **Task**: Predict optimal chemical dosage (ppm)
- **Input**: treatment_type, temperature_c, ph, water_hardness
- **Output**: Optimal dosage in ppm
- **Serialization**: pickle (`.pkl`)

#### Effectiveness Predictor
- **Algorithm**: RandomForestClassifier
- **Task**: Classify treatment effectiveness
- **Input**: dosage_ppm, temperature_c, ph, water_hardness, treatment_type
- **Output**: Effectiveness category (poor/fair/good/excellent)
- **Serialization**: pickle (`.pkl`)

### Preprocessing Pipeline

1. Categorical treatment_type encoded via OneHotEncoder
2. Numeric features scaled with StandardScaler
3. Combined feature vector passed to model

### API Layer

- **Framework**: Flask
- **Port**: 5009
- **Format**: JSON request/response
- **Endpoints**: 6 (optimize, predict, health, models, docs, index)

### Dashboard Layer

- **Frontend**: HTML/CSS/JS (Jinja2 templates)
- **Charts**: Chart.js for visualization
- **Theme**: Dark theme UI

## Data Flow

### Dosage Optimization Flow
1. User provides treatment_type + conditions
2. API validates input and treatment type
3. GradientBoosting model predicts optimal dosage
4. Response returned with dosage and echoed conditions

### Effectiveness Prediction Flow
1. User provides treatment_type + dosage + conditions
2. API validates input
3. RandomForest model classifies effectiveness
4. Response returned with effectiveness category

## Project Structure

```
chemical-treatment-optimization/
├── chemical_treatment/
│   ├── __init__.py
│   ├── data_generator.py              # Synthetic data generation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dosage_optimizer.py        # GradientBoosting regressor
│   │   └── effectiveness_predictor.py # RandomForest classifier
│   └── utils/
│       ├── __init__.py
│       └── preprocessor.py            # Scaling and encoding
├── templates/
│   └── index.html                     # Dashboard UI
├── outputs/models/                    # Saved model artifacts
├── train.py                           # Training pipeline
├── app.py                             # Flask API server
├── test_api.py                        # API test suite
├── requirements.txt
└── setup.py
```

## Model Evaluation

### Dosage Optimizer
- MAE: ~5-10 ppm
- R2: ~0.90+

### Effectiveness Predictor
- Accuracy: ~0.85+
- Balanced precision/recall across categories

---

*Elaborado por Ing. Kelvin Cabrera*
