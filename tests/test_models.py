import pytest
import os
import pickle
import numpy as np
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


def test_dosage_model_loads():
    path = os.path.join(MODELS_DIR, "dosage_optimizer.pkl")
    assert os.path.exists(path)
    with open(path, "rb") as f:
        model = pickle.load(f)
    assert model is not None


def test_effectiveness_model_loads():
    path = os.path.join(MODELS_DIR, "effectiveness_predictor.pkl")
    assert os.path.exists(path)
    with open(path, "rb") as f:
        model = pickle.load(f)
    assert model is not None


def test_dosage_prediction():
    from chemical_treatment.models.dosage_optimizer import load_model, predict
    model = load_model()
    features = {
        "dosage_ppm": 0,
        "temperature_c": 60.0,
        "ph": 7.0,
        "water_hardness": 150.0,
        "treatment_type": "scale_inhibitor",
    }
    result = predict(model, features)
    assert result is not None
    assert result > 0


def test_effectiveness_prediction():
    from chemical_treatment.models.effectiveness_predictor import load_model, predict
    model = load_model()
    features = {
        "dosage_ppm": 50.0,
        "temperature_c": 60.0,
        "ph": 7.0,
        "water_hardness": 150.0,
        "treatment_type": "scale_inhibitor",
    }
    result = predict(model, features)
    assert result is not None
    assert result in ["poor", "fair", "good", "excellent"]


def test_dosage_pipeline_predict():
    from chemical_treatment.models.dosage_optimizer import load_model
    pipeline = load_model()
    df = pd.DataFrame([{
        "dosage_ppm": 0,
        "temperature_c": 80.0,
        "ph": 5.5,
        "water_hardness": 200.0,
        "treatment_type": "corrosion_inhibitor",
    }])
    X = df[["dosage_ppm", "temperature_c", "ph", "water_hardness", "treatment_type"]]
    pred = pipeline.predict(X)
    assert pred is not None
    assert len(pred) == 1
    assert pred[0] > 0
