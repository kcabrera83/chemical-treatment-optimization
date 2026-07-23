import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app

client = app.test_client()


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"


def test_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.get_json()
    assert "dosage_optimizer" in data
    assert "effectiveness_predictor" in data
    assert "treatment_types" in data


def test_api_docs():
    response = client.get("/api/docs")
    assert response.status_code == 200
    data = response.get_json()
    assert data["openapi"] == "3.0.0"


def test_optimize_valid():
    response = client.post("/api/optimize", json={
        "treatment_type": "scale_inhibitor",
        "temperature_c": 60.0,
        "ph": 7.0,
        "water_hardness": 150.0,
    })
    assert response.status_code == 200
    data = response.get_json()
    assert "optimal_dosage_ppm" in data
    assert "treatment_type" in data
    assert "conditions" in data
    assert data["optimal_dosage_ppm"] > 0


def test_optimize_missing_fields():
    response = client.post("/api/optimize", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_optimize_invalid_treatment_type():
    response = client.post("/api/optimize", json={
        "treatment_type": "invalid_type",
        "temperature_c": 60.0,
        "ph": 7.0,
        "water_hardness": 150.0,
    })
    assert response.status_code == 400


def test_optimize_all_treatment_types():
    from chemical_treatment.data_generator import TREATMENT_TYPES
    for tt in TREATMENT_TYPES:
        response = client.post("/api/optimize", json={
            "treatment_type": tt,
            "temperature_c": 60.0,
            "ph": 7.0,
            "water_hardness": 150.0,
        })
        assert response.status_code == 200
        data = response.get_json()
        assert data["optimal_dosage_ppm"] > 0


def test_predict_valid():
    response = client.post("/api/predict", json={
        "treatment_type": "scale_inhibitor",
        "dosage_ppm": 50.0,
        "temperature_c": 60.0,
        "ph": 7.0,
        "water_hardness": 150.0,
    })
    assert response.status_code == 200
    data = response.get_json()
    assert "predicted_effectiveness" in data
    assert data["predicted_effectiveness"] in ["poor", "fair", "good", "excellent"]


def test_predict_missing_fields():
    response = client.post("/api/predict", json={
        "treatment_type": "scale_inhibitor",
    })
    assert response.status_code == 400


def test_predict_invalid_treatment_type():
    response = client.post("/api/predict", json={
        "treatment_type": "bad_type",
        "dosage_ppm": 50.0,
        "temperature_c": 60.0,
        "ph": 7.0,
        "water_hardness": 150.0,
    })
    assert response.status_code == 400


def test_optimize_no_json():
    response = client.post("/api/optimize", content_type="application/json")
    assert response.status_code in [400, 415]
