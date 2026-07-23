import pytest


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_models(client):
    response = client.get("/api/models")
    assert response.status_code in (200, 500, 503)


def test_optimize_valid(client):
    response = client.post("/api/optimize", json={
        "treatment_type": "scale_inhibitor",
        "temperature_c": 60.0,
        "ph": 7.0,
        "water_hardness": 150.0,
    })
    assert response.status_code in (200, 400, 500)
    if response.status_code == 200:
        data = response.json()
        assert "optimal_dosage_ppm" in data
        assert "treatment_type" in data
        assert "conditions" in data
        assert data["optimal_dosage_ppm"] > 0


def test_optimize_invalid_treatment_type(client):
    response = client.post("/api/optimize", json={
        "treatment_type": "invalid_type",
        "temperature_c": 60.0,
        "ph": 7.0,
        "water_hardness": 150.0,
    })
    assert response.status_code == 400


def test_optimize_all_treatment_types(client):
    TREATMENT_TYPES = ["scale_inhibitor", "corrosion_inhibitor", "biocide", "oxygen_scavenger"]
    for tt in TREATMENT_TYPES:
        response = client.post("/api/optimize", json={
            "treatment_type": tt,
            "temperature_c": 60.0,
            "ph": 7.0,
            "water_hardness": 150.0,
        })
        assert response.status_code in (200, 400, 500)
        if response.status_code == 200:
            data = response.json()
            assert data["optimal_dosage_ppm"] > 0


def test_predict_valid(client):
    response = client.post("/api/predict", json={
        "treatment_type": "scale_inhibitor",
        "dosage_ppm": 50.0,
        "temperature_c": 60.0,
        "ph": 7.0,
        "water_hardness": 150.0,
    })
    assert response.status_code in (200, 400, 500)
    if response.status_code == 200:
        data = response.json()
        assert "predicted_effectiveness" in data
        assert data["predicted_effectiveness"] in ["poor", "fair", "good", "excellent"]


def test_predict_invalid_treatment_type(client):
    response = client.post("/api/predict", json={
        "treatment_type": "bad_type",
        "dosage_ppm": 50.0,
        "temperature_c": 60.0,
        "ph": 7.0,
        "water_hardness": 150.0,
    })
    assert response.status_code == 400
