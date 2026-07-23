"""API integration tests for Chemical Treatment Optimization FastAPI app."""

import sys
from fastapi.testclient import TestClient

sys.path.insert(0, ".")
from app import app

client = TestClient(app)

passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:
        failed += 1
        print(f"  FAIL  {name}: {e}")


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_models():
    r = client.get("/api/models")
    assert r.status_code == 200
    data = r.json()
    assert "dosage_optimizer" in data
    assert "effectiveness_predictor" in data


def test_optimize_valid():
    r = client.post("/api/optimize", json={
        "treatment_type": "scale_inhibitor",
        "temperature_c": 65,
        "ph": 7.2,
        "water_hardness": 350,
    })
    assert r.status_code == 200
    assert "optimal_dosage_ppm" in r.json()


def test_optimize_missing_field():
    r = client.post("/api/optimize", json={
        "treatment_type": "corrosion_inhibitor"
    })
    assert r.status_code == 422


def test_optimize_invalid_type():
    r = client.post("/api/optimize", json={
        "treatment_type": "invalid_type",
        "temperature_c": 60,
        "ph": 7,
        "water_hardness": 300,
    })
    assert r.status_code == 400


def test_predict_valid():
    r = client.post("/api/predict", json={
        "treatment_type": "corrosion_inhibitor",
        "dosage_ppm": 100,
        "temperature_c": 50,
        "ph": 6.5,
        "water_hardness": 200,
    })
    assert r.status_code == 200
    data = r.json()
    assert "predicted_effectiveness" in data
    assert data["predicted_effectiveness"] in ["poor", "fair", "good", "excellent"]


def test_predict_missing_field():
    r = client.post("/api/predict", json={
        "treatment_type": "demulsifier"
    })
    assert r.status_code == 422


def main():
    print("=" * 50)
    print("  API Test Suite")
    print("=" * 50)

    print("\n[Health]")
    test("GET /api/health", test_health)

    print("\n[Models Info]")
    test("GET /api/models", test_models)

    print("\n[Dosage Optimizer]")
    test("POST /api/optimize (valid)", test_optimize_valid)
    test("POST /api/optimize (missing field)", test_optimize_missing_field)
    test("POST /api/optimize (invalid type)", test_optimize_invalid_type)

    print("\n[Effectiveness Predictor]")
    test("POST /api/predict (valid)", test_predict_valid)
    test("POST /api/predict (missing field)", test_predict_missing_field)

    print("\n" + "=" * 50)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 50)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
