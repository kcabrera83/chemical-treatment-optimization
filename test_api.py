import sys
import os
import json
import time
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = "http://127.0.0.1:5009"
passed = 0
failed = 0


def test(name, method, path, data=None, expect_status=200):
    global passed, failed
    url = BASE + path
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req)
        status = resp.getcode()
        raw = resp.read().decode()
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            body = {"html": True}
    except urllib.error.HTTPError as e:
        status = e.code
        raw = e.read().decode()
        body = json.loads(raw) if raw else {}
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        failed += 1
        return

    if status == expect_status:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name}: expected {expect_status}, got {status}")
    return body


def main():
    global passed, failed
    print("=" * 50)
    print("  API Test Suite")
    print("=" * 50)

    print("\n[Health]")
    test("GET /api/health", "GET", "/api/health")

    print("\n[Models Info]")
    r = test("GET /api/models", "GET", "/api/models")
    if r:
        assert "dosage_optimizer" in r
        assert "effectiveness_predictor" in r

    print("\n[Dosage Optimizer]")
    test("POST /api/optimize (valid)", "POST", "/api/optimize", {
        "treatment_type": "scale_inhibitor",
        "temperature_c": 65,
        "ph": 7.2,
        "water_hardness": 350,
    })
    test("POST /api/optimize (missing field)", "POST", "/api/optimize", {
        "treatment_type": "corrosion_inhibitor"
    }, expect_status=400)
    test("POST /api/optimize (invalid type)", "POST", "/api/optimize", {
        "treatment_type": "invalid_type",
        "temperature_c": 60,
        "ph": 7,
        "water_hardness": 300,
    }, expect_status=400)

    print("\n[Effectiveness Predictor]")
    r = test("POST /api/predict (valid)", "POST", "/api/predict", {
        "treatment_type": "corrosion_inhibitor",
        "dosage_ppm": 100,
        "temperature_c": 50,
        "ph": 6.5,
        "water_hardness": 200,
    })
    if r:
        assert "predicted_effectiveness" in r
        assert r["predicted_effectiveness"] in ["poor", "fair", "good", "excellent"]
    test("POST /api/predict (missing field)", "POST", "/api/predict", {
        "treatment_type": "demulsifier"
    }, expect_status=400)

    print("\n[Root]")
    test("GET /", "GET", "/")

    print("\n" + "=" * 50)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 50)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
