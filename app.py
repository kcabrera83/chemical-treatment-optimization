import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template
from chemical_treatment.models.dosage_optimizer import load_model as load_dosage_model, predict as predict_dosage
from chemical_treatment.models.effectiveness_predictor import load_model as load_effectiveness_model, predict as predict_effectiveness
from chemical_treatment.data_generator import TREATMENT_TYPES

app = Flask(__name__)

dosage_model = None
effectiveness_model = None


def get_models():
    global dosage_model, effectiveness_model
    if dosage_model is None:
        dosage_model = load_dosage_model()
    if effectiveness_model is None:
        effectiveness_model = load_effectiveness_model()
    return dosage_model, effectiveness_model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "chemical-treatment-optimization"})


@app.route("/api/models", methods=["GET"])
def models_info():
    get_models()
    return jsonify({
        "dosage_optimizer": {
            "type": "GradientBoostingRegressor",
            "features": ["treatment_type", "temperature_c", "ph", "water_hardness"],
            "target": "dosage_ppm",
        },
        "effectiveness_predictor": {
            "type": "RandomForestClassifier",
            "features": ["dosage_ppm", "temperature_c", "ph", "water_hardness", "treatment_type"],
            "target": "effectiveness_category (poor/fair/good/excellent)",
        },
        "treatment_types": TREATMENT_TYPES,
    })


@app.route("/api/optimize", methods=["POST"])
def optimize():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    required = ["treatment_type", "temperature_c", "ph", "water_hardness"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    if data["treatment_type"] not in TREATMENT_TYPES:
        return jsonify({"error": f"Invalid treatment_type. Must be one of: {TREATMENT_TYPES}"}), 400

    dm, _ = get_models()
    features = {
        "dosage_ppm": 0,
        "temperature_c": float(data["temperature_c"]),
        "ph": float(data["ph"]),
        "water_hardness": float(data["water_hardness"]),
        "treatment_type": data["treatment_type"],
    }
    optimal_dosage = predict_dosage(dm, features)

    return jsonify({
        "optimal_dosage_ppm": optimal_dosage,
        "treatment_type": data["treatment_type"],
        "conditions": {
            "temperature_c": features["temperature_c"],
            "ph": features["ph"],
            "water_hardness": features["water_hardness"],
        },
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    required = ["treatment_type", "dosage_ppm", "temperature_c", "ph", "water_hardness"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    if data["treatment_type"] not in TREATMENT_TYPES:
        return jsonify({"error": f"Invalid treatment_type. Must be one of: {TREATMENT_TYPES}"}), 400

    _, em = get_models()
    features = {
        "dosage_ppm": float(data["dosage_ppm"]),
        "temperature_c": float(data["temperature_c"]),
        "ph": float(data["ph"]),
        "water_hardness": float(data["water_hardness"]),
        "treatment_type": data["treatment_type"],
    }
    effectiveness = predict_effectiveness(em, features)

    return jsonify({
        "predicted_effectiveness": effectiveness,
        "treatment_type": data["treatment_type"],
        "dosage_ppm": features["dosage_ppm"],
        "conditions": {
            "temperature_c": features["temperature_c"],
            "ph": features["ph"],
            "water_hardness": features["water_hardness"],
        },
    })


@app.route("/api/docs", methods=["GET"])
def api_docs():
    return jsonify({
        "openapi": "3.0.0",
        "info": {"title": "Chemical Treatment Optimization", "version": "1.0.0"},
        "paths": {
            "/api/health": {"get": {"summary": "Health check"}},
            "/api/models": {"get": {"summary": "Model info"}},
            "/api/optimize": {"post": {"summary": "Optimize chemical treatment dosage"}},
            "/api/predict": {"post": {"summary": "Predict treatment effectiveness"}},
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5009, debug=False)
