"""FastAPI for chemical treatment optimization using PyMC Bayesian inference + Optuna."""

import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from chemical_treatment.models.dosage_optimizer import load_model as load_dosage_model, predict as predict_dosage
from chemical_treatment.models.effectiveness_predictor import load_model as load_effectiveness_model, predict as predict_effectiveness
from chemical_treatment.data_generator import TREATMENT_TYPES

app = FastAPI(
    title="Chemical Treatment Optimization",
    description="Chemical dosage optimization and effectiveness prediction (PyMC Bayesian + Optuna)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

dosage_model = None
effectiveness_model = None


def get_models():
    global dosage_model, effectiveness_model
    if dosage_model is None:
        dosage_model = load_dosage_model()
    if effectiveness_model is None:
        effectiveness_model = load_effectiveness_model()
    return dosage_model, effectiveness_model


@app.on_event("startup")
async def load_models():
    try:
        get_models()
    except Exception as e:
        print(f"[WARN] Error loading models: {e}")


class OptimizeRequest(BaseModel):
    treatment_type: str
    temperature_c: float
    ph: float
    water_hardness: float


class OptimizeResponse(BaseModel):
    optimal_dosage_ppm: float
    treatment_type: str
    conditions: dict


class PredictRequest(BaseModel):
    treatment_type: str
    dosage_ppm: float
    temperature_c: float
    ph: float
    water_hardness: float


class PredictResponse(BaseModel):
    predicted_effectiveness: str
    treatment_type: str
    dosage_ppm: float
    conditions: dict


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "service": "chemical-treatment-optimization",
        "framework": "pymc/optuna",
    }


@app.get("/api/models")
async def models_info():
    get_models()
    return {
        "dosage_optimizer": {
            "type": "PyMC Bayesian Regression + Optuna tuning",
            "features": ["treatment_type", "temperature_c", "ph", "water_hardness"],
            "target": "dosage_ppm",
            "framework": "pymc/optuna",
        },
        "effectiveness_predictor": {
            "type": "PyMC Bayesian Classification + Optuna tuning",
            "features": ["dosage_ppm", "temperature_c", "ph", "water_hardness", "treatment_type"],
            "target": "effectiveness_category (poor/fair/good/excellent)",
            "framework": "pymc/optuna",
        },
        "treatment_types": TREATMENT_TYPES,
    }


@app.post("/api/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest):
    if request.treatment_type not in TREATMENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid treatment_type. Must be one of: {TREATMENT_TYPES}",
        )
    dm, _ = get_models()
    features = {
        "dosage_ppm": 0,
        "temperature_c": float(request.temperature_c),
        "ph": float(request.ph),
        "water_hardness": float(request.water_hardness),
        "treatment_type": request.treatment_type,
    }
    optimal_dosage = predict_dosage(dm, features)
    return OptimizeResponse(
        optimal_dosage_ppm=optimal_dosage,
        treatment_type=request.treatment_type,
        conditions={
            "temperature_c": features["temperature_c"],
            "ph": features["ph"],
            "water_hardness": features["water_hardness"],
        },
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if request.treatment_type not in TREATMENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid treatment_type. Must be one of: {TREATMENT_TYPES}",
        )
    _, em = get_models()
    features = {
        "dosage_ppm": float(request.dosage_ppm),
        "temperature_c": float(request.temperature_c),
        "ph": float(request.ph),
        "water_hardness": float(request.water_hardness),
        "treatment_type": request.treatment_type,
    }
    effectiveness = predict_effectiveness(em, features)
    return PredictResponse(
        predicted_effectiveness=effectiveness,
        treatment_type=request.treatment_type,
        dosage_ppm=features["dosage_ppm"],
        conditions={
            "temperature_c": features["temperature_c"],
            "ph": features["ph"],
            "water_hardness": features["water_hardness"],
        },
    )
