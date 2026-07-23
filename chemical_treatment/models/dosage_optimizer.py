import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

from chemical_treatment.utils.preprocessor import build_preprocessor, prepare_features

MODEL_PATH = os.path.join("outputs", "models", "dosage_optimizer.pkl")


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("regressor", GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )),
        ]
    )


def train(df: pd.DataFrame) -> dict:
    X = prepare_features(df)
    y = df["dosage_ppm"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    return {"mae": round(mae, 4), "r2": round(r2, 4), "model_path": MODEL_PATH}


def load_model() -> Pipeline:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(pipeline: Pipeline, features: dict) -> float:
    df = pd.DataFrame([features])
    X = df[["dosage_ppm", "temperature_c", "ph", "water_hardness", "treatment_type"]]
    pred = pipeline.predict(X)
    return round(float(pred[0]), 2)
