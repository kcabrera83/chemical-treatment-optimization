import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

from chemical_treatment.utils.preprocessor import (
    build_preprocessor,
    prepare_features,
    prepare_target_classification,
)

MODEL_PATH = os.path.join("outputs", "models", "effectiveness_predictor.pkl")


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                class_weight="balanced",
            )),
        ]
    )


def train(df: pd.DataFrame) -> dict:
    X = prepare_features(df)
    y = prepare_target_classification(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    return {
        "accuracy": round(acc, 4),
        "report": report,
        "model_path": MODEL_PATH,
    }


def load_model() -> Pipeline:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(pipeline: Pipeline, features: dict) -> str:
    df = pd.DataFrame([features])
    X = df[["dosage_ppm", "temperature_c", "ph", "water_hardness", "treatment_type"]]
    pred = pipeline.predict(X)
    return str(pred[0])
