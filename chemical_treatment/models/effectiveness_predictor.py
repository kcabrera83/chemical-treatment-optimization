import os
import pickle

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from chemical_treatment.utils.preprocessor import (
    build_preprocessor,
    prepare_features,
    prepare_target_classification,
)

MODEL_PATH = os.path.join("outputs", "models", "effectiveness_predictor.pkl")


def _objective(trial, X_train, y_train_enc, X_val, y_val_enc, n_classes):
    """Optuna objective for Bayesian classification hyperparameters."""
    beta_sigma = trial.suggest_float("beta_sigma", 0.01, 5.0)
    obs_sigma = trial.suggest_float("obs_sigma", 0.1, 5.0)

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=beta_sigma, shape=n_classes)
        beta = pm.Normal("beta", mu=0, sigma=beta_sigma, shape=(X_train.shape[1], n_classes))
        logits = intercept + pm.math.dot(X_train, beta)
        y_obs = pm.Categorical("y_obs", logit_p=logits, observed=y_train_enc)
        trace = pm.sample(300, return_inferencedata=True, progressbar=False, chains=2)

    beta_samples = trace.posterior["beta"].mean(dim=["chain", "draw"]).values
    intercept_samples = trace.posterior["intercept"].mean(dim=["chain", "draw"]).values
    logits_val = intercept_samples + X_val @ beta_samples
    preds = np.argmax(logits_val, axis=1)
    acc = float(np.mean(preds == y_val_enc))
    return acc


def train(df: pd.DataFrame) -> dict:
    X_raw = prepare_features(df)
    y_str = prepare_target_classification(df)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)

    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    n_classes = len(le.classes_)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _objective(trial, X_train, y_train, X_test, y_test, n_classes),
        n_trials=5,
        show_progress_bar=False,
    )

    best = study.best_params
    with pm.Model() as final_model:
        intercept = pm.Normal("intercept", mu=0, sigma=best["beta_sigma"], shape=n_classes)
        beta = pm.Normal("beta", mu=0, sigma=best["beta_sigma"], shape=(X_train.shape[1], n_classes))
        logits = intercept + pm.math.dot(X_train, beta)
        y_obs = pm.Categorical("y_obs", logit_p=logits, observed=y_train)
        trace = pm.sample(500, return_inferencedata=True, progressbar=False)

    beta_mean = trace.posterior["beta"].mean(dim=["chain", "draw"]).values
    intercept_mean = trace.posterior["intercept"].mean(dim=["chain", "draw"]).values
    logits_test = intercept_mean + X_test @ beta_mean
    preds = np.argmax(logits_test, axis=1)
    acc = float(np.mean(preds == y_test))

    from sklearn.metrics import classification_report
    report = classification_report(
        y_test, preds, target_names=le.classes_, output_dict=True
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": final_model,
            "trace": trace,
            "preprocessor": preprocessor,
            "label_encoder": le,
        }, f)

    return {
        "accuracy": round(acc, 4),
        "report": report,
        "model_path": MODEL_PATH,
    }


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(model_data: dict, features: dict) -> str:
    df = pd.DataFrame([features])
    X_raw = df[["dosage_ppm", "temperature_c", "ph", "water_hardness", "treatment_type"]]
    preprocessor = model_data["preprocessor"]
    X = preprocessor.transform(X_raw)
    trace = model_data["trace"]
    le = model_data["label_encoder"]
    beta_mean = trace.posterior["beta"].mean(dim=["chain", "draw"]).values
    intercept_mean = trace.posterior["intercept"].mean(dim=["chain", "draw"]).values
    logits = intercept_mean + X @ beta_mean
    pred_idx = int(np.argmax(logits, axis=1)[0])
    return str(le.classes_[pred_idx])
