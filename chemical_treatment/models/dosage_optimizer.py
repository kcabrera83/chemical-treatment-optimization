import os
import pickle

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import optuna
from sklearn.model_selection import train_test_split

from chemical_treatment.utils.preprocessor import build_preprocessor, prepare_features

MODEL_PATH = os.path.join("outputs", "models", "dosage_optimizer.pkl")


def _objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for hyperparameter tuning of Bayesian model."""
    mu_sigma = trial.suggest_float("mu_sigma", 0.1, 10.0)
    beta_sigma = trial.suggest_float("beta_sigma", 0.01, 5.0)
    obs_sigma = trial.suggest_float("obs_sigma", 0.1, 10.0)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=mu_sigma)
        beta = pm.Normal("beta", mu=0, sigma=beta_sigma, shape=X_train.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=obs_sigma)
        y_obs = pm.Normal(
            "y_obs",
            mu=mu + pm.math.dot(X_train, beta),
            sigma=sigma,
            observed=y_train,
        )
        trace = pm.sample(500, return_inferencedata=True, progressbar=False, chains=2)

    beta_mean = trace.posterior["beta"].mean(dim=["chain", "draw"]).values
    mu_mean = trace.posterior["mu"].mean(dim=["chain", "draw"]).values
    y_pred = mu_mean + X_val @ beta_mean
    mae = np.mean(np.abs(y_val - y_pred))
    return mae


def train(df: pd.DataFrame) -> dict:
    X_raw = prepare_features(df)
    y = df["dosage_ppm"].values

    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: _objective(trial, X_train, y_train, X_test, y_test),
        n_trials=5,
        show_progress_bar=False,
    )

    best = study.best_params
    with pm.Model() as final_model:
        mu = pm.Normal("mu", mu=0, sigma=best["mu_sigma"])
        beta = pm.Normal("beta", mu=0, sigma=best["beta_sigma"], shape=X_train.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=best["obs_sigma"])
        y_obs = pm.Normal(
            "y_obs",
            mu=mu + pm.math.dot(X_train, beta),
            sigma=sigma,
            observed=y_train,
        )
        trace = pm.sample(1000, return_inferencedata=True, progressbar=False)

    y_pred = (
        trace.posterior["mu"].mean(dim=["chain", "draw"]).values
        + X_test @ trace.posterior["beta"].mean(dim=["chain", "draw"]).values
    )
    mae = float(np.mean(np.abs(y_test - y_pred)))
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": final_model, "trace": trace, "preprocessor": preprocessor}, f)

    return {"mae": round(mae, 4), "r2": round(r2, 4), "model_path": MODEL_PATH}


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(model_data: dict, features: dict) -> float:
    df = pd.DataFrame([features])
    X_raw = df[["dosage_ppm", "temperature_c", "ph", "water_hardness", "treatment_type"]]
    preprocessor = model_data["preprocessor"]
    X = preprocessor.transform(X_raw)
    trace = model_data["trace"]
    mu_mean = trace.posterior["mu"].mean(dim=["chain", "draw"]).values
    beta_mean = trace.posterior["beta"].mean(dim=["chain", "draw"]).values
    pred = mu_mean + X @ beta_mean
    return round(float(pred[0]), 2)
