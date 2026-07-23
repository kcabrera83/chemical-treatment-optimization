import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["dosage_ppm", "temperature_c", "ph", "water_hardness"]
CATEGORICAL_FEATURES = ["treatment_type"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[ALL_FEATURES].copy()


def prepare_target_regression(df: pd.DataFrame) -> np.ndarray:
    return df["dosage_ppm"].values


def prepare_target_classification(df: pd.DataFrame) -> np.ndarray:
    bins = [0, 0.25, 0.50, 0.75, 1.01]
    labels = ["poor", "fair", "good", "excellent"]
    return pd.cut(df["effectiveness_score"], bins=bins, labels=labels).astype(str)
