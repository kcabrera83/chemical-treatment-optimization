import numpy as np
import pandas as pd


TREATMENT_TYPES = [
    "scale_inhibitor",
    "corrosion_inhibitor",
    "demulsifier",
    "paraffin_inhibitor",
]

DOSAGE_RANGES = {
    "scale_inhibitor": (10, 150),
    "corrosion_inhibitor": (20, 200),
    "demulsifier": (5, 100),
    "paraffin_inhibitor": (15, 180),
}


def generate_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    treatment_type = rng.choice(TREATMENT_TYPES, size=n_samples)
    dosage_ppm = np.zeros(n_samples)
    temperature_c = rng.uniform(20, 120, size=n_samples)
    ph = rng.uniform(4.0, 10.0, size=n_samples)
    water_hardness = rng.uniform(50, 800, size=n_samples)

    for i, tt in enumerate(treatment_type):
        lo, hi = DOSAGE_RANGES[tt]
        dosage_ppm[i] = rng.uniform(lo, hi)

    effectiveness_base = (
        0.3
        + 0.15 * (dosage_ppm / 200)
        - 0.02 * ((temperature_c - 60) / 60)
        + 0.1 * ((ph - 7) / 3)
        - 0.1 * (water_hardness / 800)
    )
    effectiveness_noise = rng.normal(0, 0.05, size=n_samples)
    effectiveness_score = np.clip(effectiveness_base + effectiveness_noise, 0, 1)

    corrosion_base = (
        0.5
        - 0.3 * effectiveness_score
        + 0.002 * temperature_c
        - 0.0003 * dosage_ppm
        + 0.0001 * water_hardness
    )
    corrosion_rate = np.clip(
        corrosion_base + rng.normal(0, 0.05, size=n_samples), 0.01, 2.0
    )

    scale_base = (
        0.4
        - 0.35 * effectiveness_score
        + 0.001 * water_hardness
        - 0.001 * dosage_ppm
    )
    scale_deposition = np.clip(
        scale_base + rng.normal(0, 0.05, size=n_samples), 0.01, 1.5
    )

    cost_base = (
        0.4 * dosage_ppm
        + 0.1 * temperature_c
        + 0.05 * water_hardness
        + rng.normal(0, 5, size=n_samples)
    )
    treatment_cost_usd = np.clip(cost_base, 10, 500)

    df = pd.DataFrame(
        {
            "treatment_type": treatment_type,
            "dosage_ppm": np.round(dosage_ppm, 2),
            "temperature_c": np.round(temperature_c, 2),
            "ph": np.round(ph, 2),
            "water_hardness": np.round(water_hardness, 2),
            "treatment_cost_usd": np.round(treatment_cost_usd, 2),
            "effectiveness_score": np.round(effectiveness_score, 4),
            "corrosion_rate": np.round(corrosion_rate, 4),
            "scale_deposition": np.round(scale_deposition, 4),
        }
    )
    return df


def classify_effectiveness(score: float) -> str:
    if score < 0.25:
        return "poor"
    elif score < 0.50:
        return "fair"
    elif score < 0.75:
        return "good"
    return "excellent"
