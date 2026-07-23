import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chemical_treatment.data_generator import generate_dataset
from chemical_treatment.models.dosage_optimizer import train as train_dosage
from chemical_treatment.models.effectiveness_predictor import train as train_effectiveness


def main():
    print("=" * 60)
    print("  Chemical Treatment Optimization - Training Pipeline")
    print("  Framework: PyMC (Bayesian) + Optuna")
    print("=" * 60)

    print("\n[1/3] Generating synthetic dataset (2000 samples)...")
    df = generate_dataset(n_samples=2000)
    csv_path = os.path.join("outputs", "training_data.csv")
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  Dataset saved to {csv_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Treatment types: {df['treatment_type'].unique().tolist()}")

    print("\n[2/3] Training Dosage Optimizer (PyMC Bayesian Regression + Optuna)...")
    dosage_results = train_dosage(df)
    print(f"  MAE:  {dosage_results['mae']}")
    print(f"  R2:   {dosage_results['r2']}")
    print(f"  Model saved to: {dosage_results['model_path']}")

    print("\n[3/3] Training Effectiveness Predictor (PyMC Bayesian Classification + Optuna)...")
    eff_results = train_effectiveness(df)
    print(f"  Accuracy: {eff_results['accuracy']}")
    report = eff_results["report"]
    for cls in ["poor", "fair", "good", "excellent"]:
        if cls in report:
            print(f"    {cls}: P={report[cls]['precision']:.3f}  R={report[cls]['recall']:.3f}  F1={report[cls]['f1-score']:.3f}")
    print(f"  Model saved to: {eff_results['model_path']}")

    print("\n" + "=" * 60)
    print("  Training complete. All models saved to outputs/models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
