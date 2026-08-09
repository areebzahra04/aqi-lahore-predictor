"""
Register TF Models: uploads the TensorFlow model ENSEMBLES trained by
train_tf_model.py (plus the feature/target scalers) to the Hopsworks
Model Registry, as separate entries (aqi_model_24h_tf, etc.) alongside
the existing Ridge/RandomForest models from train_model.py.

Each horizon is a folder of several .keras files (member_0.keras,
member_1.keras, ...). At inference time, load all members and average
their predictions -- do not use a single member alone.

This script only uploads files and metadata — it does NOT need
tensorflow installed, so it runs in the SAME venv as train_model.py
(the one with `hopsworks`).

Run this in the existing project venv, after train_tf_model.py has
produced training_pipeline/tf_models/:
    python training_pipeline/register_tf_model.py
"""

import os
import json
import hopsworks
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]

TF_MODELS_DIR = "training_pipeline/tf_models"
TRAINING_CSV = "training_pipeline/tf_training_data.csv"


def main():
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, cert_folder="hopsworks_certs")
    mr = project.get_model_registry()

    with open(os.path.join(TF_MODELS_DIR, "tf_metrics.json")) as f:
        all_metrics = json.load(f)

    sample_df = pd.read_csv(TRAINING_CSV).iloc[[0]]
    feature_cols = [c for c in sample_df.columns if not c.startswith("target_")]
    input_example = sample_df[feature_cols]

    for horizon, metrics in all_metrics.items():
        # Whole ensemble directory gets uploaded as one model version
        ensemble_dir = os.path.join(TF_MODELS_DIR, f"aqi_model_{horizon}_tf_ensemble")

        hw_model = mr.python.create_model(
            name=f"aqi_model_{horizon}_tf",
            metrics=metrics,
            description=(
                f"AQI forecast model for t+{horizon} horizon "
                f"(ensemble of small TensorFlow dense NNs, averaged), "
                f"trained on pollutant+weather features for Lahore"
            ),
            input_example=input_example,
        )
        hw_model.save(ensemble_dir)
        print(f"Registered aqi_model_{horizon}_tf (ensemble) | metrics: {metrics}")

    scaler_model = mr.python.create_model(
        name="aqi_tf_scaler",
        description="StandardScaler fitted on training features, required to use the TensorFlow AQI models for inference",
    )
    scaler_model.save(os.path.join(TF_MODELS_DIR, "tf_scaler.pkl"))
    print("Registered aqi_tf_scaler")

    y_scaler_model = mr.python.create_model(
        name="aqi_tf_y_scalers",
        description="Per-horizon target StandardScalers -- TF model outputs are scaled and must be inverse-transformed with these before use",
    )
    y_scaler_model.save(os.path.join(TF_MODELS_DIR, "tf_y_scalers.pkl"))
    print("Registered aqi_tf_y_scalers")


if __name__ == "__main__":
    main()