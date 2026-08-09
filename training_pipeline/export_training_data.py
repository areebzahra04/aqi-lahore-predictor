"""
Export Training Data: reads features from the Hopsworks Feature Store,
builds the same 24h/48h/72h targets used by train_model.py, and writes
the result to a local CSV.

Why this exists: TensorFlow and hopsworks/hsfs pin conflicting
numpy/protobuf/grpcio versions, so they can't reliably live in the same
Python environment. This script is the hand-off point — it runs in the
SAME venv as train_model.py (the one with `hopsworks` installed) and
never imports tensorflow. train_tf_model.py then picks up the CSV in a
separate venv that never imports hopsworks.

Run this in the existing project venv:
    python training_pipeline/export_training_data.py
"""

import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]

FEATURE_COLS = [
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "hour", "day", "month", "day_of_week",
    "pm2_5_change_rate", "aqi_target_change_rate",
    "pm2_5_rolling_3h", "pm2_5_rolling_24h",
    "temperature", "humidity", "wind_speed", "pressure", "precipitation",
]

TARGET_COLS = ["target_24h", "target_48h", "target_72h"]

OUTPUT_PATH = "training_pipeline/tf_training_data.csv"


def load_features(fs):
    fg = fs.get_feature_group(name="aqi_lahore_features", version=2)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_targets(df):
    df = df.copy()
    df["target_24h"] = df["aqi_target"].shift(-24)
    df["target_48h"] = df["aqi_target"].shift(-48)
    df["target_72h"] = df["aqi_target"].shift(-72)
    df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
    return df


def main():
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, cert_folder="hopsworks_certs")
    fs = project.get_feature_store()

    df = load_features(fs)
    print(f"Loaded {len(df)} feature rows")

    df = build_targets(df)
    print(f"{len(df)} rows usable after building targets")

    os.makedirs("training_pipeline", exist_ok=True)
    df[FEATURE_COLS + TARGET_COLS].to_csv(OUTPUT_PATH, index=False)
    print(f"Exported {len(df)} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()