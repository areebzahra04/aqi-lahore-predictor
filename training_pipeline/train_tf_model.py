"""
TensorFlow Training: trains a small feed-forward neural network for each
forecast horizon (24h, 48h, 72h) using the CSV produced by
export_training_data.py.

This script does NOT import hopsworks or hsfs on purpose — it runs in a
separate venv installed from requirements-tf.txt, to avoid the
numpy/protobuf/grpcio version conflicts between TensorFlow and hopsworks.

Run this in the TF venv:
    python -m venv venv-tf
    venv-tf\\Scripts\\activate   (Windows)
    pip install -r requirements-tf.txt
    python training_pipeline/train_tf_model.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

FEATURE_COLS = [
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "hour", "day", "month", "day_of_week",
    "pm2_5_change_rate", "aqi_target_change_rate",
    "pm2_5_rolling_3h", "pm2_5_rolling_24h",
    "temperature", "humidity", "wind_speed", "pressure", "precipitation",
]

HORIZONS = ["24h", "48h", "72h"]
ENSEMBLE_SIZE = 5   # number of models averaged per horizon, cancels out seed-to-seed noise

INPUT_CSV = "training_pipeline/tf_training_data.csv"
OUTPUT_DIR = "training_pipeline/tf_models"


def build_model(input_dim, seed):
    # Smaller network + L2 regularization: ~1,300 training rows is not
    # enough data for a large layer to learn reliably, it tends to
    # overfit or fail to escape a "predict near the mean" solution.
    init = keras.initializers.GlorotUniform(seed=seed)
    reg = regularizers.l2(2e-3)
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(12, activation="relu", kernel_regularizer=reg, kernel_initializer=init),
        layers.Dropout(0.3),
        layers.Dense(6, activation="relu", kernel_regularizer=reg, kernel_initializer=init),
        layers.Dense(1, kernel_initializer=init),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


def main():
    tf.random.set_seed(42)

    df = pd.read_csv(INPUT_CSV)
    split_idx = int(len(df) * 0.8)

    X = df[FEATURE_COLS].values
    X_train_full, X_test_full = X[:split_idx], X[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_metrics = {}
    y_scalers = {}

    for horizon in HORIZONS:
        y = df[f"target_{horizon}"].values.reshape(-1, 1)
        y_train_raw, y_test_raw = y[:split_idx], y[split_idx:]

        # Scale the target too -- with MSE loss, an unscaled AQI target
        # (range roughly 0-300+) makes optimization much harder for a
        # randomly-initialized network than for Ridge/RandomForest,
        # and was the main cause of the negative R2 seen without this.
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train_raw).flatten()
        y_test = y_test_raw.flatten()

        # Train an ensemble of ENSEMBLE_SIZE small networks with different
        # seeds and average their predictions. A single network's R2 can
        # swing a lot run-to-run on a dataset this small (you saw this:
        # 24h went from -0.218 to +0.234 between two runs of the same
        # code) -- averaging several seeds cancels most of that noise out.
        horizon_dir = os.path.join(OUTPUT_DIR, f"aqi_model_{horizon}_tf_ensemble")
        os.makedirs(horizon_dir, exist_ok=True)

        test_preds_scaled = []

        for seed in range(ENSEMBLE_SIZE):
            tf.random.set_seed(seed)
            model = build_model(X_train_scaled.shape[1], seed=seed)
            early_stop = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True
            )
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5
            )

            model.fit(
                X_train_scaled, y_train,
                validation_split=0.15,
                epochs=300,
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=0,
            )

            test_preds_scaled.append(model.predict(X_test_scaled, verbose=0).flatten())
            model.save(os.path.join(horizon_dir, f"member_{seed}.keras"))

        preds_scaled_avg = np.mean(test_preds_scaled, axis=0).reshape(-1, 1)
        preds = y_scaler.inverse_transform(preds_scaled_avg).flatten()

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        metrics = {"RMSE": round(rmse, 3), "MAE": round(mae, 3), "R2": round(r2, 3)}
        all_metrics[horizon] = metrics
        print(f"{horizon} (ensemble of {ENSEMBLE_SIZE}): {metrics}")

        y_scalers[horizon] = y_scaler

    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "tf_scaler.pkl"))
    joblib.dump(y_scalers, os.path.join(OUTPUT_DIR, "tf_y_scalers.pkl"))

    with open(os.path.join(OUTPUT_DIR, "tf_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Saved TF models, scaler, and metrics to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()