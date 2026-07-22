"""
Training Pipeline: reads features from Hopsworks Feature Store,
trains models for 3 forecast horizons (24h, 48h, 72h),
evaluates them, and registers the best model per horizon
to the Hopsworks Model Registry.
Designed to run daily via GitHub Actions.
"""

import os
import joblib
import numpy as np
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()

HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]

FEATURE_COLS = [
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "hour", "day", "month", "day_of_week",
    "pm2_5_change_rate", "aqi_target_change_rate",
    "pm2_5_rolling_3h", "pm2_5_rolling_24h",
    "temperature", "humidity", "wind_speed", "pressure", "precipitation",
]

MODELS = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
}


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
    df = df.dropna(subset=["target_24h", "target_48h", "target_72h"]).reset_index(drop=True)
    return df


def train_and_evaluate(df):
    X = df[FEATURE_COLS]
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

    horizons = {
        "24h": (df["target_24h"].iloc[:split_idx], df["target_24h"].iloc[split_idx:]),
        "48h": (df["target_48h"].iloc[:split_idx], df["target_48h"].iloc[split_idx:]),
        "72h": (df["target_72h"].iloc[:split_idx], df["target_72h"].iloc[split_idx:]),
    }

    results = []
    trained = {}

    for horizon_name, (y_tr, y_te) in horizons.items():
        for model_name, model in MODELS.items():
            m = model.__class__(**model.get_params())
            m.fit(X_train, y_tr)
            preds = m.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
            mae = float(mean_absolute_error(y_te, preds))
            r2 = float(r2_score(y_te, preds))

            results.append({"horizon": horizon_name, "model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2})
            trained[f"{horizon_name}_{model_name}"] = (m, {"RMSE": round(rmse, 3), "MAE": round(mae, 3), "R2": round(r2, 3)})

    return pd.DataFrame(results), trained, X_train


def select_best_per_horizon(results_df, trained):
    best = {}
    for horizon in ["24h", "48h", "72h"]:
        subset = results_df[results_df["horizon"] == horizon]
        best_row = subset.loc[subset["R2"].idxmax()]
        key = f"{horizon}_{best_row['model']}"
        best[horizon] = (best_row["model"], trained[key][0], trained[key][1])
    return best


def register_models(project, best_models, X_train_sample):
    mr = project.get_model_registry()
    os.makedirs("models", exist_ok=True)

    for horizon, (model_name, model, metrics) in best_models.items():
        path = f"models/aqi_model_{horizon}.pkl"
        joblib.dump(model, path)

        hw_model = mr.python.create_model(
            name=f"aqi_model_{horizon}",
            metrics=metrics,
            description=f"AQI forecast model for t+{horizon} horizon ({model_name}), daily retrain on pollutant+weather features for Lahore",
            input_example=X_train_sample.iloc[[0]],
        )
        hw_model.save(path)
        print(f"Registered {horizon} model: {model_name} | metrics: {metrics}")


def main():
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, cert_folder="hopsworks_certs")
    fs = project.get_feature_store()

    df = load_features(fs)
    print(f"Loaded {len(df)} feature rows")

    df = build_targets(df)
    print(f"{len(df)} rows usable after building targets")

    results_df, trained, X_train_sample = train_and_evaluate(df)
    print(results_df)

    best_models = select_best_per_horizon(results_df, trained)
    for horizon, (name, _, metrics) in best_models.items():
        print(f"Best for {horizon}: {name} -> {metrics}")

    register_models(project, best_models, X_train_sample)


if __name__ == "__main__":
    main()