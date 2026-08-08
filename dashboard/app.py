"""
Streamlit dashboard: shows current AQI, 3-day forecast (24h/48h/72h),
historical trend, SHAP feature importance, and hazard alerts for Lahore.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import hopsworks
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dotenv import load_dotenv
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
load_dotenv()

st.set_page_config(page_title="Lahore AQI Forecast", page_icon="🌫️", layout="wide")

FEATURE_COLS = [
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "hour", "day", "month", "day_of_week",
    "pm2_5_change_rate", "aqi_target_change_rate",
    "pm2_5_rolling_3h", "pm2_5_rolling_24h",
    "temperature", "humidity", "wind_speed", "pressure", "precipitation",
]

AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]


def get_aqi_category(aqi):
    aqi = round(aqi)
    for low, high, label, color in AQI_CATEGORIES:
        if low <= aqi <= high:
            return label, color
    return "Unknown", "#888888"


@st.cache_resource
def connect_hopsworks():
    project = hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        cert_folder="hopsworks_certs",
    )
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    return project, fs, mr


@st.cache_data(ttl=1800)  # refresh every 30 min
def load_recent_features(_fs, hours=200):
    fg = _fs.get_feature_group(name="aqi_lahore_features", version=2)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df.tail(hours).reset_index(drop=True)


@st.cache_resource
def load_models():
    models = {}
    for horizon in ["24h", "48h", "72h"]:
        path = os.path.join(MODELS_DIR, f"aqi_model_{horizon}.pkl")
        models[horizon] = joblib.load(path)
    return models


def main():
    st.title("🌫️ Lahore AQI Forecast Dashboard")
    st.caption("Real-time Air Quality Index tracking and 3-day forecast — Data science internship project")

    with st.spinner("Connecting to feature store and loading models..."):
        project, fs, mr = connect_hopsworks()
        df = load_recent_features(fs)
        models = load_models()

    latest = df.iloc[-1]
    current_aqi = latest["aqi_target"]
    current_label, current_color = get_aqi_category(current_aqi)

    # --- current AQI ---
    st.subheader("Current Conditions")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current AQI", f"{current_aqi:.0f}", current_label)
    col2.metric("PM2.5", f"{latest['pm2_5']:.1f} µg/m³")
    col3.metric("Last updated", latest["timestamp"].strftime("%Y-%m-%d %H:%M UTC"))

    if current_aqi > 150:
        st.error(f"⚠️ HAZARD ALERT: Current AQI ({current_aqi:.0f}) is in the '{current_label}' range. Limit outdoor exposure.")
    elif current_aqi > 100:
        st.warning(f"⚠️ Current AQI ({current_aqi:.0f}) is '{current_label}' — sensitive groups should take precaution.")

    # --- forecast ---
    st.subheader("3-Day Forecast")
    X_latest = latest[FEATURE_COLS].to_frame().T.astype(float)

    forecast_cols = st.columns(3)
    forecast_values = {}
    for i, horizon in enumerate(["24h", "48h", "72h"]):
        pred = models[horizon].predict(X_latest)[0]
        forecast_values[horizon] = pred
        label, color = get_aqi_category(pred)
        with forecast_cols[i]:
            st.metric(f"In {horizon}", f"{pred:.0f}", label)
            if pred > 150:
                st.error(f"Hazardous conditions expected in {horizon}")
            elif pred > 100:
                st.warning(f"Unhealthy air expected in {horizon}")

    # --- historical trend chart ---
    st.subheader("Recent AQI Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["aqi_target"], mode="lines", name="Historical AQI", line=dict(color="steelblue")))

    future_times = [latest["timestamp"] + pd.Timedelta(hours=int(h[:-1])) for h in ["24h", "48h", "72h"]]
    fig.add_trace(go.Scatter(x=future_times, y=list(forecast_values.values()), mode="markers+lines", name="Forecast", line=dict(color="orange", dash="dot"), marker=dict(size=10)))

    fig.update_layout(xaxis_title="Time", yaxis_title="AQI", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # --- SHAP feature importance ---
    st.subheader("Feature Importance (SHAP) — 24h Model")
    with st.spinner("Computing SHAP values..."):
        explainer = shap.Explainer(models["24h"], df[FEATURE_COLS].tail(100))
        shap_values = explainer(X_latest)

        fig2, ax = plt.subplots(figsize=(8, 6))
        shap.plots.bar(shap_values[0], show=False, ax=ax)
        st.pyplot(fig2)

    st.caption("Built with Streamlit, Hopsworks Feature Store & Model Registry, scikit-learn, and SHAP. Data: OpenWeather + Open-Meteo.")


if __name__ == "__main__":
    main()