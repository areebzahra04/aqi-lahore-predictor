# """
# Streamlit dashboard: shows current AQI, 3-day forecast (24h/48h/72h),
# historical trend, SHAP feature importance, and hazard alerts for Lahore.
# """

# import os
# import joblib
# import numpy as np
# import pandas as pd
# import streamlit as st
# import hopsworks
# import shap
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# from dotenv import load_dotenv
# MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
# load_dotenv()

# st.set_page_config(page_title="Lahore AQI Forecast", page_icon="🌫️", layout="wide")

# FEATURE_COLS = [
#     "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
#     "hour", "day", "month", "day_of_week",
#     "pm2_5_change_rate", "aqi_target_change_rate",
#     "pm2_5_rolling_3h", "pm2_5_rolling_24h",
#     "temperature", "humidity", "wind_speed", "pressure", "precipitation",
# ]

# AQI_CATEGORIES = [
#     (0, 50, "Good", "#00e400"),
#     (51, 100, "Moderate", "#ffff00"),
#     (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
#     (151, 200, "Unhealthy", "#ff0000"),
#     (201, 300, "Very Unhealthy", "#8f3f97"),
#     (301, 500, "Hazardous", "#7e0023"),
# ]


# def get_aqi_category(aqi):
#     aqi = round(aqi)
#     for low, high, label, color in AQI_CATEGORIES:
#         if low <= aqi <= high:
#             return label, color
#     return "Unknown", "#888888"


# @st.cache_resource
# def connect_hopsworks():
#     project = hopsworks.login(
#         api_key_value=os.environ["HOPSWORKS_API_KEY"],
#         cert_folder="hopsworks_certs",
#     )
#     fs = project.get_feature_store()
#     mr = project.get_model_registry()
#     return project, fs, mr


# @st.cache_data(ttl=1800)  # refresh every 30 min
# def load_recent_features(_fs, hours=200):
#     fg = _fs.get_feature_group(name="aqi_lahore_features", version=2)
#     df = fg.read()
#     df = df.sort_values("timestamp").reset_index(drop=True)
#     return df.tail(hours).reset_index(drop=True)


# @st.cache_resource
# def load_models():
#     models = {}
#     for horizon in ["24h", "48h", "72h"]:
#         path = os.path.join(MODELS_DIR, f"aqi_model_{horizon}.pkl")
#         models[horizon] = joblib.load(path)
#     return models


# def main():
#     st.title("🌫️ Lahore AQI Forecast Dashboard")
#     st.caption("Real-time Air Quality Index tracking and 3-day forecast — Data science internship project")

#     with st.spinner("Connecting to feature store and loading models..."):
#         project, fs, mr = connect_hopsworks()
#         df = load_recent_features(fs)
#         models = load_models()

#     latest = df.iloc[-1]
#     current_aqi = latest["aqi_target"]
#     current_label, current_color = get_aqi_category(current_aqi)

#     # --- current AQI ---
#     st.subheader("Current Conditions")
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Current AQI", f"{current_aqi:.0f}", current_label)
#     col2.metric("PM2.5", f"{latest['pm2_5']:.1f} µg/m³")
#     col3.metric("Last updated", latest["timestamp"].strftime("%Y-%m-%d %H:%M UTC"))

#     if current_aqi > 150:
#         st.error(f"⚠️ HAZARD ALERT: Current AQI ({current_aqi:.0f}) is in the '{current_label}' range. Limit outdoor exposure.")
#     elif current_aqi > 100:
#         st.warning(f"⚠️ Current AQI ({current_aqi:.0f}) is '{current_label}' — sensitive groups should take precaution.")

#     # --- forecast ---
#     st.subheader("3-Day Forecast")
#     X_latest = latest[FEATURE_COLS].to_frame().T.astype(float)

#     forecast_cols = st.columns(3)
#     forecast_values = {}
#     for i, horizon in enumerate(["24h", "48h", "72h"]):
#         pred = models[horizon].predict(X_latest)[0]
#         forecast_values[horizon] = pred
#         label, color = get_aqi_category(pred)
#         with forecast_cols[i]:
#             st.metric(f"In {horizon}", f"{pred:.0f}", label)
#             if pred > 150:
#                 st.error(f"Hazardous conditions expected in {horizon}")
#             elif pred > 100:
#                 st.warning(f"Unhealthy air expected in {horizon}")

#     # --- historical trend chart ---
#     st.subheader("Recent AQI Trend")
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=df["timestamp"], y=df["aqi_target"], mode="lines", name="Historical AQI", line=dict(color="steelblue")))

#     future_times = [latest["timestamp"] + pd.Timedelta(hours=int(h[:-1])) for h in ["24h", "48h", "72h"]]
#     fig.add_trace(go.Scatter(x=future_times, y=list(forecast_values.values()), mode="markers+lines", name="Forecast", line=dict(color="orange", dash="dot"), marker=dict(size=10)))

#     fig.update_layout(xaxis_title="Time", yaxis_title="AQI", height=400)
#     st.plotly_chart(fig, use_container_width=True)

#     # --- SHAP feature importance ---
#     st.subheader("Feature Importance (SHAP) — 24h Model")
#     with st.spinner("Computing SHAP values..."):
#         explainer = shap.Explainer(models["24h"], df[FEATURE_COLS].tail(100))
#         shap_values = explainer(X_latest)

#         fig2, ax = plt.subplots(figsize=(8, 6))
#         shap.plots.bar(shap_values[0], show=False, ax=ax)
#         st.pyplot(fig2)

#     st.caption("Built with Streamlit, Hopsworks Feature Store & Model Registry, scikit-learn, and SHAP. Data: OpenWeather + Open-Meteo.")


# if __name__ == "__main__":
#     main()

"""
Streamlit dashboard - Simplified version that works with .env
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import hopsworks
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dotenv import load_dotenv

# Get the project root directory (parent of dashboard folder)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env from project root
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

# Also try loading from current directory if .env exists there
load_dotenv()

# Print debug info (remove after testing)
print(f"Project root: {project_root}")
print(f"Env file path: {env_path}")
print(f"Env file exists: {os.path.exists(env_path)}")
print(f"HOPSWORKS_API_KEY set: {bool(os.environ.get('HOPSWORKS_API_KEY'))}")

MODELS_DIR = os.path.join(project_root, "models")

st.set_page_config(page_title="Lahore AQI Forecast", page_icon="", layout="wide")

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

MIN_RECORDS_FOR_DISPLAY = 1


def get_aqi_category(aqi):
    aqi = round(aqi)
    for low, high, label, color in AQI_CATEGORIES:
        if low <= aqi <= high:
            return label, color
    return "Unknown", "#888888"


@st.cache_resource
def connect_hopsworks():
    try:
        api_key = os.environ.get("HOPSWORKS_API_KEY")
        
        if not api_key:
            st.error("HOPSWORKS_API_KEY not found in .env file")
            st.info(f"Please check that .env file exists at: {env_path}")
            return None, None, None
        
        project = hopsworks.login(
            api_key_value=api_key,
            cert_folder="hopsworks_certs",
        )
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        return project, fs, mr
    except Exception as e:
        st.error(f"Could not connect to Hopsworks: {e}")
        return None, None, None


@st.cache_data(ttl=1800)
def load_recent_features_safely(_fs, hours=200):
    try:
        fg = _fs.get_feature_group(name="aqi_lahore_features", version=2)
        
        try:
            query = fg.select_all()
            df = query.read()
            
            if df is None or len(df) == 0:
                return None, 0, "No data available yet"
            
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df.tail(hours).reset_index(drop=True), len(df), None
            
        except Exception as read_error:
            error_msg = str(read_error)
            if "Set changed size during iteration" in error_msg:
                return None, 0, "Feature group is empty. Data is being collected hourly..."
            else:
                return None, 0, f"Error reading data: {read_error}"
                
    except Exception as e:
        return None, 0, f"Could not access feature group: {e}"


@st.cache_resource
def load_models():
    models = {}
    model_paths = {
        "24h": os.path.join(MODELS_DIR, "aqi_model_24h.pkl"),
        "48h": os.path.join(MODELS_DIR, "aqi_model_48h.pkl"),
        "72h": os.path.join(MODELS_DIR, "aqi_model_72h.pkl"),
    }
    
    for horizon, path in model_paths.items():
        if os.path.exists(path):
            try:
                models[horizon] = joblib.load(path)
            except Exception as e:
                st.warning(f"Could not load model for {horizon}: {e}")
    
    return models


def show_waiting_state():
    st.title("Lahore AQI Forecast Dashboard")
    st.caption("Real-time Air Quality Index tracking and 3-day forecast")
    
    st.warning("Waiting for Data Collection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        The feature pipeline is collecting data hourly.
        
        Current Status:
        - Data collection in progress
        - Need at least 24 records for training
        - Check back later
        """)
    
    with col2:
        st.info("Data Collection Progress: 0/24 records (0%)")
        st.progress(0.0)
        st.caption("Estimated time: ~24 hours remaining")
    
    st.subheader("System Status")
    status_data = {
        "Component": ["Feature Pipeline", "Data Storage", "Model Training", "Dashboard"],
        "Status": ["Running", "Waiting for data", "Waiting for data", "Connected"],
    }
    status_df = pd.DataFrame(status_data)
    st.table(status_df)


def main():
    # Show debug info (remove after testing)
    st.sidebar.write("Debug Info:")
    st.sidebar.write(f"Project root: {project_root}")
    st.sidebar.write(f"Env file exists: {os.path.exists(env_path)}")
    st.sidebar.write(f"API Key set: {bool(os.environ.get('HOPSWORKS_API_KEY'))}")
    
    with st.spinner("Connecting to feature store..."):
        project, fs, mr = connect_hopsworks()
        
        if fs is None:
            st.error("Failed to connect to Hopsworks.")
            st.info("Make sure HOPSWORKS_API_KEY is set in your .env file")
            return
        
        df, record_count, error_msg = load_recent_features_safely(fs)
        
        if df is None or len(df) == 0:
            show_waiting_state()
            return
        
        models = load_models()
        
        if not models:
            st.warning("No trained models found. Training will start after 24+ hours of data.")
            show_partial_dashboard(df, record_count)
            return
        
        show_full_dashboard(df, record_count, models)


def show_partial_dashboard(df, record_count):
    st.title("Lahore AQI Forecast Dashboard")
    st.caption("Real-time Air Quality Index tracking and 3-day forecast")
    
    st.warning("Models are being trained. Forecasts will appear after 24+ hours of data.")
    
    latest = df.iloc[-1]
    current_aqi = latest["aqi_target"]
    current_label, current_color = get_aqi_category(current_aqi)
    
    st.subheader("Current Conditions")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current AQI", f"{current_aqi:.0f}", current_label)
    col2.metric("PM2.5", f"{latest['pm2_5']:.1f} µg/m")
    col3.metric("Last updated", latest["timestamp"].strftime("%Y-%m-%d %H:%M UTC"))
    
    if current_aqi > 150:
        st.error(f"HAZARD ALERT: Current AQI ({current_aqi:.0f}) is in the '{current_label}' range.")
    elif current_aqi > 100:
        st.warning(f"Current AQI ({current_aqi:.0f}) is '{current_label}' - sensitive groups should take precaution.")
    
    st.subheader("Recent AQI Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], 
        y=df["aqi_target"], 
        mode="lines", 
        name="Historical AQI", 
        line=dict(color="steelblue")
    ))
    fig.update_layout(xaxis_title="Time", yaxis_title="AQI", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption(f"Last updated: {latest['timestamp']}")
    st.caption(f"Total records: {record_count}")


def show_full_dashboard(df, record_count, models):
    st.title("Lahore AQI Forecast Dashboard")
    st.caption("Real-time Air Quality Index tracking and 3-day forecast")
    
    latest = df.iloc[-1]
    current_aqi = latest["aqi_target"]
    current_label, current_color = get_aqi_category(current_aqi)
    
    st.subheader("Current Conditions")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current AQI", f"{current_aqi:.0f}", current_label)
    col2.metric("PM2.5", f"{latest['pm2_5']:.1f} µg/m")
    col3.metric("Last updated", latest["timestamp"].strftime("%Y-%m-%d %H:%M UTC"))
    
    if current_aqi > 150:
        st.error(f"HAZARD ALERT: Current AQI ({current_aqi:.0f}) is in the '{current_label}' range.")
    elif current_aqi > 100:
        st.warning(f"Current AQI ({current_aqi:.0f}) is '{current_label}' - sensitive groups should take precaution.")
    
    st.subheader("3-Day Forecast")
    X_latest = latest[FEATURE_COLS].to_frame().T.astype(float)
    
    forecast_cols = st.columns(3)
    forecast_values = {}
    
    for i, horizon in enumerate(["24h", "48h", "72h"]):
        if horizon in models:
            pred = models[horizon].predict(X_latest)[0]
            forecast_values[horizon] = pred
            label, color = get_aqi_category(pred)
            with forecast_cols[i]:
                st.metric(f"In {horizon}", f"{pred:.0f}", label)
                if pred > 150:
                    st.error(f"Hazardous conditions expected in {horizon}")
                elif pred > 100:
                    st.warning(f"Unhealthy air expected in {horizon}")
        else:
            with forecast_cols[i]:
                st.metric(f"In {horizon}", "N/A", "Model not available")
    
    st.subheader("Recent AQI Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], 
        y=df["aqi_target"], 
        mode="lines", 
        name="Historical AQI", 
        line=dict(color="steelblue")
    ))
    
    if forecast_values:
        future_times = [latest["timestamp"] + pd.Timedelta(hours=int(h[:-1])) for h in ["24h", "48h", "72h"] if h in forecast_values]
        future_values = [forecast_values[h] for h in ["24h", "48h", "72h"] if h in forecast_values]
        fig.add_trace(go.Scatter(
            x=future_times, 
            y=future_values, 
            mode="markers+lines", 
            name="Forecast", 
            line=dict(color="orange", dash="dot"), 
            marker=dict(size=10)
        ))
    
    fig.update_layout(xaxis_title="Time", yaxis_title="AQI", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    if "24h" in models:
        st.subheader("Feature Importance (SHAP) - 24h Model")
        try:
            with st.spinner("Computing SHAP values..."):
                explainer = shap.Explainer(models["24h"], df[FEATURE_COLS].tail(100))
                shap_values = explainer(X_latest)
                fig2, ax = plt.subplots(figsize=(8, 6))
                shap.plots.bar(shap_values[0], show=False, ax=ax)
                st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Could not compute SHAP values: {e}")
    
    st.subheader("Pollutant Details")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CO", f"{latest['co']:.1f} µg/m")
        st.metric("NO2", f"{latest['no2']:.1f} µg/m")
        st.metric("O3", f"{latest['o3']:.1f} µg/m")
    with col2:
        st.metric("SO2", f"{latest['so2']:.1f} µg/m")
        st.metric("NH3", f"{latest['nh3']:.1f} µg/m")
        st.metric("NO", f"{latest['no']:.1f} µg/m")
    
    st.subheader("Weather Conditions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Temperature", f"{latest['temperature']:.1f}°C")
    with col2:
        st.metric("Humidity", f"{latest['humidity']:.1f}%")
    with col3:
        st.metric("Wind Speed", f"{latest['wind_speed']:.1f} km/h")
    with col4:
        st.metric("Pressure", f"{latest['pressure']:.0f} hPa")
    
    st.caption(f"Last updated: {latest['timestamp']}")
    st.caption(f"Total records: {record_count}")
    
    with st.expander("View Recent Data"):
        st.dataframe(df.tail(10))


if __name__ == "__main__":
    main()