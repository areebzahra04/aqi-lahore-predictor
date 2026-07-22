"""
Feature Pipeline: fetches current pollutant + weather data for Lahore,
computes accurate rolling/change-rate features using recent history,
and inserts into the Hopsworks Feature Store.
Designed to run hourly via GitHub Actions.
"""

import os
import pandas as pd
import requests
import hopsworks
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

LAT, LON = 31.5497, 74.3436

OWM_API_KEY = os.environ["OWM_API_KEY"]
HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]


def fetch_current_pollution():
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": LAT, "lon": LON, "appid": OWM_API_KEY}
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()


def fetch_current_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation",
        "timezone": "UTC",
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()


def calc_aqi_from_concentration(conc, breakpoints):
    for (c_low, c_high, aqi_low, aqi_high) in breakpoints:
        if c_low <= conc <= c_high:
            return round(((aqi_high - aqi_low) / (c_high - c_low)) * (conc - c_low) + aqi_low)
    return None


PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500),
]
PM10_BREAKPOINTS = [
    (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
    (255, 354, 151, 200), (355, 424, 201, 300),
    (425, 504, 301, 400), (505, 604, 401, 500),
]


def compute_epa_aqi(pm2_5, pm10):
    aqi_pm25 = calc_aqi_from_concentration(pm2_5, PM25_BREAKPOINTS) if pm2_5 is not None else None
    aqi_pm10 = calc_aqi_from_concentration(pm10, PM10_BREAKPOINTS) if pm10 is not None else None
    candidates = [v for v in [aqi_pm25, aqi_pm10] if v is not None]
    return max(candidates) if candidates else None


def build_new_row():
    pollution = fetch_current_pollution()
    weather = fetch_current_weather()

    record = pollution["list"][0]
    comp = record["components"]
    dt = datetime.fromtimestamp(record["dt"], tz=timezone.utc)

    row = {
        "timestamp": dt,
        "aqi_owm": record["main"]["aqi"],
        "co": comp.get("co"), "no": comp.get("no"), "no2": comp.get("no2"),
        "o3": comp.get("o3"), "so2": comp.get("so2"),
        "pm2_5": comp.get("pm2_5"), "pm10": comp.get("pm10"), "nh3": comp.get("nh3"),
        "hour": dt.hour, "day": dt.day, "month": dt.month, "day_of_week": dt.weekday(),
        "temperature": weather["current"]["temperature_2m"],
        "humidity": weather["current"]["relative_humidity_2m"],
        "wind_speed": weather["current"]["wind_speed_10m"],
        "pressure": weather["current"]["surface_pressure"],
        "precipitation": weather["current"]["precipitation"],
    }
    row["aqi_target"] = compute_epa_aqi(row["pm2_5"], row["pm10"])
    return row


def main():
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY,cert_folder="hopsworks_certs",)
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_lahore_features",
        version=2,
        primary_key=["timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
        online_enabled=False,
    )

    recent = fg.read()
    recent = recent.sort_values("timestamp").reset_index(drop=True)
    recent_tail = recent.tail(30).copy()

    new_row = build_new_row()
    print(f"Fetched row for {new_row['timestamp']}, AQI={new_row['aqi_target']}")

    combined = pd.concat([recent_tail, pd.DataFrame([new_row])], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined = combined.drop_duplicates(subset="timestamp", keep="last")

    combined["pm2_5_change_rate"] = combined["pm2_5"].diff()
    combined["aqi_target_change_rate"] = combined["aqi_target"].diff()
    combined["pm2_5_rolling_3h"] = combined["pm2_5"].rolling(window=3, min_periods=1).mean()
    combined["pm2_5_rolling_24h"] = combined["pm2_5"].rolling(window=24, min_periods=1).mean()
    combined = combined.bfill()

    final_row = combined.tail(1)
    fg.insert(final_row)
    print("Inserted successfully:", final_row["timestamp"].iloc[0])


if __name__ == "__main__":
    main()
