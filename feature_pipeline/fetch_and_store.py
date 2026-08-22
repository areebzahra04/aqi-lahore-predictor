# """
# Feature Pipeline: fetches current pollutant + weather data for Lahore,
# computes accurate rolling/change-rate features using recent history,
# and inserts into the Hopsworks Feature Store.
# Designed to run hourly via GitHub Actions.
# """

# import os
# import pandas as pd
# import requests
# import hopsworks
# from datetime import datetime, timezone
# from dotenv import load_dotenv

# load_dotenv()

# LAT, LON = 31.5497, 74.3436

# OWM_API_KEY = os.environ["OWM_API_KEY"]
# HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]


# def fetch_current_pollution():
#     url = "http://api.openweathermap.org/data/2.5/air_pollution"
#     params = {"lat": LAT, "lon": LON, "appid": OWM_API_KEY}
#     r = requests.get(url, params=params)
#     r.raise_for_status()
#     return r.json()


# def fetch_current_weather():
#     url = "https://api.open-meteo.com/v1/forecast"
#     params = {
#         "latitude": LAT,
#         "longitude": LON,
#         "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation",
#         "timezone": "UTC",
#     }
#     r = requests.get(url, params=params)
#     r.raise_for_status()
#     return r.json()


# def calc_aqi_from_concentration(conc, breakpoints):
#     for (c_low, c_high, aqi_low, aqi_high) in breakpoints:
#         if c_low <= conc <= c_high:
#             return round(((aqi_high - aqi_low) / (c_high - c_low)) * (conc - c_low) + aqi_low)
#     return None


# PM25_BREAKPOINTS = [
#     (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
#     (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
#     (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500),
# ]
# PM10_BREAKPOINTS = [
#     (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
#     (255, 354, 151, 200), (355, 424, 201, 300),
#     (425, 504, 301, 400), (505, 604, 401, 500),
# ]


# def compute_epa_aqi(pm2_5, pm10):
#     aqi_pm25 = calc_aqi_from_concentration(pm2_5, PM25_BREAKPOINTS) if pm2_5 is not None else None
#     aqi_pm10 = calc_aqi_from_concentration(pm10, PM10_BREAKPOINTS) if pm10 is not None else None
#     candidates = [v for v in [aqi_pm25, aqi_pm10] if v is not None]
#     return max(candidates) if candidates else None


# def build_new_row():
#     pollution = fetch_current_pollution()
#     weather = fetch_current_weather()

#     record = pollution["list"][0]
#     comp = record["components"]
#     dt = datetime.fromtimestamp(record["dt"], tz=timezone.utc)

#     row = {
#         "timestamp": dt,
#         "aqi_owm": record["main"]["aqi"],
#         "co": comp.get("co"), "no": comp.get("no"), "no2": comp.get("no2"),
#         "o3": comp.get("o3"), "so2": comp.get("so2"),
#         "pm2_5": comp.get("pm2_5"), "pm10": comp.get("pm10"), "nh3": comp.get("nh3"),
#         "hour": dt.hour, "day": dt.day, "month": dt.month, "day_of_week": dt.weekday(),
#         "temperature": weather["current"]["temperature_2m"],
#         "humidity": weather["current"]["relative_humidity_2m"],
#         "wind_speed": weather["current"]["wind_speed_10m"],
#         "pressure": weather["current"]["surface_pressure"],
#         "precipitation": weather["current"]["precipitation"],
#     }
#     row["aqi_target"] = compute_epa_aqi(row["pm2_5"], row["pm10"])
#     return row


# def main():
#     project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY,cert_folder="hopsworks_certs",)
#     fs = project.get_feature_store()
#     fg = fs.get_or_create_feature_group(
#         name="aqi_lahore_features",
#         version=2,
#         primary_key=["timestamp"],
#         event_time="timestamp",
#         time_travel_format="HUDI",
#         online_enabled=False,
#     )

#     recent = fg.read()
#     recent = recent.sort_values("timestamp").reset_index(drop=True)
#     recent_tail = recent.tail(30).copy()

#     new_row = build_new_row()
#     print(f"Fetched row for {new_row['timestamp']}, AQI={new_row['aqi_target']}")

#     combined = pd.concat([recent_tail, pd.DataFrame([new_row])], ignore_index=True)
#     combined = combined.sort_values("timestamp").reset_index(drop=True)
#     combined = combined.drop_duplicates(subset="timestamp", keep="last")

#     combined["pm2_5_change_rate"] = combined["pm2_5"].diff()
#     combined["aqi_target_change_rate"] = combined["aqi_target"].diff()
#     combined["pm2_5_rolling_3h"] = combined["pm2_5"].rolling(window=3, min_periods=1).mean()
#     combined["pm2_5_rolling_24h"] = combined["pm2_5"].rolling(window=24, min_periods=1).mean()
#     combined = combined.bfill()

#     final_row = combined.tail(1)
#     fg.insert(final_row)
#     print("Inserted successfully:", final_row["timestamp"].iloc[0])


# if __name__ == "__main__":
#     main()
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


def ensure_float_types(df):
    """Convert integer columns to float where needed."""
    # Columns that should be float/double
    float_columns = [
        'pm2_5_change_rate', 
        'aqi_target_change_rate',
        'pm2_5_rolling_3h',
        'pm2_5_rolling_24h'
    ]
    
    for col in float_columns:
        if col in df.columns:
            df[col] = df[col].astype(float)
    
    return df


def main():
    try:
        print("=" * 60)
        print("Starting AQI Feature Pipeline")
        print("=" * 60)
        
        # Connect to Hopsworks
        print("\n1. Connecting to Hopsworks...")
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            cert_folder="hopsworks_certs"
        )
        fs = project.get_feature_store()
        print("✅ Connected to Hopsworks")
        
        # Get or create feature group
        print("\n2. Getting/Creating feature group...")
        try:
            fg = fs.get_feature_group(name="aqi_lahore_features", version=3)
            print(f"✅ Found existing feature group: {fg.name} v{fg.version}")
        except Exception as e:
            print(f"   Feature group not found, creating new one...")
            fg = fs.create_feature_group(
                name="aqi_lahore_features",
                version=3,
                primary_key=["timestamp"],
                event_time="timestamp",
                time_travel_format="HUDI",
                online_enabled=False,
            )
            print(f"✅ Created new feature group: {fg.name} v{fg.version}")
        
        # Build new data point
        print("\n3. Fetching current data from APIs...")
        new_row_dict = build_new_row()
        print(f"   ✅ Fetched data for {new_row_dict['timestamp']}")
        print(f"   AQI: {new_row_dict['aqi_target']}")
        print(f"   PM2.5: {new_row_dict['pm2_5']}")
        print(f"   PM10: {new_row_dict['pm10']}")
        
        # Create new row as DataFrame
        new_row_df = pd.DataFrame([new_row_dict])
        
        # Try to get existing data, but handle empty case gracefully
        print("\n4. Checking for existing data...")
        recent_tail = pd.DataFrame()
        try:
            print("   Attempting to read existing data (will handle if empty)...")
            try:
                query = fg.select_all()
                recent_data = query.read()
                if recent_data is not None and len(recent_data) > 0:
                    recent_tail = recent_data.sort_values("timestamp").reset_index(drop=True).tail(30).copy()
                    print(f"   ✅ Found {len(recent_data)} existing records, using last 30 for features")
                else:
                    print("   ℹ️ Feature group is empty - this will be the first record")
            except Exception as read_error:
                print(f"   ℹ️ No existing data found (empty feature group)")
                print(f"   This is normal for the first run.")
                
        except Exception as e:
            print(f"   ℹ️ Could not read existing data: {e}")
            print(f"   This is likely because the feature group is empty.")
        
        # Combine with existing data
        print("\n5. Processing features...")
        if len(recent_tail) > 0:
            combined = pd.concat([recent_tail, new_row_df], ignore_index=True)
            combined = combined.sort_values("timestamp").reset_index(drop=True)
            print(f"   Combined {len(recent_tail)} existing + 1 new record")
        else:
            print("   No existing data - this will be the first record")
            combined = new_row_df
        
        # Remove duplicates
        combined = combined.drop_duplicates(subset="timestamp", keep="last")
        
        # Compute rolling features
        # IMPORTANT: Use float values (0.0 instead of 0)
        if len(combined) > 1:
            combined["pm2_5_change_rate"] = combined["pm2_5"].diff().astype(float)
            combined["aqi_target_change_rate"] = combined["aqi_target"].diff().astype(float)
        else:
            # Use float values to match the schema
            combined["pm2_5_change_rate"] = 0.0
            combined["aqi_target_change_rate"] = 0.0
        
        # Use min() to handle cases where we have less data than window size
        window_3h = min(3, len(combined))
        window_24h = min(24, len(combined))
        
        combined["pm2_5_rolling_3h"] = combined["pm2_5"].rolling(window=window_3h, min_periods=1).mean().astype(float)
        combined["pm2_5_rolling_24h"] = combined["pm2_5"].rolling(window=window_24h, min_periods=1).mean().astype(float)
        
        # Fill NaN values
        combined = combined.bfill()
        combined = combined.fillna(0)
        
        # Ensure all numeric columns are proper float types
        combined = ensure_float_types(combined)
        
        # Get the final row to insert
        final_row = combined.tail(1)
        
        # Verify data types before insertion
        print("\n   Verifying data types...")
        for col in final_row.columns:
            if col in ['pm2_5_change_rate', 'aqi_target_change_rate', 'pm2_5_rolling_3h', 'pm2_5_rolling_24h']:
                print(f"   {col}: {final_row[col].dtype}")
                # Convert to float if needed
                if final_row[col].dtype in ['int64', 'bigint']:
                    final_row[col] = final_row[col].astype(float)
        
        # Insert into feature group
        print("\n6. Inserting into feature store...")
        fg.insert(final_row)
        print(f"✅ Successfully inserted record for {final_row['timestamp'].iloc[0]}")
        print(f"   AQI: {final_row['aqi_target'].iloc[0]}")
        print(f"   PM2.5: {final_row['pm2_5'].iloc[0]}")
        print(f"   PM2.5 Change Rate: {final_row['pm2_5_change_rate'].iloc[0]}")
        print(f"   PM2.5 Rolling 3h: {final_row['pm2_5_rolling_3h'].iloc[0]}")
        
        print("\n" + "=" * 60)
        print("✅ Feature pipeline completed successfully!")
        print("=" * 60)
        print("\n💡 Next steps:")
        print("   1. Run the training pipeline to train your model")
        print("   2. Check the Hopsworks UI to verify the data was inserted")
        print("   https://eu-west.cloud.hopsworks.ai:443/p/42106")
        
    except Exception as e:
        print(f"\n❌ Error in feature pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()