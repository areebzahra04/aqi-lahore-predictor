# """
# Export Training Data: reads features from the Hopsworks Feature Store,
# builds the same 24h/48h/72h targets used by train_model.py, and writes
# the result to a local CSV.

# Why this exists: TensorFlow and hopsworks/hsfs pin conflicting
# numpy/protobuf/grpcio versions, so they can't reliably live in the same
# Python environment. This script is the hand-off point — it runs in the
# SAME venv as train_model.py (the one with `hopsworks` installed) and
# never imports tensorflow. train_tf_model.py then picks up the CSV in a
# separate venv that never imports hopsworks.

# Run this in the existing project venv:
#     python training_pipeline/export_training_data.py
# """

# import os
# import hopsworks
# from dotenv import load_dotenv

# load_dotenv()

# HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]

# FEATURE_COLS = [
#     "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
#     "hour", "day", "month", "day_of_week",
#     "pm2_5_change_rate", "aqi_target_change_rate",
#     "pm2_5_rolling_3h", "pm2_5_rolling_24h",
#     "temperature", "humidity", "wind_speed", "pressure", "precipitation",
# ]

# TARGET_COLS = ["target_24h", "target_48h", "target_72h"]

# OUTPUT_PATH = "training_pipeline/tf_training_data.csv"


# def load_features(fs):
#     fg = fs.get_feature_group(name="aqi_lahore_features", version=2)
#     df = fg.read()
#     df = df.sort_values("timestamp").reset_index(drop=True)
#     return df


# def build_targets(df):
#     df = df.copy()
#     df["target_24h"] = df["aqi_target"].shift(-24)
#     df["target_48h"] = df["aqi_target"].shift(-48)
#     df["target_72h"] = df["aqi_target"].shift(-72)
#     df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
#     return df


# def main():
#     project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, cert_folder="hopsworks_certs")
#     fs = project.get_feature_store()

#     df = load_features(fs)
#     print(f"Loaded {len(df)} feature rows")

#     df = build_targets(df)
#     print(f"{len(df)} rows usable after building targets")

#     os.makedirs("training_pipeline", exist_ok=True)
#     df[FEATURE_COLS + TARGET_COLS].to_csv(OUTPUT_PATH, index=False)
#     print(f"Exported {len(df)} rows -> {OUTPUT_PATH}")


# if __name__ == "__main__":
#     main()
"""
Export Training Data: reads features from the Hopsworks Feature Store,
builds the same 24h/48h/72h targets used by train_model.py, and writes
the result to a local CSV.

Why this exists: TensorFlow and hopsworks/hsfs pin conflicting
numpy/protobuf/grpcio versions, so they can't reliably live in the same
Python environment. This script is the hand-off point - it runs in the
SAME venv as train_model.py (the one with `hopsworks` installed) and
never imports tensorflow. train_tf_model.py then picks up the CSV in a
separate venv that never imports hopsworks.

Run this in the existing project venv:
    python training_pipeline/export_training_data.py
"""

import os
import hopsworks
import pandas as pd
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
MIN_RECORDS = 24


def check_feature_group_has_data(fs):
    """
    Check if the feature group exists and has data without triggering
    the "Set changed size during iteration" error for empty groups.
    """
    try:
        fg = fs.get_feature_group(name="aqi_lahore_features", version=2)
        print(f"Found feature group: {fg.name} v{fg.version}")
        
        # Try a different approach - use the get_feature_group API to check
        # if there's any data by attempting to get the schema
        if fg.schema is not None:
            print("Feature group schema exists")
            
            # Try to read just the first row with a limit
            # Some versions support limit parameter differently
            try:
                # Try using head() or limit() if available
                if hasattr(fg, 'head'):
                    sample = fg.head(1)
                    if sample is not None and len(sample) > 0:
                        return True, fg, len(sample)
                else:
                    # Try a different approach - use the statistics
                    try:
                        stats = fg.statistics
                        if stats is not None:
                            return True, fg, 1
                    except:
                        pass
            except:
                pass
            
            # If we can't determine, assume it has data but we'll find out when reading
            return True, fg, -1
            
    except Exception as e:
        print(f"Could not access feature group: {e}")
        return False, None, 0
    
    return False, None, 0


def load_features_safely(fs):
    """
    Safely load features from feature group with proper error handling
    for empty or insufficient data.
    """
    try:
        print("Attempting to load features from Hopsworks...")
        
        # First check if feature group exists and try to get data without reading
        exists, fg, estimated_count = check_feature_group_has_data(fs)
        
        if not exists:
            print("Feature group does not exist or is not accessible.")
            return None, 0
        
        # Try to read with error handling
        try:
            query = fg.select_all()
            df = query.read()
            
            if df is None or len(df) == 0:
                print("Feature group is EMPTY (no records found)")
                return None, 0
            
            print(f"Successfully loaded {len(df)} records")
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df, len(df)
            
        except Exception as read_error:
            error_msg = str(read_error)
            if "Set changed size during iteration" in error_msg:
                print("Feature group exists but appears to be EMPTY")
                print("This is normal when the feature group has no data yet.")
            else:
                print(f"Error reading data: {read_error}")
            return None, 0
        
    except Exception as e:
        print(f"Could not read data: {e}")
        if "Set changed size during iteration" in str(e):
            print("Feature group is likely empty.")
        return None, 0


def build_targets(df):
    df = df.copy()
    df["target_24h"] = df["aqi_target"].shift(-24)
    df["target_48h"] = df["aqi_target"].shift(-48)
    df["target_72h"] = df["aqi_target"].shift(-72)
    df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
    return df


def main():
    print("=" * 60)
    print("Exporting Training Data for TensorFlow")
    print("=" * 60)
    
    try:
        print("\n1. Connecting to Hopsworks...")
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            cert_folder="hopsworks_certs"
        )
        fs = project.get_feature_store()
        print("Connected to Hopsworks")
        
        print("\n2. Loading features...")
        df, record_count = load_features_safely(fs)
        
        if df is None or record_count == 0:
            print("\n" + "=" * 60)
            print("NO DATA AVAILABLE")
            print("Please wait for the feature pipeline to collect data.")
            print("The feature pipeline runs hourly.")
            print("")
            print("Current status:")
            print(f"  - Records needed: {MIN_RECORDS}")
            print(f"  - Records available: {record_count}")
            print("")
            print("After the feature pipeline has run for 24+ hours,")
            print("run this script again to export the data.")
            print("=" * 60)
            return
        
        if record_count < MIN_RECORDS:
            print("\n" + "=" * 60)
            print("INSUFFICIENT DATA")
            print(f"Current records: {record_count}")
            print(f"Need at least: {MIN_RECORDS} records")
            print(f"Remaining: {MIN_RECORDS - record_count} records")
            print(f"Estimated time: ~{MIN_RECORDS - record_count} more hours")
            print("")
            print("Please wait for the feature pipeline to collect more data.")
            print("Run this script again after collecting more data.")
            print("=" * 60)
            return
        
        print(f"\n3. Building targets for 24h, 48h, 72h horizons...")
        df = build_targets(df)
        print(f"{len(df)} rows usable after building targets")
        
        if len(df) == 0:
            print("Not enough data to build targets. Need more records.")
            return
        
        print("\n4. Exporting to CSV...")
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        export_df = df[FEATURE_COLS + TARGET_COLS]
        export_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Exported {len(export_df)} rows -> {OUTPUT_PATH}")
        print(f"File shape: {export_df.shape}")
        
        print("\n" + "=" * 60)
        print("Export completed successfully")
        print("You can now run: python training_pipeline/train_tf_model.py")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError in export: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()