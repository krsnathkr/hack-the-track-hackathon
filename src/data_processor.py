import pandas as pd
import numpy as np

def load_and_pivot_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        # Low_memory=False to prevent type inference errors on large files
        df = pd.read_csv(filepath, low_memory=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

    # Filter for only the columns we actually need to save memory/speed
    needed_params = [
        'Speed', 'speed', 'accx_can', 'accy_can', 'aps', 'pbrake_f', 'pbrake_r',
        'Steering_Angle', 'Laptrigger_lapdist_dls', 
        'VBOX_Long_Minutes', 'VBOX_Lat_Min', 'VBOX_Long_Min' # Added VBOX_Long_Min just in case
    ]
    df = df[df['telemetry_name'].isin(needed_params)]

    # Standardize VBOX_Long_Minutes -> VBOX_Long_Min EARLY
    df['telemetry_name'] = df['telemetry_name'].replace({'VBOX_Long_Minutes': 'VBOX_Long_Min'})

    print("Pivoting data...")
    # Pivot to wide format
    df_pivot = df.pivot_table(
        index=['timestamp', 'vehicle_id'], 
        columns='telemetry_name', 
        values='telemetry_value',
        aggfunc='first' 
    ).reset_index()

    # Convert timestamp
    df_pivot['timestamp'] = pd.to_datetime(df_pivot['timestamp'])
    df_pivot = df_pivot.sort_values(['vehicle_id', 'timestamp'])

    print("Handling missing values...")
    # Fix: Apply ffill/bfill within each vehicle group to prevent data leakage
    df_pivot = df_pivot.groupby('vehicle_id').apply(lambda x: x.ffill().bfill()).reset_index(drop=True)

# --- CRITICAL FIX: GENERATE LAP NUMBERS ---
    print("Generating Lap Numbers...")
    
    # 1. Calculate the diff within each group
    df_pivot['dist_diff'] = df_pivot.groupby('vehicle_id')['Laptrigger_lapdist_dls'].diff()
    
    # 2. Define the logic to identify a reset
    def count_laps(series):
        return (series < -1000).cumsum() + 1

    # 3. Use TRANSFORM (not apply) to ensure the index matches the original dataframe
    df_pivot['Lap_Number'] = df_pivot.groupby('vehicle_id')['dist_diff'].transform(count_laps)
    
    # Fill NaN laps (first row of each car) with 1
    df_pivot['Lap_Number'] = df_pivot['Lap_Number'].fillna(1).astype(int)
    
    # Drop the temp column
    df_pivot = df_pivot.drop(columns=['dist_diff'])

    # Ensure numeric types
    cols_to_numeric = ['Speed', 'accx_can', 'accy_can', 'aps', 'pbrake_f', 'pbrake_r', 'Steering_Angle', 'Laptrigger_lapdist_dls', 'VBOX_Long_Min', 'VBOX_Lat_Min']
    for col in cols_to_numeric:
        if col in df_pivot.columns:
            df_pivot[col] = pd.to_numeric(df_pivot[col], errors='coerce')

    # Standardize 'Speed' column name (handle case sensitivity)
    if 'speed' in df_pivot.columns and 'Speed' not in df_pivot.columns:
        df_pivot = df_pivot.rename(columns={'speed': 'Speed'})

    print(f"Data processed. Shape: {df_pivot.shape}")
    return df_pivot