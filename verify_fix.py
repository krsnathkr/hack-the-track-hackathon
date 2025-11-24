from src.data_processor import load_and_pivot_data
import pandas as pd

try:
    print("Loading data...")
    # Load demo data to test
    df = load_and_pivot_data("data/demo_data.csv")
    
    print("Checking columns...")
    if 'VBOX_Long_Min' in df.columns and 'VBOX_Lat_Min' in df.columns:
        print("SUCCESS: VBOX columns found.")
        print(df[['VBOX_Long_Min', 'VBOX_Lat_Min']].head())
    else:
        print("FAILURE: VBOX columns NOT found.")
        print("Columns:", df.columns)
        
    print("Checking numeric types...")
    if pd.api.types.is_numeric_dtype(df['VBOX_Long_Min']):
        print("SUCCESS: VBOX_Long_Min is numeric.")
    else:
        print("FAILURE: VBOX_Long_Min is NOT numeric.")
        
except Exception as e:
    print(f"FAILURE: Exception occurred: {e}")
