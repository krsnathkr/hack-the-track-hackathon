import os
import pandas as pd

def explore_data(root_dir):
    """
    Walks through a directory, reads each CSV file into a pandas DataFrame,
    and prints some information about it.
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv') or file.endswith('.CSV'):
                file_path = os.path.join(subdir, file)
                print(f"--- Information for {file_path} ---")
                try:
                    # Attempt to read with utf-8, then fall back to latin1 if it fails
                    try:
                        df = pd.read_csv(file_path, on_bad_lines='skip')
                    except UnicodeDecodeError:
                        df = pd.read_csv(file_path, on_bad_lines='skip', encoding='latin1')

                    print("Shape:", df.shape)
                    print("Head:")
                    print(df.head())
                    print("Info:")
                    df.info()
                    print("Descriptive Statistics:")
                    print(df.describe())
                except Exception as e:
                    print(f"Could not read file {file_path}: {e}")
                print("\n" * 2)

if __name__ == "__main__":
    data_directory = "data"
    if os.path.exists(data_directory):
        explore_data(data_directory)
    else:
        print(f"Directory '{data_directory}' not found.")
