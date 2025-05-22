import os
import pandas as pd

def find_missing_audio_files(csv_path, folder_path, extension=".wav"):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Check if the 'path' column exists
    if 'path' not in df.columns:
        raise ValueError("CSV must contain a 'path' column")

    # Extract audio names and form expected filenames
    expected_files = [f"{name}{extension}" for name in df['path']]

    # List all files in the folder
    existing_files = set(os.listdir(folder_path))

    # Find missing files
    missing_files = [file for file in expected_files if file not in existing_files]

    # Output results
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(file)
    else:
        print("All files are present.")

# Example usage:
find_missing_audio_files("vits_nepali/data/csv/manifest.csv", "vits_nepali/data/dataset/audio")
