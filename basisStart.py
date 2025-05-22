import os
from vits_nepali.data.preprocess import split_manifest

def main():
    # Define the path to manifest.csv relative to the working directory (old/)
    manifest_path = "vits_nepali/data/csv/manifest.csv"
    
    try:
        # Check if the manifest file exists
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found at: {manifest_path}")
        
        # Call split_manifest with default ratios (train=0.8, val=0.1, test=0.1)
        split_manifest(manifest_path, train_ratio=0.8, val_ratio=0.1)
        print("Successfully split the manifest and moved audio files.")
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()