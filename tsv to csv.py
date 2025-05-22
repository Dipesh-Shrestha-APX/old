import pandas as pd
import os

def convert_tsv_to_csv():
    # Hardcoded paths
    tsv_file1 = "vits-nepali/data/csv/MaleVoice.tsv"
    tsv_file2 = "vits-nepali/data/csv/FemaleVoice.tsv"
    output_csv = "vits-nepali/data/csv/manifest.csv"
    
    try:
        # Check if input files exist
        if not os.path.exists(tsv_file1):
            raise FileNotFoundError(f"Input file {tsv_file1} does not exist")
        if not os.path.exists(tsv_file2):
            raise FileNotFoundError(f"Input file {tsv_file2} does not exist")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Read the TSV files
        df1 = pd.read_csv(tsv_file1, sep='\t', names=['audio_id', 'sentence'], encoding='utf-8')
        df2 = pd.read_csv(tsv_file2, sep='\t', names=['audio_id', 'sentence'], encoding='utf-8')
        
        # Combine the dataframes
        combined_df = pd.concat([df1, df2], ignore_index=True)
        
        # Rename columns to match desired CSV structure
        combined_df = combined_df.rename(columns={'audio_id': 'path', 'sentence': 'labels'})
        
        # Write to CSV
        combined_df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"Successfully created {output_csv}")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse TSV files. Check file format: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    convert_tsv_to_csv()