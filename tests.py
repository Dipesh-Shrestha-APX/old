#####################
# Text to phonemes module test
from vits_nepali.utils.text import text_to_phonemes
text = "नमस्ते"
indices = text_to_phonemes(text)
print(f"Text: {text}, Indices: {indices}")  # Expected: [29, 34, 41, 25, 6]

#####################
# VITS module test
from vits_nepali.data.dataset import VITSDataset
dataset = VITSDataset("vits_nepali/data/dataset/", "vits_nepali/data/csv/train_phonemes.csv")
phonemes, mel = dataset[0]
print(f"Phonemes: {phonemes}")  # Expected: tensor([29, 34, 41, 25, 6], dtype=torch.long)
print(f"Mel shape: {mel.shape}")  # Expected: torch.Size([80, T]) for 80 mel bins

#####################
# Pipeline init test
from vits_nepali.pipeline.training_pipeline import TrainingPipeline
pipeline = TrainingPipeline("vits_nepali/configs/config.yaml")
print("Pipeline initialized successfully")

#####################
# Audio path verification
import csv
import os
from pathlib import Path

def check_audio_paths(manifest_file: str, data_dir: str = "vits_nepali/data/dataset/"):
    data_dir = Path(data_dir)
    missing_files = []
    with open(manifest_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = data_dir / row['path']
            if not audio_path.exists():
                missing_files.append(str(audio_path))
    return missing_files

# Test all CSVs
csvs = ["data/csv/train_phonemes.csv", "data/csv/val_phonemes.csv", "data/csv/test_phonemes.csv"]
for csv_file in csvs:
    missing = check_audio_paths("vits_nepali/" + csv_file)
    if missing:
        print(f"Missing files in {csv_file}: {missing}")
    else:
        print(f"All audio files found for {csv_file}")