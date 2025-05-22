# Placeholder file
# data/preprocess.py
import torch
import torchaudio
import numpy as np
from typing import List, Tuple
# from utils.text import text_to_phonemes
import logging
import os
import csv
import random
import shutil

logger = logging.getLogger(__name__)

def audio_to_mel(audio_path: str, sample_rate: int = 16000, n_mels: int = 80) -> torch.Tensor:
    """Convert audio file to log-mel spectrogram."""
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        waveform, sr = torchaudio.load(audio_path)
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=256, f_min=0, f_max=8000
        )
        mel = mel_transform(waveform)
        mel = torch.log(mel + 1e-5)
        return mel.squeeze(0)
    except Exception as e:
        logger.error(f"Failed to process audio {audio_path}: {str(e)}")
        raise

def preprocess_data(data_dir: str, manifest_file: str) -> List[Tuple[str, str]]:
    """Load and validate manifest CSV file from data/csv/."""
    try:
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        if not os.path.exists(manifest_file):
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
        data = []
        with open(manifest_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header row (path,labels)
            if header != ['path', 'labels']:
                raise ValueError(f"Invalid CSV header: expected ['path', 'labels'], got {header}")
            for row in reader:
                if len(row) != 2:
                    raise ValueError(f"Invalid CSV row: expected 2 columns, got {row}")
                audio, text = row
                audio_path = os.path.join(data_dir, audio.strip())
                text = text.strip()
                if not (audio_path.endswith('.wav') and len(text) > 0):
                    raise ValueError(f"Invalid entry: {audio_path},{text}")
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                data.append((audio_path, text))
        return data
    except Exception as e:
        logger.error(f"Failed to preprocess data from {manifest_file}: {str(e)}")
        raise

# def split_manifest(manifest_file: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> None:
#     """Split manifest CSV and move audio files to data/dataset/{train,val,test}/."""
#     try:
#         csv_dir = os.path.dirname(manifest_file)  # e.g., data/csv/
#         print(csv_dir)
#         dataset_dir = os.path.normpath(os.path.join(os.path.dirname(csv_dir), 'dataset'))
#         print(dataset_dir)

#         audio_dir = os.path.normpath(os.path.join(dataset_dir, 'audio'))
#         print(audio_dir)

#         if not os.path.exists(csv_dir):
#             raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
#         if not os.path.exists(dataset_dir):
#             raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
#         if not os.path.exists(audio_dir):
#             raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
#         if not os.path.exists(manifest_file):
#             raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

#         # Create subdirectories for audio splits
#         for split in ['train', 'val', 'test']:
#             split_dir = os.path.join(dataset_dir, split)
#             os.makedirs(split_dir, exist_ok=True)

#         with open(manifest_file, 'r', encoding='utf-8') as f:
#             reader = csv.reader(f)
#             header = next(reader)  # Skip header (path,labels)
#             if header != ['path', 'labels']:
#                 raise ValueError(f"Invalid CSV header: expected ['path', 'labels'], got {header}")
#             rows = list(reader)  # Read all data rows

#         random.seed(42)  # For reproducibility
#         random.shuffle(rows)
#         n = len(rows)
#         train_end = int(n * train_ratio)
#         val_end = train_end + int(n * val_ratio)

#         # Process each split
#         for split_name, data, start, end in [
#             ('train', rows[:train_end], 0, train_end),
#             ('val', rows[train_end:val_end], train_end, val_end),
#             ('test', rows[val_end:], val_end, n)
#         ]:
#             split_csv = os.path.join(csv_dir, f'{split_name}.csv')
#             split_dataset_dir = os.path.join(dataset_dir, split_name)
#             split_rows = []
#             for audio, text in data:
#                 src_path = os.path.join(audio_dir, audio.strip())
#                 if not os.path.exists(src_path):
#                     raise FileNotFoundError(f"Audio file not found: {src_path}")
#                 dst_path = os.path.join(split_dataset_dir, audio.strip())
#                 # Move audio file to split directory
#                 shutil.move(src_path, dst_path)
#                 # Update CSV to reference audio in dataset/{split}/
#                 split_rows.append([os.path.join(split_name, audio.strip()), text.strip()])
#             # Write CSV
#             with open(split_csv, 'w', encoding='utf-8', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(['path', 'labels'])  # Write header
#                 writer.writerows(split_rows)

#         logger.info(f"Split manifest: {train_end} train, {val_end - train_end} val, {n - val_end} test")
#         logger.info(f"Original manifest.csv at {manifest_file} remains unchanged")
#     except Exception as e:
#         logger.error(f"Failed to split manifest: {str(e)}")
#         raise


# def split_manifest(manifest_file: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> None:
#     """Split manifest CSV and move audio files to data/dataset/{train,val,test}/."""
#     try:
#         csv_dir = os.path.dirname(manifest_file)  # e.g., data/csv/
#         print("CSV dir:", csv_dir)

#         dataset_dir = os.path.normpath(os.path.join(os.path.dirname(csv_dir), 'dataset'))
#         print("Dataset dir:", dataset_dir)

#         audio_dir = os.path.normpath(os.path.join(dataset_dir, 'audio'))
#         print("Audio dir:", audio_dir)

#         if not os.path.exists(csv_dir):
#             raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
#         if not os.path.exists(dataset_dir):
#             raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
#         if not os.path.exists(audio_dir):
#             raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
#         if not os.path.exists(manifest_file):
#             raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

#         # Create subdirectories for audio splits
#         for split in ['train', 'val', 'test']:
#             split_dir = os.path.join(dataset_dir, split)
#             os.makedirs(split_dir, exist_ok=True)

#         with open(manifest_file, 'r', encoding='utf-8') as f:
#             reader = csv.reader(f)
#             header = next(reader)  # Skip header (path,labels)
#             if header != ['path', 'labels']:
#                 raise ValueError(f"Invalid CSV header: expected ['path', 'labels'], got {header}")
#             rows = list(reader)  # Read all data rows

#         random.seed(42)  # For reproducibility
#         random.shuffle(rows)
#         n = len(rows)
#         train_end = int(n * train_ratio)
#         val_end = train_end + int(n * val_ratio)

#         # Process each split
#         for split_name, data, start, end in [
#             ('train', rows[:train_end], 0, train_end),
#             ('val', rows[train_end:val_end], train_end, val_end),
#             ('test', rows[val_end:], val_end, n)
#         ]:
#             split_csv = os.path.join(csv_dir, f'{split_name}.csv')
#             split_dataset_dir = os.path.join(dataset_dir, split_name)
#             split_rows = []

#             for audio, text in data:
#                 audio_name = audio.strip()
#                 if not audio_name.lower().endswith('.wav'):
#                     audio_name += '.wav'

#                 src_path = os.path.join(audio_dir, audio_name)
#                 if not os.path.exists(src_path):
#                     raise FileNotFoundError(f"Audio file not found: {src_path}")

#                 dst_path = os.path.join(split_dataset_dir, audio_name)
#                 shutil.move(src_path, dst_path)

#                 split_rows.append([os.path.join(split_name, audio_name), text.strip()])

#             # Write split CSV
#             with open(split_csv, 'w', encoding='utf-8', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(['path', 'labels'])  # Write header
#                 writer.writerows(split_rows)

#         logger.info(f"Split manifest: {train_end} train, {val_end - train_end} val, {n - val_end} test")
#         logger.info(f"Original manifest.csv at {manifest_file} remains unchanged")

#     except Exception as e:
#         logger.error(f"Failed to split manifest: {str(e)}")
#         raise

logging.basicConfig(level=logging.INFO)

def split_manifest(manifest_file: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> None:
    """Split manifest CSV and move audio files to data/dataset/{train,val,test}/."""
    try:
        csv_dir = os.path.dirname(manifest_file)
        print("CSV dir:", csv_dir)

        dataset_dir = os.path.normpath(os.path.join(os.path.dirname(csv_dir), 'dataset'))
        print("Dataset dir:", dataset_dir)

        audio_dir = os.path.normpath(os.path.join(dataset_dir, 'audio'))
        print("Audio dir:", audio_dir)

        if not os.path.exists(csv_dir):
            raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        if not os.path.exists(audio_dir):
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        if not os.path.exists(manifest_file):
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(dataset_dir, split)
            os.makedirs(split_dir, exist_ok=True)

        with open(manifest_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            if header != ['path', 'labels']:
                raise ValueError(f"Invalid CSV header: expected ['path', 'labels'], got {header}")
            rows = list(reader)

        random.seed(42)
        random.shuffle(rows)
        n = len(rows)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        for split_name, data in [
            ('train', rows[:train_end]),
            ('val', rows[train_end:val_end]),
            ('test', rows[val_end:])
        ]:
            split_csv = os.path.join(csv_dir, f'{split_name}.csv')
            split_dataset_dir = os.path.join(dataset_dir, split_name)
            split_rows = []

            for audio, text in data:
                audio_name = audio.strip()
                if not audio_name.lower().endswith('.wav'):
                    audio_name += '.wav'

                src_path = os.path.join(audio_dir, audio_name)
                if not os.path.exists(src_path):
                    raise FileNotFoundError(f"Audio file not found: {src_path}")

                dst_path = os.path.join(split_dataset_dir, audio_name)
                shutil.move(src_path, dst_path)

                split_rows.append([os.path.join(split_name, audio_name), text.strip()])

            with open(split_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['path', 'labels'])
                writer.writerows(split_rows)

        logger.info(f"Split manifest: {train_end} train, {val_end - train_end} val, {n - val_end} test")
        logger.info(f"Original manifest.csv at {manifest_file} remains unchanged")

    except Exception as e:
        logger.error(f"Failed to split manifest: {str(e)}")
        raise
