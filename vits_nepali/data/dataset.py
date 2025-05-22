# # # Placeholder file
# # # data/dataset.py
# # from torch.utils.data import Dataset, DataLoader
# # from typing import Tuple, List
# # import torch
# # from .preprocess import audio_to_mel, preprocess_data
# # from utils.text import text_to_phonemes
# # import logging

# # logger = logging.getLogger(__name__)

# # class VITSDataset(Dataset):
# #     def __init__(self, data_dir: str, manifest_file: str, sample_rate: int = 16000, n_mels: int = 80):
# #         self.data = preprocess_data(data_dir, manifest_file)
# #         self.sample_rate = sample_rate
# #         self.n_mels = n_mels

# #     def __len__(self) -> int:
# #         return len(self.data)

# #     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
# #         audio_path, text = self.data[idx]
# #         try:
# #             phonemes = torch.tensor(text_to_phonemes(text), dtype=torch.long)
# #             mel = audio_to_mel(audio_path, self.sample_rate, self.n_mels)
# #             return phonemes, mel
# #         except Exception as e:
# #             logger.error(f"Error loading item {idx} ({audio_path}): {str(e)}")
# #             raise

# # def get_dataloader(dataset: VITSDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
# #     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
# import torch
# import torchaudio
# import csv
# from pathlib import Path
# from typing import Tuple
# from utils.text import text_to_phonemes

# class VITSDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir: str, manifest_file: str):
#         self.data_dir = Path(data_dir)
#         self.audio_files = []
#         self.texts = []
#         self.phonemes = []  # Store precomputed phoneme indices
        
#         with open(manifest_file, 'r', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             if 'path' not in reader.fieldnames or 'labels' not in reader.fieldnames:
#                 raise ValueError(f"CSV {manifest_file} must have 'path' and 'labels' columns")
#             for row in reader:
#                 self.audio_files.append(row['path'])
#                 self.texts.append(row['labels'])
#                 # Check for phonemes column
#                 if 'phonemes' in row and row['phonemes']:
#                     self.phonemes.append(row['phonemes'])  # Store phoneme string
#                 else:
#                     self.phonemes.append(None)
        
#         self.transform = torchaudio.transforms.MelSpectrogram(
#             sample_rate=16000, n_mels=80, hop_length=256
#         )
    
#     def __len__(self) -> int:
#         return len(self.audio_files)
    
#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         audio_path = self.data_dir / self.audio_files[idx]
#         text = self.texts[idx]
#         phoneme_str = self.phonemes[idx]
        
#         # Load audio and compute mel-spectrogram
#         waveform, sample_rate = torchaudio.load(audio_path)
#         if sample_rate != 16000:
#             waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
#         mel = self.transform(waveform).squeeze(0).transpose(0, 1)  # (T, n_mels)
        
#         # Use precomputed phonemes if available, else compute on-the-fly
#         phonemes = torch.tensor(text_to_phonemes(phoneme_str or text), dtype=torch.long)
        
#         return phonemes, mel

import torch
import torchaudio
import csv
from pathlib import Path
from typing import Tuple
from utils.text import text_to_phonemes, phoneme_vocab  # Make sure phoneme_vocab is defined in text.py

class VITSDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, manifest_file: str):
        self.data_dir = Path(data_dir)
        self.audio_files = []
        self.texts = []
        self.phonemes = []

        with open(manifest_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'path' not in reader.fieldnames or 'labels' not in reader.fieldnames:
                raise ValueError(f"CSV {manifest_file} must have 'path' and 'labels' columns")

            for row in reader:
                self.audio_files.append(row['path'])
                self.texts.append(row['labels'])

                # Check if phonemes column exists and is valid
                if 'phonemes' in row and row['phonemes']:
                    phoneme_str = row['phonemes']
                    phoneme_list = phoneme_str.split()
                    phoneme_indices = [phoneme_vocab.get(p, 0) for p in phoneme_list]
                    self.phonemes.append(phoneme_indices)
                else:
                    self.phonemes.append(None)

        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=80, hop_length=256
        )

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_path = self.data_dir / self.audio_files[idx]
        text = self.texts[idx]
        precomputed_phonemes = self.phonemes[idx]

        # Load audio and compute mel-spectrogram
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        mel = self.transform(waveform).squeeze(0).transpose(0, 1)  # (T, n_mels)

        # Convert phonemes to tensor
        if precomputed_phonemes is not None:
            phonemes = torch.tensor(precomputed_phonemes, dtype=torch.long)
        else:
            phonemes = torch.tensor(text_to_phonemes(text), dtype=torch.long)

        return phonemes, mel
