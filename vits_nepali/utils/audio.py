# utils/audio.py
import torchaudio
import torch
import logging
import os

logger = logging.getLogger(__name__)

def save_audio(audio: torch.Tensor, path: str, sample_rate: int = 16000) -> None:
    """Save audio tensor to file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchaudio.save(path, audio, sample_rate)
    except Exception as e:
        logger.error(f"Failed to save audio to {path}: {str(e)}")
        raise