# Placeholder file
# VITS-Nepali: Enterprise-Grade Text-to-Speech

A production-ready implementation of VITS (Variational Inference with Normalizing Flows for Text-to-Speech) for Nepali text-to-speech synthesis.

## Features
- Modular architecture (text encoder, posterior encoder, flow, duration predictor, HiFi-GAN)
- Robust error handling and logging
- Scalable training and inference pipelines
- Unit tests for model validation
- Docker support for deployment
- CSV-based manifest files with dataset splitting

## Setup
1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Prepare Dataset**:
   - Place 16 kHz WAV files in `data/dataset/audio/`.
   - Create `data/csv/manifest.csv` with `path,labels` pairs (e.g., `audio1.wav|नमस्ते`).
3. **Split Dataset**: Run `from data.preprocess import split_manifest; split_manifest('data/csv/manifest.csv')` to create `train.csv`, `val.csv`, `test.csv` and move audio files.
4. **Configure**: Verify `configs/config.yaml` points to `data/csv/train.csv` and `data/csv/val.csv`.
5. **Generate Structure**: Run `python template.py` to create the file structure.
6. **Train**: Run `python train.py`.
7. **Synthesize**: Run `python inference.py`.

## Notes
- Replace the placeholder `text_to_phonemes` in `utils/text.py` with a proper Nepali phonemizer.
- Requires a CUDA-enabled GPU for optimal performance.
- Checkpoints are saved in `checkpoints/` every 10 epochs.
- Original `manifest.csv` remains unchanged after splitting.

## Testing
Run `python -m unittest discover tests` to execute unit tests.

## Docker
Build and run with:
```bash
docker build -t vits-nepali .
docker run -v $(pwd)/data:/app/data vits-nepali