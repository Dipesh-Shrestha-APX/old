# Placeholder file
# inference.py
from pipeline.prediction_pipeline import PredictionPipeline
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        pipeline = PredictionPipeline("configs/config.yaml", "checkpoints/epoch_100.pt")
        pipeline.predict("नमस्ते", "output.wav")
    except Exception as e:
        logger.error(f"Inference script failed: {str(e)}")
        raise