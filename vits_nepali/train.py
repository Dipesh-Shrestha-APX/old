# Placeholder file
# train.py
from pipeline.training_pipeline import TrainingPipeline
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline("configs/config.yaml", manifest_file="data/csv/train.csv")
        pipeline.run()
        test_loss = pipeline.evaluate("data/csv/test.csv")
        logger.info(f"Test Loss: {test_loss}")
    except Exception as e:
        logger.error(f"Training script failed: {str(e)}")
        raise