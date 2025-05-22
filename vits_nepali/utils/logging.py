# Placeholder file
# utils/logging.py
import logging
import os
from datetime import datetime
from typing import Dict

class Logger:
    def __init__(self, log_dir: str = "logs/"):
        try:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[
                    logging.FileHandler(f"{log_dir}/train_{timestamp}.log"),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
        except Exception as e:
            print(f"Failed to initialize Logger: {str(e)}")
            raise

    def log(self, metrics: Dict[str, float]) -> None:
        try:
            self.logger.info("Metrics: " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {str(e)}")
            raise