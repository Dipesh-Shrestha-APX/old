# Placeholder file
# tests/test_models.py
import unittest
import torch
from models import TextEncoder, PosteriorEncoder
import logging

logger = logging.getLogger(__name__)

class TestModels(unittest.TestCase):
    def test_text_encoder(self):
        try:
            model = TextEncoder(n_vocab=100, embed_dim=192)
            x = torch.randint(0, 100, (2, 10))
            out = model(x)
            self.assertEqual(out.shape, (2, 10, 192))
        except Exception as e:
            logger.error(f"TextEncoder test failed: {str(e)}")
            raise

    def test_posterior_encoder(self):
        try:
            model = PosteriorEncoder()
            x = torch.randn(2, 80, 100)
            mu, logvar = model(x)
            self.assertEqual(mu.shape, (2, 192, 100))
        except Exception as e:
            logger.error(f"PosteriorEncoder test failed: {str(e)}")
            raise

if __name__ == "__main__":
    unittest.main()