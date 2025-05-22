# models/flow.py
import torch
import torch.nn as nn
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class Flow(nn.Module):
    def __init__(self, in_channels: int = 192, hidden_channels: int = 192, n_layers: int = 4):
        super().__init__()
        try:
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(in_channels, hidden_channels, 1),
                    nn.ReLU(),
                    nn.Conv1d(hidden_channels, in_channels, 1)
                ) for _ in range(n_layers)
            ])
        except Exception as e:
            logger.error(f"Failed to initialize Flow: {str(e)}")
            raise

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            log_det = torch.zeros(x.size(0), device=x.device)
            for layer in self.layers:
                x = layer(x)
                log_det += torch.sum(x, dim=(1, 2))  # Simplified log-determinant
            return x, log_det
        except Exception as e:
            logger.error(f"Flow forward pass failed: {str(e)}")
            raise