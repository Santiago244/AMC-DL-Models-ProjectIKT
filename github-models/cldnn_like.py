"""PyTorch CLDNN-like model translated from leena201818/radioml.

Reference:
https://github.com/leena201818/radioml/blob/master/rmlmodels/CLDNNLikeModel.py
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from .common import SamePadConv1d, init_keras_like
except ImportError:  # Loaded dynamically from a path by src.models.factory.
    from common import SamePadConv1d, init_keras_like


class CLDNNLikeModel(nn.Module):
    def __init__(self, num_classes: int = 24, input_channels: int = 2) -> None:
        super().__init__()
        dropout = 0.5
        kernel_size = 8

        conv_blocks: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(4):
            conv_blocks.extend(
                [
                    SamePadConv1d(in_channels, 64, kernel_size),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                ]
            )
            in_channels = 64

        self.conv = nn.Sequential(*conv_blocks)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=50, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(50, 128),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.apply(init_keras_like)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.classifier(x[:, -1, :])
