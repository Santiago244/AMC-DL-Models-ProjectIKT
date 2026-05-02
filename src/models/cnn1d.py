"""1D CNN baseline for automatic modulation classification."""

from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(
        self,
        num_classes: int = 24,
        input_channels: int = 2,
        filters: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.5,
        dense_units: int = 256,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, filters, kernel_size, padding=padding),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(filters, filters, kernel_size, padding=padding),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
            nn.Conv1d(filters, filters * 2, kernel_size, padding=padding),
            nn.BatchNorm1d(filters * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(filters * 2, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dense_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
