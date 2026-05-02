"""Small 1D ResNet for automatic modulation classification."""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        return self.relu(out)


class ResNet1D(nn.Module):
    def __init__(
        self,
        num_classes: int = 24,
        input_channels: int = 2,
        base_channels: int = 64,
        layers: tuple[int, int, int] = (2, 2, 2),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = base_channels

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(base_channels, layers[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2, dropout=dropout)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, num_classes),
        )

    def _make_layer(self, out_channels: int, blocks: int, stride: int, dropout: float) -> nn.Sequential:
        layers = [BasicBlock1D(self.in_channels, out_channels, stride=stride, dropout=dropout)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)
