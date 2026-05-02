"""CNN-LSTM model for I/Q sequence classification."""

from __future__ import annotations

import torch
from torch import nn


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 24,
        input_channels: int = 2,
        conv_filters: int = 64,
        kernel_size: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, conv_filters, kernel_size, padding=padding),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(conv_filters, conv_filters * 2, kernel_size, padding=padding),
            nn.BatchNorm1d(conv_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=conv_filters * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        directions = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden * directions),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * directions, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.transpose(1, 2)
        output, _ = self.lstm(x)
        x = output.mean(dim=1)
        return self.classifier(x)
