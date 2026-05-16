"""PyTorch MCLDNN translated from Richardzhangxx/AMR-Benchmark.

Reference:
https://github.com/Richardzhangxx/AMR-Benchmark/blob/main/RML2018/MCLDNN/rmlmodels/MCLDNN.py
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from .common import CausalConv1d, SamePadConv2d, init_keras_like
except ImportError:  # Loaded dynamically from a path by src.models.factory.
    from common import CausalConv1d, SamePadConv2d, init_keras_like


class MCLDNN(nn.Module):
    def __init__(self, num_classes: int = 24, input_channels: int = 2) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError("MCLDNN expects two I/Q input channels.")

        dropout = 0.5
        self.conv1_1 = SamePadConv2d(1, 50, (2, 8))
        self.conv1_2 = CausalConv1d(1, 50, 8)
        self.conv1_3 = CausalConv1d(1, 50, 8)
        self.conv2 = SamePadConv2d(50, 50, (1, 8))
        self.conv4 = nn.Conv2d(100, 100, kernel_size=(2, 5))

        self.lstm1 = nn.LSTM(input_size=100, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.apply(init_keras_like)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        iq = x.unsqueeze(1)
        i_channel = x[:, 0:1, :]
        q_channel = x[:, 1:2, :]

        x1 = torch.relu(self.conv1_1(iq))
        x2 = torch.relu(self.conv1_2(i_channel)).unsqueeze(2)
        x3 = torch.relu(self.conv1_3(q_channel)).unsqueeze(2)

        x_iq = torch.cat([x2, x3], dim=2)
        x_iq = torch.relu(self.conv2(x_iq))
        x = torch.cat([x1, x_iq], dim=1)
        x = torch.relu(self.conv4(x))

        x = x.squeeze(2).transpose(1, 2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.classifier(x[:, -1, :])
