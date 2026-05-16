"""Small layer helpers for Keras/MATLAB-to-PyTorch model translations."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def init_keras_like(module: nn.Module) -> None:
    """Approximate Keras defaults used by the reference models."""

    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, parameter in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)


class SamePadConv1d(nn.Module):
    """Conv1d with TensorFlow/Keras-style ``padding='same'`` for stride 1."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        total_padding = kernel_size - 1
        self.left = total_padding // 2
        self.right = total_padding - self.left
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left, self.right)))


class CausalConv1d(nn.Module):
    """Conv1d equivalent to Keras ``padding='causal'``."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.left = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left, 0)))


class SamePadConv2d(nn.Module):
    """Conv2d with TensorFlow/Keras-style ``padding='same'`` for stride 1."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int]) -> None:
        super().__init__()
        pad_h = kernel_size[0] - 1
        pad_w = kernel_size[1] - 1
        self.padding = (
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, self.padding))


class ExplicitPadConv2d(nn.Module):
    """Conv2d with MATLAB-style explicit [top bottom left right] padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        super().__init__()
        top, bottom, left, right = padding
        self.padding = (left, right, top, bottom)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, self.padding))


class ExplicitPadPool2d(nn.Module):
    """Pooling with MATLAB-style explicit [top bottom left right] padding."""

    def __init__(
        self,
        pool: nn.Module,
        padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        super().__init__()
        top, bottom, left, right = padding
        self.padding = (left, right, top, bottom)
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.pad(x, self.padding))
