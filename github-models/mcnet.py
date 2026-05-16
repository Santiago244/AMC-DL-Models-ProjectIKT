"""PyTorch MCNet translated from ThienHuynhThe/MCNet.

Reference:
https://github.com/ThienHuynhThe/MCNet/blob/master/mcnet_commlett.m
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from .common import ExplicitPadConv2d, ExplicitPadPool2d
except ImportError:  # Loaded dynamically from a path by src.models.factory.
    from common import ExplicitPadConv2d, ExplicitPadPool2d


def _relu_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> nn.Sequential:
    return nn.Sequential(
        ExplicitPadConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
    )


def _max_pool(
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int, int, int],
) -> nn.Module:
    return ExplicitPadPool2d(nn.MaxPool2d(kernel_size=kernel_size, stride=stride), padding=padding)


def _avg_pool(
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int, int, int],
) -> nn.Module:
    return ExplicitPadPool2d(nn.AvgPool2d(kernel_size=kernel_size, stride=stride), padding=padding)


class MCNet(nn.Module):
    def __init__(self, num_classes: int = 24, input_channels: int = 2) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError("MCNet expects two I/Q input channels.")

        self.conv1 = _relu_conv(1, 64, (3, 7), stride=(1, 2), padding=(1, 1, 3, 3))
        self.pool1 = _max_pool((1, 3), stride=(1, 2), padding=(0, 0, 1, 1))

        self.pre_a = nn.Sequential(
            _relu_conv(64, 32, (3, 1), padding=(1, 1, 0, 0)),
            _avg_pool((1, 3), stride=(1, 2), padding=(0, 0, 1, 1)),
        )
        self.pre_b = _relu_conv(64, 32, (1, 3), stride=(1, 2), padding=(0, 0, 1, 1))

        self.jump_a = nn.Sequential(
            _relu_conv(64, 128, (1, 1), stride=(1, 2)),
            _max_pool((1, 3), stride=(1, 2), padding=(0, 0, 1, 1)),
        )

        self.post_pooling = _max_pool((1, 3), stride=(1, 2), padding=(0, 0, 1, 1))
        self.mblock_a_a = _relu_conv(64, 32, (1, 1))
        self.mblock_a_b = nn.Sequential(
            _relu_conv(32, 48, (3, 1), padding=(1, 1, 0, 0)),
            _avg_pool((1, 3), stride=(1, 2), padding=(0, 0, 1, 1)),
        )
        self.mblock_a_c = _relu_conv(32, 48, (1, 3), stride=(1, 2), padding=(0, 0, 1, 1))
        self.mblock_a_d = _relu_conv(32, 32, (1, 1), stride=(1, 2))

        self.mblock_b_a = _relu_conv(128, 32, (1, 1))
        self.mblock_b_b = _relu_conv(32, 48, (3, 1), padding=(1, 1, 0, 0))
        self.mblock_b_c = _relu_conv(32, 48, (1, 3), padding=(0, 0, 1, 1))
        self.mblock_b_d = _relu_conv(32, 32, (1, 1))

        self.jump_c = _max_pool((2, 2), stride=(1, 2), padding=(1, 0, 0, 0))
        self.mblock_c_a = _relu_conv(128, 32, (1, 1))
        self.mblock_c_b = nn.Sequential(
            _relu_conv(32, 48, (3, 1), padding=(1, 1, 0, 0)),
            _avg_pool((1, 3), stride=(1, 2), padding=(0, 0, 1, 1)),
        )
        self.mblock_c_c = _relu_conv(32, 48, (1, 3), stride=(1, 2), padding=(0, 0, 1, 1))
        self.mblock_c_d = _relu_conv(32, 32, (1, 1), stride=(1, 2))

        self.mblock_d_a = _relu_conv(128, 32, (1, 1))
        self.mblock_d_b = _relu_conv(32, 48, (3, 1), padding=(1, 1, 0, 0))
        self.mblock_d_c = _relu_conv(32, 48, (1, 3), padding=(0, 0, 1, 1))
        self.mblock_d_d = _relu_conv(32, 32, (1, 1))

        self.jump_e = _max_pool((2, 2), stride=(1, 2), padding=(1, 0, 0, 0))
        self.mblock_e_a = _relu_conv(128, 32, (1, 1))
        self.mblock_e_b = nn.Sequential(
            _relu_conv(32, 48, (3, 1), padding=(1, 1, 0, 0)),
            _max_pool((1, 3), stride=(1, 2), padding=(0, 0, 1, 1)),
        )
        self.mblock_e_c = _relu_conv(32, 48, (1, 3), stride=(1, 2), padding=(0, 0, 1, 1))
        self.mblock_e_d = _relu_conv(32, 32, (1, 1), stride=(1, 2))

        self.mblock_f_a = _relu_conv(128, 32, (1, 1))
        self.mblock_f_b = _relu_conv(32, 96, (3, 1), padding=(1, 1, 0, 0))
        self.mblock_f_c = _relu_conv(32, 96, (1, 3), padding=(0, 0, 1, 1))
        self.mblock_f_d = _relu_conv(32, 64, (1, 1))

        self.global_pool = nn.AvgPool2d(kernel_size=(2, 8))
        self.fc = nn.Linear(384, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))

        pre = torch.cat([self.pre_b(x), self.pre_a(x)], dim=1)
        jump_a = self.jump_a(pre)

        a0 = self.mblock_a_a(self.post_pooling(pre))
        mix_a = torch.cat([self.mblock_a_c(a0), self.mblock_a_b(a0), self.mblock_a_d(a0)], dim=1)
        add_a = mix_a + jump_a

        b0 = self.mblock_b_a(add_a)
        mix_b = torch.cat([self.mblock_b_b(b0), self.mblock_b_c(b0), self.mblock_b_d(b0)], dim=1)
        add_b = mix_b + add_a

        c0 = self.mblock_c_a(add_b)
        mix_c = torch.cat([self.mblock_c_b(c0), self.mblock_c_c(c0), self.mblock_c_d(c0)], dim=1)
        add_c = mix_c + self.jump_c(add_b)

        d0 = self.mblock_d_a(add_c)
        mix_d = torch.cat([self.mblock_d_c(d0), self.mblock_d_b(d0), self.mblock_d_d(d0)], dim=1)
        add_d = mix_d + add_c

        e0 = self.mblock_e_a(add_d)
        mix_e = torch.cat([self.mblock_e_c(e0), self.mblock_e_b(e0), self.mblock_e_d(e0)], dim=1)
        add_e = mix_e + self.jump_e(add_d)

        f0 = self.mblock_f_a(add_e)
        mix_f = torch.cat([self.mblock_f_c(f0), self.mblock_f_b(f0), self.mblock_f_d(f0)], dim=1)

        x = torch.cat([mix_f, add_e], dim=1)
        x = self.global_pool(x).flatten(1)
        return self.dropout(self.fc(x))

