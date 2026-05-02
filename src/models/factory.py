"""Model factory used by training and evaluation scripts."""

from __future__ import annotations

from typing import Any

from torch import nn

from src.models.cnn1d import CNN1D
from src.models.cnn_lstm import CNNLSTM
from src.models.resnet1d import ResNet1D


def build_model(config: dict[str, Any]) -> nn.Module:
    model_config = dict(config["model"])
    model_type = model_config.pop("type").lower()

    if model_type == "cnn1d":
        return CNN1D(**model_config)
    if model_type == "cnn_lstm":
        return CNNLSTM(**model_config)
    if model_type == "resnet1d":
        layers = model_config.get("layers")
        if isinstance(layers, list):
            model_config["layers"] = tuple(layers)
        return ResNet1D(**model_config)

    raise ValueError(f"Unknown model type: {model_type}")
