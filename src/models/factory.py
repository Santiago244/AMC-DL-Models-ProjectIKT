"""Model factory used by training and evaluation scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from torch import nn

from src.models.cnn1d import CNN1D
from src.models.cnn_lstm import CNNLSTM
from src.models.resnet1d import ResNet1D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GITHUB_MODELS_DIR = PROJECT_ROOT / "github-models"


def _load_github_model(module_name: str, class_name: str) -> type[nn.Module]:
    common_path = GITHUB_MODELS_DIR / "common.py"
    if "common" not in sys.modules:
        common_spec = importlib.util.spec_from_file_location("common", common_path)
        if common_spec is None or common_spec.loader is None:
            raise ImportError(f"Could not load helper module from {common_path}")
        common_module = importlib.util.module_from_spec(common_spec)
        sys.modules["common"] = common_module
        common_spec.loader.exec_module(common_module)

    module_path = GITHUB_MODELS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"github_models_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


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
    if model_type == "cldnn_like":
        return _load_github_model("cldnn_like", "CLDNNLikeModel")(**model_config)
    if model_type == "mcldnn":
        return _load_github_model("mcldnn", "MCLDNN")(**model_config)
    if model_type == "mcnet":
        return _load_github_model("mcnet", "MCNet")(**model_config)

    raise ValueError(f"Unknown model type: {model_type}")
