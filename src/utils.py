"""Shared helpers for experiment scripts."""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def create_run_dir(config: dict[str, Any], config_path: str | Path) -> Path:
    experiments_dir = Path(config.get("output", {}).get("experiments_dir", "experiments"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get("experiment_name", "experiment")
    run_dir = experiments_dir / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(config_path, run_dir / "config.yaml")
    return run_dir


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
