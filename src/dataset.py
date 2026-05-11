"""RadioML 2018.01A dataset utilities for PyTorch training.

The original HDF5 file stays in ``data/raw``. The split file in ``data/splits``
contains only train/validation/test row indices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


CLASSES = [
    "OOK",
    "4ASK",
    "8ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "16APSK",
    "32APSK",
    "64APSK",
    "128APSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "AM-SSB-WC",
    "AM-SSB-SC",
    "AM-DSB-WC",
    "AM-DSB-SC",
    "FM",
    "GMSK",
    "OQPSK",
]


DEFAULT_HDF5_PATH = Path("data/raw/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5")
DEFAULT_SPLIT_PATH = Path("data/splits/radioml2018a_seed42_60_20_20.npz")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve config paths relative to the project root, not the launch cwd."""

    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def require_existing_file(path: str | Path, description: str) -> Path:
    resolved = resolve_project_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{description} not found: {resolved}\n"
            f"Configured path was: {path}\n"
            "If this is the RadioML HDF5 dataset, place "
            "GOLD_XYZ_OSC.0001_1024.hdf5 in data/raw/radioml2018/ "
            "or update data.hdf5_path in the YAML config."
        )
    return resolved


def _validate_split_ratios(train_split: float, val_split: float, test_split: float) -> None:
    total = train_split + val_split + test_split
    if not np.isclose(total, 1.0):
        raise ValueError(
            "train_split + val_split + test_split must equal 1.0 "
            f"(got {total:.6f})"
        )

    if min(train_split, val_split, test_split) <= 0:
        raise ValueError("All split ratios must be greater than 0.")


def read_labels_and_snr(hdf5_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read labels and SNR values without loading the full signal array."""

    hdf5_path = require_existing_file(hdf5_path, "HDF5 dataset")
    with h5py.File(hdf5_path, "r") as f:
        y = np.argmax(f["Y"][:], axis=1).astype(np.int64).reshape(-1)
        z = f["Z"][:].reshape(-1)

    return y, z


def stratified_train_val_test_split(
    indices: np.ndarray,
    stratify_key: np.ndarray,
    train_split: float,
    val_split: float,
    test_split: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic stratified split using only NumPy."""

    rng = np.random.default_rng(random_seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for key in np.unique(stratify_key):
        group_indices = indices[stratify_key == key].copy()
        rng.shuffle(group_indices)

        n = len(group_indices)
        n_test = int(round(n * test_split))
        n_val = int(round(n * val_split))

        test_parts.append(group_indices[:n_test]) # from 0 to n_test
        val_parts.append(group_indices[n_test : n_test + n_val]) # from n_test to n_test + n_val
        train_parts.append(group_indices[n_test + n_val :]) # from n_test + n_val to end (rest of the dataset labels)

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def create_split_indices(
    hdf5_path: str | Path,
    output_path: str | Path = DEFAULT_SPLIT_PATH,
    train_split: float = 0.6, # 60% of the dataset for training
    val_split: float = 0.2, # 60% of the dataset for training
    test_split: float = 0.2,# 60% of the dataset for training
    random_seed: int = 42, # Random seed helps to ensure that the split is reproducible. Using the same seed will yield the same split. Reproducibility means that if someone else runs the same code with the same seed, they will get the same train/validation/test split, which is important for consistent results and fair comparisons.
    stratify_by_snr: bool = True, # Meaning we are going to split the different samples in the dataset based on both their modulation type and their SNR level. This is done to ensure that each split (train, validation, test) has a representative distribution of samples across different modulation types and SNR levels. By stratifying by both modulation and SNR, we can help ensure that the model is trained and evaluated on a diverse set of conditions, which can lead to better generalization performance.
) -> Path:
    """Create and save train/validation/test indices for RadioML."""

    _validate_split_ratios(train_split, val_split, test_split)

    hdf5_path = Path(hdf5_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y, z = read_labels_and_snr(hdf5_path)
    if len(y) != len(z):
        raise ValueError(f"Label and SNR arrays must have the same length, got {len(y)} and {len(z)}.")

    indices = np.arange(len(y), dtype=np.int64)

    if stratify_by_snr:
        snr_levels, snr_codes = np.unique(z, return_inverse=True)
        stratify_key = y * len(snr_levels) + snr_codes.reshape(-1)
    else:
        snr_levels = np.unique(z)
        stratify_key = y

    train_idx, val_idx, test_idx = stratified_train_val_test_split(
        indices=indices,
        stratify_key=stratify_key,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        random_seed=random_seed,
    )

    metadata = {
        "source_hdf5": str(hdf5_path),
        "num_samples": int(len(indices)),
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
        "random_seed": random_seed,
        "stratify_by": "modulation_and_snr" if stratify_by_snr else "modulation",
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "snr_levels": [float(snr) for snr in snr_levels.tolist()],
    }

    np.savez_compressed(
        output_path,
        train_idx=np.sort(train_idx),
        val_idx=np.sort(val_idx),
        test_idx=np.sort(test_idx),
        metadata=json.dumps(metadata, indent=2),
    )

    return output_path


def load_split_indices(split_path: str | Path = DEFAULT_SPLIT_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_path = require_existing_file(split_path, "Split file")
    split = np.load(split_path)
    return split["train_idx"], split["val_idx"], split["test_idx"]


def limit_indices(indices: np.ndarray, max_samples: int | None, random_seed: int) -> np.ndarray:
    """Return a deterministic subset of indices for quick experiments."""

    if max_samples is None or max_samples <= 0 or max_samples >= len(indices):
        return indices

    rng = np.random.default_rng(random_seed)
    selected = rng.choice(indices, size=max_samples, replace=False)
    return np.sort(selected)


class RadioMLDataset(Dataset):
    """Memory-efficient HDF5-backed dataset.

    Each worker opens its own HDF5 handle lazily. Samples are returned as:
    ``x`` with shape ``(2, 1024)``, integer ``y``, and float ``snr``.
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        indices: np.ndarray,
        sample_normalize: bool = False,
    ) -> None:
        self.hdf5_path = str(require_existing_file(hdf5_path, "HDF5 dataset"))
        self.indices = np.asarray(indices, dtype=np.int64)
        self.sample_normalize = sample_normalize
        self._file: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_idx = int(self.indices[idx])

        x = self.file["X"][sample_idx].astype(np.float32).T
        if self.sample_normalize:
            std = x.std()
            if std > 0:
                x = (x - x.mean()) / std

        y = int(np.argmax(self.file["Y"][sample_idx]))
        snr = float(np.asarray(self.file["Z"][sample_idx]).reshape(-1)[0])

        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(snr, dtype=torch.float32),
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_config = config["data"]
    training_config = config["training"]

    train_idx, val_idx, test_idx = load_split_indices(data_config["split_path"])
    hdf5_path = data_config["hdf5_path"]
    sample_normalize = bool(data_config.get("sample_normalize", False))
    random_seed = int(data_config.get("random_seed", 42))

    train_idx = limit_indices(train_idx, data_config.get("max_train_samples"), random_seed)
    val_idx = limit_indices(val_idx, data_config.get("max_val_samples"), random_seed + 1)
    test_idx = limit_indices(test_idx, data_config.get("max_test_samples"), random_seed + 2)

    batch_size = int(training_config.get("batch_size", 512))
    num_workers = int(training_config.get("num_workers", 0))
    pin_memory = bool(training_config.get("pin_memory", True))

    train_dataset = RadioMLDataset(hdf5_path, train_idx, sample_normalize=sample_normalize)
    val_dataset = RadioMLDataset(hdf5_path, val_idx, sample_normalize=sample_normalize)
    test_dataset = RadioMLDataset(hdf5_path, test_idx, sample_normalize=sample_normalize)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create RadioML train/val/test split indices.")
    parser.add_argument("--hdf5-path", default=str(DEFAULT_HDF5_PATH))
    parser.add_argument("--output", default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument("--train-split", type=float, default=0.6)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--stratify-modulation-only", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    split_path = create_split_indices(
        hdf5_path=args.hdf5_path,
        output_path=args.output,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        random_seed=args.random_seed,
        stratify_by_snr=not args.stratify_modulation_only,
    )

    split = np.load(split_path)
    metadata = json.loads(str(split["metadata"]))
    print(f"Saved split indices to: {split_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
