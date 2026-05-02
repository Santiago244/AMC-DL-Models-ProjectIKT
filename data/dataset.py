"""Dataset utilities for RadioML 2018.01A.

This module keeps the expensive 25 GB HDF5 file untouched and saves only
lightweight index splits. The same split file should be reused for every model
so CNN, CNN-LSTM, and ResNet are compared on identical samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
# from tabulate import tabulate

DEFAULT_HDF5_PATH = Path("data/raw/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5")
DEFAULT_SPLIT_PATH = Path("data/splits/radioml2018a_seed42_60_20_20.npz") # Where the split indices are saved.


def _validate_split_ratios(train_split: float, val_split: float, test_split: float) -> None:
    total = train_split + val_split + test_split
    # print(tabulate(np.isclose(total, 1.0)))
    if not np.isclose(total, 1.0):
        raise ValueError(
            "train_split + val_split + test_split must equal 1.0 "
            f"(got {total:.6f})"
        )

    if min(train_split, val_split, test_split) <= 0:
        raise ValueError("All split ratios must be greater than 0.")


def read_labels_and_snr(hdf5_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read labels and SNR values without loading the full signal array.

    Returns:
        y: Integer modulation labels with shape (N,).
        z: SNR values with shape (N,).
    """

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
    """Create a deterministic stratified split using only NumPy.

    Splitting inside each stratum preserves the class/SNR distribution while
    avoiding an extra dependency just to create index files.
    """

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

        test_parts.append(group_indices[:n_test])
        val_parts.append(group_indices[n_test : n_test + n_val])
        train_parts.append(group_indices[n_test + n_val :])

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
    train_split: float = 0.6,
    val_split: float = 0.2,
    test_split: float = 0.2,
    random_seed: int = 42,
    stratify_by_snr: bool = True,
) -> Path:
    """Create and save train/validation/test indices for RadioML.

    The split is stratified by modulation class and, by default, SNR level.
    This preserves the class/SNR distribution across train, validation, and
    test sets, which is important for fair per-SNR evaluation.
    """

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
        indices,
        stratify_key,
        train_split,
        val_split,
        test_split,
        random_seed,
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
    """Load previously saved train/validation/test indices."""

    split = np.load(split_path)
    return split["train_idx"], split["val_idx"], split["test_idx"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create RadioML train/val/test split indices.")
    parser.add_argument(
        "--hdf5-path",
        default=str(DEFAULT_HDF5_PATH),
        help="Path to the RadioML 2018.01A HDF5 file.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_SPLIT_PATH),
        help="Where to save the split .npz file.",
    )
    parser.add_argument("--train-split", type=float, default=0.6)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--stratify-modulation-only",
        action="store_true",
        help="Use only modulation labels for stratification instead of modulation + SNR.",
    )
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
