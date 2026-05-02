"""Evaluate a trained checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.dataset import CLASSES, build_dataloaders
from src.models.factory import build_model
from src.utils import get_device, load_config, save_json, set_seed


def compute_report(
    targets: np.ndarray,
    predictions: np.ndarray,
    snrs: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for target, prediction in zip(targets, predictions):
        confusion[int(target), int(prediction)] += 1

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    overall_accuracy = correct / total if total else 0.0

    per_class_accuracy = {}
    for idx, class_name in enumerate(class_names):
        class_total = int(confusion[idx].sum())
        per_class_accuracy[class_name] = (
            float(confusion[idx, idx] / class_total) if class_total else 0.0
        )

    true_positive = np.diag(confusion).astype(np.float64)
    predicted_total = confusion.sum(axis=0).astype(np.float64)
    actual_total = confusion.sum(axis=1).astype(np.float64)
    precision = np.divide(true_positive, predicted_total, out=np.zeros_like(true_positive), where=predicted_total > 0)
    recall = np.divide(true_positive, actual_total, out=np.zeros_like(true_positive), where=actual_total > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)

    per_snr_accuracy = {}
    for snr in sorted(np.unique(snrs)):
        mask = snrs == snr
        per_snr_accuracy[str(int(snr))] = float((predictions[mask] == targets[mask]).mean())

    return {
        "overall_accuracy": float(overall_accuracy),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_snr_accuracy": per_snr_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
        "class_names": class_names,
    }


def evaluate(config_path: str | Path, checkpoint_path: str | Path, output_dir: str | Path | None, device_name: str) -> Path:
    config = load_config(config_path)
    seed = int(config.get("data", {}).get("random_seed", 42))
    set_seed(seed)

    device = get_device(device_name)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir) if output_dir is not None else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    _train_loader, _val_loader, test_loader = build_dataloaders(config)
    model = build_model(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_snrs: list[np.ndarray] = []

    with torch.no_grad():
        for x, y, snr in tqdm(test_loader, desc="Evaluating test split"):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            predictions = logits.argmax(dim=1).cpu().numpy()

            all_predictions.append(predictions)
            all_targets.append(y.numpy())
            all_snrs.append(snr.numpy())

    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    snrs = np.concatenate(all_snrs)

    report = compute_report(targets, predictions, snrs, CLASSES)
    report.update(
        {
            "experiment_name": config.get("experiment_name"),
            "model_type": config["model"]["type"],
            "checkpoint": str(checkpoint_path),
            "test_samples": int(len(targets)),
        }
    )

    report_path = output_dir / "test_report.json"
    save_json(report, report_path)
    np.savetxt(output_dir / "confusion_matrix.csv", np.asarray(report["confusion_matrix"]), delimiter=",", fmt="%d")

    print(f"Saved test report to: {report_path}")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained AMC checkpoint.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth.")
    parser.add_argument("--output-dir", default=None, help="Defaults to the checkpoint directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate(args.config, args.checkpoint, args.output_dir, args.device)


if __name__ == "__main__":
    main()
