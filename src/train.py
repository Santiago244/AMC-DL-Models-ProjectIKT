"""Train one PyTorch model from a YAML experiment config."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from src.dataset import build_dataloaders
from src.models.factory import build_model
from src.utils import count_parameters, create_run_dir, get_device, load_config, save_json, set_seed


def apply_runtime_overrides(
    config: dict[str, Any],
    epochs: int | None = None,
    batch_size: int | None = None,
    patience: int | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict[str, Any]:
    """Apply command-line overrides after loading the YAML config."""

    if epochs is not None:
        config["training"]["epochs"] = epochs
    if batch_size is not None:
        config["training"]["batch_size"] = batch_size
    if patience is not None:
        config["training"]["early_stopping_patience"] = patience

    if max_train_samples is not None:
        config["data"]["max_train_samples"] = max_train_samples
    if max_val_samples is not None:
        config["data"]["max_val_samples"] = max_val_samples
    if max_test_samples is not None:
        config["data"]["max_test_samples"] = max_test_samples

    return config


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    training = config["training"]
    learning_rate = float(training.get("learning_rate", 1e-3))
    weight_decay = float(training.get("weight_decay", 0.0))
    optimizer_name = training.get("optimizer", "adam").lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=desc, leave=False)
    for x, y, _snr in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(x)
            loss = criterion(logits, y)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

        progress.set_postfix(
            loss=f"{total_loss / max(total_samples, 1):.4f}",
            acc=f"{total_correct / max(total_samples, 1):.4f}",
        )

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def save_history_csv(history: list[dict[str, float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"],
        )
        writer.writeheader()
        writer.writerows(history)


def train(
    config_path: str | Path,
    device_name: str = "auto",
    epochs: int | None = None,
    batch_size: int | None = None,
    patience: int | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> Path:
    config = load_config(config_path)
    config = apply_runtime_overrides(
        config,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
    )
    seed = int(config.get("data", {}).get("random_seed", 42))
    set_seed(seed)

    device = get_device(device_name)
    run_dir = create_run_dir(config, config_path)
    save_json(config, run_dir / "effective_config.json")

    train_loader, val_loader, _test_loader = build_dataloaders(config)

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)

    epochs = int(config["training"].get("epochs", 5))
    patience = int(config["training"].get("early_stopping_patience", 3))

    print(
        "Training setup | "
        f"experiment={config.get('experiment_name')} | "
        f"model={config['model']['type']} | "
        f"epochs={epochs} | "
        f"batch_size={config['training'].get('batch_size')} | "
        f"max_train_samples={config['data'].get('max_train_samples', 'all')} | "
        f"max_val_samples={config['data'].get('max_val_samples', 'all')} | "
        f"device={device}"
    )

    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, float]] = []
    checkpoint_path = run_dir / "best_model.pth"

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            desc=f"Epoch {epoch}/{epochs} train",
        )
        # Validation is done after each epoch but without gradient updates to track best model by val accuracy and enable early stopping.
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            desc=f"Epoch {epoch}/{epochs} val",
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        save_history_csv(history, run_dir / "history.csv")

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {row['train_loss']:.4f}, acc {row['train_accuracy']:.4f} | "
            f"val loss {row['val_loss']:.4f}, acc {row['val_accuracy']:.4f}"
        )

        improved = val_metrics["accuracy"] > best_val_accuracy
        if improved:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_accuracy": best_val_accuracy,
                    "best_val_loss": best_val_loss,
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        train_report = {
            "experiment_name": config.get("experiment_name"),
            "run_dir": str(run_dir),
            "device": str(device),
            "model_type": config["model"]["type"],
            "num_parameters": count_parameters(model),
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "checkpoint": str(checkpoint_path),
            "history": history,
        }
        save_json(train_report, run_dir / "train_report.json")

        if patience_counter >= patience:
            print(f"Early stopping after {epoch} epochs. Best epoch: {best_epoch}.")
            break

    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an AMC model from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--epochs", type=int, default=None, help="Override config training.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config training.batch_size.")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Override quick train subset size.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Override quick validation subset size.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Override quick test subset size.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = train(
        args.config,
        device_name=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )
    print(f"Training complete. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
