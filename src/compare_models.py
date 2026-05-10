"""Build KPI comparison tables from saved experiment evaluation reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def compute_macro_fpr(report: dict[str, Any]) -> float:
    per_class_fpr = report.get("per_class_fpr", {})
    if isinstance(per_class_fpr, dict) and per_class_fpr:
        return mean([float(value) for value in per_class_fpr.values()])

    confusion = report.get("confusion_matrix")
    if not isinstance(confusion, list) or not confusion:
        return 0.0

    matrix = [[float(value) for value in row] for row in confusion]
    total = sum(sum(row) for row in matrix)
    if total <= 0:
        return 0.0

    diagonal = [matrix[idx][idx] for idx in range(len(matrix))]
    predicted_total = [sum(matrix[row][col] for row in range(len(matrix))) for col in range(len(matrix))]
    actual_total = [sum(row) for row in matrix]

    fpr_values: list[float] = []
    for idx in range(len(matrix)):
        true_positive = diagonal[idx]
        false_positive = predicted_total[idx] - true_positive
        negatives = total - actual_total[idx]
        fpr_values.append(false_positive / negatives if negatives > 0 else 0.0)

    return mean(fpr_values)


def read_train_report(report_path: Path) -> dict[str, Any]:
    train_report_path = report_path.with_name("train_report.json")
    if train_report_path.exists():
        return load_json(train_report_path)
    return {}


def build_row(report_path: Path) -> dict[str, Any]:
    report = load_json(report_path)
    train_report = read_train_report(report_path)

    model_type = str(report.get("model_type") or train_report.get("model_type") or "unknown")
    experiment_name = str(report.get("experiment_name") or train_report.get("experiment_name") or report_path.parent.name)
    best_val_accuracy = train_report.get("best_val_accuracy")
    best_epoch = train_report.get("best_epoch")

    return {
        "experiment_name": experiment_name,
        "model_type": model_type,
        "run_dir": str(report_path.parent),
        "test_samples": int(report.get("test_samples", 0)),
        "overall_accuracy": float(report.get("overall_accuracy", 0.0)),
        "macro_precision": float(report.get("macro_precision", 0.0)),
        "macro_recall": float(report.get("macro_recall", 0.0)),
        "macro_f1": float(report.get("macro_f1", 0.0)),
        "macro_fpr": float(report.get("macro_fpr", compute_macro_fpr(report))),
        "best_val_accuracy": float(best_val_accuracy) if best_val_accuracy is not None else None,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "checkpoint": str(train_report.get("checkpoint", report.get("checkpoint", ""))),
    }


def discover_rows(experiments_dir: str | Path) -> list[dict[str, Any]]:
    experiments_dir = Path(experiments_dir)
    report_paths = sorted(experiments_dir.glob("**/test_report.json"))
    return [build_row(report_path) for report_path in report_paths]


def sort_rows(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: row.get(metric, float("-inf")) if row.get(metric) is not None else float("-inf"), reverse=True)


def best_per_model_type(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    best_rows: dict[str, dict[str, Any]] = {}
    for row in sort_rows(rows, metric):
        model_type = str(row.get("model_type", "unknown"))
        if model_type not in best_rows:
            best_rows[model_type] = row
    return sort_rows(list(best_rows.values()), metric)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(rows: list[dict[str, Any]], path: str | Path, columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_markdown(rows: list[dict[str, Any]], path: str | Path, columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [column.replace("_", " ").title() for column in columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for row in rows:
        values = [format_value(row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a KPI comparison table from experiment reports.")
    parser.add_argument("--experiments-dir", default="experiments", help="Directory containing experiment runs.")
    parser.add_argument(
        "--sort-by",
        default="macro_f1",
        choices=["overall_accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_fpr", "best_val_accuracy"],
        help="Metric used to rank runs when building the summary table.",
    )
    parser.add_argument("--output-dir", default="experiments", help="Where the comparison tables will be written.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = discover_rows(args.experiments_dir)

    if not rows:
        raise SystemExit(f"No test_report.json files found under {args.experiments_dir!r}.")

    columns = [
        "experiment_name",
        "model_type",
        "run_dir",
        "test_samples",
        "overall_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_fpr",
        "best_val_accuracy",
        "best_epoch",
    ]

    output_dir = Path(args.output_dir)
    all_runs = sort_rows(rows, args.sort_by)
    best_runs = best_per_model_type(rows, args.sort_by)

    write_csv(all_runs, output_dir / "model_kpis_all_runs.csv", columns)
    write_markdown(all_runs, output_dir / "model_kpis_all_runs.md", columns)
    write_csv(best_runs, output_dir / "model_kpis_best_per_model.csv", columns)
    write_markdown(best_runs, output_dir / "model_kpis_best_per_model.md", columns)

    print(f"Saved comparison tables to: {output_dir}")
    print(f"Ranking metric: {args.sort_by}")
    print(f"Runs discovered: {len(rows)}")
    print(f"Unique model types: {len({row['model_type'] for row in rows})}")


if __name__ == "__main__":
    main()