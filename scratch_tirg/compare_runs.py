"""Compares the official TIRG run against the scratch implementation."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

from .event_reader import infer_steps_per_epoch, read_scalar_series
from .utils import save_json


PAPER_REFERENCE = {
    "test_recall_top1_correct_composition": 0.737,
}

REPO_REFERENCE = {
    "test_recall_top1_correct_composition": 0.760,
}

METRICS = [
    "test_recall_top1_correct_composition",
    "test_recall_top5_correct_composition",
    "test_recall_top10_correct_composition",
]


def _scratch_metric_name(metric: str) -> str:
    if metric.startswith("test_"):
        return metric[len("test_") :]
    return metric


def _best_from_official(run_dir: str) -> Dict[str, dict]:
    scalars = read_scalar_series(run_dir)
    loss_steps = sorted(scalars["soft_triplet"])
    steps_per_epoch = infer_steps_per_epoch(loss_steps)
    output = {}
    for metric in METRICS:
        series = scalars[metric]
        best_step, best_value = max(series.items(), key=lambda item: item[1])
        latest_step = max(series)
        output[metric] = {
            "best_value": best_value,
            "best_epoch": best_step // steps_per_epoch - 1,
            "latest_value": series[latest_step],
            "latest_epoch": latest_step // steps_per_epoch - 1,
        }
    return output


def _best_from_scratch(run_dir: str) -> Dict[str, dict]:
    history_path = os.path.join(run_dir, "metrics_history.jsonl")
    records = []
    with open(history_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                if "test_metrics" in record:
                    records.append(record)
    if not records:
        raise ValueError(f"No evaluated scratch records found in {history_path}")
    output = {}
    for metric in METRICS:
        scratch_metric = _scratch_metric_name(metric)
        best_record = max(records, key=lambda record: record["test_metrics"][scratch_metric])
        latest_record = records[-1]
        output[metric] = {
            "best_value": best_record["test_metrics"][scratch_metric],
            "best_epoch": best_record["epoch"],
            "latest_value": latest_record["test_metrics"][scratch_metric],
            "latest_epoch": latest_record["epoch"],
        }
    return output


def _write_latex_table(path: str, comparison: dict) -> None:
    rows = []
    rows.append(r"\begin{tabular}{lcccc}")
    rows.append(r"\toprule")
    rows.append(r"Metric & Paper & Official & Scratch & Scratch-Official \\")
    rows.append(r"\midrule")
    for metric in METRICS:
        paper_value = PAPER_REFERENCE.get(metric)
        official_value = comparison["official"][metric]["best_value"]
        scratch_value = comparison["scratch"][metric]["best_value"]
        delta = scratch_value - official_value
        paper_cell = "--" if paper_value is None else f"{paper_value:.4f}"
        rows.append(
            f"{metric.replace('_', ' ')} & "
            f"{paper_cell} & "
            f"{official_value:.4f} & "
            f"{scratch_value:.4f} & "
            f"{delta:+.4f} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare official and scratch CSS3D runs.")
    parser.add_argument("--official-run-dir", type=str, required=True)
    parser.add_argument("--scratch-run-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--latex-table", type=str, default="")
    args = parser.parse_args()

    comparison = {
        "paper_reference": PAPER_REFERENCE,
        "repo_reference": REPO_REFERENCE,
        "official": _best_from_official(args.official_run_dir),
        "scratch": _best_from_scratch(args.scratch_run_dir),
    }
    save_json(args.output_json, comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    if args.latex_table:
        _write_latex_table(args.latex_table, comparison)
        print(f"Wrote LaTeX table to {args.latex_table}")


if __name__ == "__main__":
    main()
