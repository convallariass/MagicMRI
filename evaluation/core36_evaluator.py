#!/usr/bin/env python3
"""Evaluate a complete or explicitly allowed subset of Core-36 predictions."""

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.aggregation import family_summaries, patient_summaries, task_summaries  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    LPIPSEvaluator,
    generation_metrics,
    load_image,
    segmentation_metrics,
)
from magicmri.infer import resolve_device  # noqa: E402
from magicmri.manifest import load_inference_manifest  # noqa: E402
from magicmri.tasks import task_index  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run formal Core-36 evaluation")
    parser.add_argument("--manifest", type=Path, required=True, help="Inference predictions.jsonl")
    parser.add_argument("--task-registry", type=Path, default=ROOT / "configs/core36_tasks.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--allow-task-subset", action="store_true")
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value):
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    tasks = task_index(args.task_registry)
    rows = load_inference_manifest(args.manifest, tasks, require_target=True)
    present = {row["task_id"] for row in rows}
    missing_tasks = sorted(set(tasks).difference(present))
    if missing_tasks and not args.allow_task_subset:
        raise RuntimeError(f"Prediction manifest omits {len(missing_tasks)} Core-36 tasks: {missing_tasks}")
    for row in rows:
        if "prediction_path" not in row:
            raise ValueError(f"{row['sample_id']} has no prediction_path")
        prediction = Path(row["prediction_path"])
        if not prediction.is_file():
            raise FileNotFoundError(f"Missing prediction for {row['sample_id']}: {prediction}")
    device = resolve_device(args.device)
    lpips_evaluator = None
    has_generation = any(tasks[row["task_id"]]["metric_family"] == "generation" for row in rows)
    if not args.skip_lpips and has_generation:
        lpips_evaluator = LPIPSEvaluator(str(device))
    per_slice = []
    for row in rows:
        task = tasks[row["task_id"]]
        grayscale = task["metric_family"] == "segmentation"
        prediction = load_image(Path(row["prediction_path"]), grayscale)
        target = load_image(Path(row["query_target"]), grayscale)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {row['sample_id']}: {prediction.shape} versus {target.shape}"
            )
        if grayscale:
            metrics = segmentation_metrics(prediction, target)
        else:
            metrics = generation_metrics(prediction, target, lpips_evaluator)
            if lpips_evaluator is None:
                metrics["LPIPS"] = float("nan")
        record = {
            "sample_id": row["sample_id"],
            "task_id": row["task_id"],
            "family": task["family"],
            "metric_family": task["metric_family"],
            "patient_id": row["patient_id"],
            "slice_id": row["slice_id"],
            "slice_index": row["slice_index"],
        }
        record.update(metrics)
        per_slice.append(record)
    patients = patient_summaries(per_slice)
    task_rows = task_summaries(per_slice)
    families = family_summaries(task_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_slice_path = args.output_dir / "per_slice.jsonl"
    with per_slice_path.open("w", encoding="utf-8") as handle:
        for row in per_slice:
            handle.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n")
    write_json(args.output_dir / "patient_summary.json", patients)
    write_json(args.output_dir / "task_summary.json", task_rows)
    write_json(args.output_dir / "family_summary.json", families)
    print(
        f"evaluated slices={len(per_slice)} patients={len(patients)} "
        f"tasks={len(task_rows)} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
