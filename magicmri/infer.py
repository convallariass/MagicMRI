"""Manifest-driven Core-36 inference command."""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .inference import infer
from .manifest import load_inference_manifest
from .tasks import task_index
from .utils.checkpoint import load_pretrained_model


ROOT = Path(__file__).resolve().parents[1]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def parse_args():
    parser = argparse.ArgumentParser(description="Run manifest-driven MagicMRI Core-36 inference")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("MAGICMRI_CKPT"))
    parser.add_argument("--task-registry", type=Path, default=ROOT / "configs/core36_tasks.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--segmentation-threshold", type=int, default=128)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not safe:
        raise ValueError(f"Unsafe empty path component derived from: {value!r}")
    return safe


def render_prediction(prediction: np.ndarray, output_type: str, threshold: int) -> np.ndarray:
    if output_type != "binary_mask":
        return prediction
    gray = np.asarray(Image.fromarray(prediction).convert("L"))
    return np.where(gray > threshold, 255, 0).astype(np.uint8)


def main():
    args = parse_args()
    tasks = task_index(args.task_registry)
    rows = load_inference_manifest(args.manifest, tasks)
    if args.validate_only:
        print(f"Validated {len(rows)} manifest record(s) against {len(tasks)} Core-36 tasks")
        return
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for inference (or set MAGICMRI_CKPT)")
    device = resolve_device(args.device)
    print(f"device={device}")
    model = load_pretrained_model(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    for row in rows:
        prediction = infer(model, row, device, args.input_size)
        prediction = render_prediction(
            prediction,
            tasks[row["task_id"]]["output_type"],
            args.segmentation_threshold,
        )
        task_dir = args.output_dir / _safe_component(row["task_id"])
        task_dir.mkdir(parents=True, exist_ok=True)
        output_path = task_dir / f"{_safe_component(row['sample_id'])}.png"
        Image.fromarray(prediction).save(output_path)
        result = dict(row)
        result["prediction_path"] = str(output_path.resolve())
        prediction_rows.append(result)
        print(f"saved {row['sample_id']}: {output_path}")
    result_manifest = args.output_dir / "predictions.jsonl"
    with result_manifest.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"prediction_manifest={result_manifest}")


if __name__ == "__main__":
    main()
