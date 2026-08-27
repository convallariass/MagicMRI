#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magicmri.inference import infer  # noqa: E402
from magicmri.utils.checkpoint import load_pretrained_model  # noqa: E402
from magicmri.utils.visual_prompt import resolve_example_inputs  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run MagicMRI visual-prompt inference")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--example", help="Example directory containing config.yaml")
    selection.add_argument(
        "--family",
        choices=("translation", "enhancement", "segmentation"),
        help="Run all six examples in one task family",
    )
    selection.add_argument("--all-examples", action="store_true", help="Run all 18 examples")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--examples-root", default="examples")
    parser.add_argument("--checkpoint", default="checkpoints/magicmri_ckpt_release.pth")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate example inputs without loading a checkpoint",
    )
    return parser.parse_args()


def load_example(example_dir: Path):
    config_path = example_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Example config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"example_id", "family", "task", "exemplar_source", "exemplar_target", "query_source"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"{config_path} is missing: {sorted(missing)}")
    return config, resolve_example_inputs(example_dir, config)


def selected_directories(args):
    examples_root = Path(args.examples_root)
    if args.example:
        return [Path(args.example)]
    families = [args.family] if args.family else ["translation", "enhancement", "segmentation"]
    directories = []
    for family in families:
        directories.extend(sorted(path for path in (examples_root / family).iterdir() if path.is_dir()))
    expected = 6 * len(families)
    if len(directories) != expected:
        raise RuntimeError(f"Expected {expected} example directories, found {len(directories)}")
    return directories


def main():
    args = parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        runtime = yaml.safe_load(handle)
    examples = []
    for directory in selected_directories(args):
        config, paths = load_example(directory)
        examples.append((directory, config, paths))
        print(f"validated {config['example_id']}: {config['task']}")
    if args.validate_only:
        print(f"Validated {len(examples)} visual-prompt example(s)")
        return

    device = torch.device(args.device or runtime["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model = load_pretrained_model(args.checkpoint, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, config, paths in examples:
        prediction = infer(model, paths, device, int(runtime["input_size"]))
        output_path = output_dir / f"{config['example_id']}.png"
        if config["family"] == "segmentation":
            gray = np.asarray(Image.fromarray(prediction).convert("L"))
            prediction = np.where(
                gray > int(runtime["segmentation_threshold"]), 255, 0
            ).astype(np.uint8)
        Image.fromarray(prediction).save(output_path)
        print(f"saved {config['example_id']}: {output_path}")


if __name__ == "__main__":
    main()
