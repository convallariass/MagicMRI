#!/usr/bin/env python3
"""Generate deterministic, non-clinical visual-prompt fixtures."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageFilter


TRANSLATION = [
    ("T01", "modality_translation-t1cTOt1n"),
    ("T02", "modality_translation-t1nTOt2f"),
    ("T03", "modality_translation-t2fTOt2w"),
    ("T04", "modality_translation-t2wTOt1c"),
    ("T05", "modality_translation-t1cTOt2w"),
    ("T06", "modality_translation-t2fTOt1n"),
]
ENHANCEMENT = [
    ("E01", "image_enhancement-t1c_deblurx2"),
    ("E02", "image_enhancement-t1n_degaussian"),
    ("E03", "image_enhancement-t2f_desaltpepper"),
    ("E04", "image_enhancement-t2w_deblurx2"),
    ("E05", "image_enhancement-t1c_degaussian"),
    ("E06", "image_enhancement-t2w_desaltpepper"),
]
SEGMENTATION = [
    ("S01", "tumor_segmentation-t1c_Glioma"),
    ("S02", "tumor_segmentation-t2f_Glioma"),
    ("S03", "tumor_segmentation-t1n_Meningioma"),
    ("S04", "tumor_segmentation-t2w_Meningioma"),
    ("S05", "tumor_segmentation-t1c_Metastasis"),
    ("S06", "tumor_segmentation-t2f_Metastasis"),
]


def synthetic_mri(size: int, seed: int, lesion: bool = False):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)]
    brain = (x / 0.78) ** 2 + (y / 0.92) ** 2 < 1
    tissue = 0.18 + 0.38 * np.exp(-2.4 * (x * x + y * y))
    tissue += 0.09 * np.cos(14 * x + seed * 0.1) * np.exp(-2.0 * (x * x + y * y))
    tissue += rng.normal(0, 0.018, (size, size))
    mask = np.zeros((size, size), dtype=bool)
    if lesion:
        cx = -0.25 + 0.1 * (seed % 5)
        cy = -0.2 + 0.08 * (seed % 4)
        mask = ((x - cx) / 0.13) ** 2 + ((y - cy) / 0.09) ** 2 < 1
        tissue[mask] += 0.28
    tissue = np.clip(tissue * brain, 0, 1)
    return (tissue * 255).astype(np.uint8), mask


def translate(values: np.ndarray, task: str) -> np.ndarray:
    normalized = values.astype(np.float32) / 255.0
    task = task.lower()
    if "tot1n" in task:
        transformed = normalized**1.25
    elif "tot2f" in task:
        transformed = np.sqrt(normalized) * 0.88
    elif "tot2w" in task:
        transformed = np.clip(normalized * 1.22 + 0.05, 0, 1)
    else:
        transformed = np.clip(normalized**0.82 * 1.05, 0, 1)
    return (transformed * 255).astype(np.uint8)


def degrade(values: np.ndarray, task: str, seed: int) -> np.ndarray:
    image = Image.fromarray(values)
    task = task.lower()
    if "blurx2" in task:
        image = image.resize((values.shape[1] // 2, values.shape[0] // 2), Image.Resampling.BICUBIC)
        return np.asarray(image.resize(values.shape[::-1], Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(1.0)))
    rng = np.random.default_rng(seed)
    array = values.astype(np.float32)
    if "gaussian" in task:
        array += rng.normal(0, 18, array.shape)
    else:
        draw = rng.random(array.shape)
        array[draw < 0.015] = 0
        array[draw > 0.985] = 255
    return np.clip(array, 0, 255).astype(np.uint8)


def save_grayscale(path: Path, values: np.ndarray):
    Image.fromarray(values, mode="L").convert("RGB").save(path, optimize=True)


def write_example(root: Path, family: str, example_id: str, task: str, size: int, seed: int):
    directory = root / family / example_id
    directory.mkdir(parents=True, exist_ok=True)
    if family == "translation":
        exemplar, _ = synthetic_mri(size, seed)
        query, _ = synthetic_mri(size, seed + 100)
        exemplar_target = translate(exemplar, task)
        exemplar_source, query_source = exemplar, query
    elif family == "enhancement":
        exemplar_target, _ = synthetic_mri(size, seed)
        query_target, _ = synthetic_mri(size, seed + 100)
        exemplar_source = degrade(exemplar_target, task, seed)
        query_source = degrade(query_target, task, seed + 100)
    else:
        exemplar_source, exemplar_mask = synthetic_mri(size, seed, lesion=True)
        query_source, _ = synthetic_mri(size, seed + 100, lesion=True)
        exemplar_target = np.where(exemplar_mask, 255, 0).astype(np.uint8)
    save_grayscale(directory / "exemplar_source.png", exemplar_source)
    save_grayscale(directory / "exemplar_target.png", exemplar_target)
    save_grayscale(directory / "query_source.png", query_source)
    config = {
        "example_id": example_id,
        "family": family,
        "task": task,
        "exemplar_source": "exemplar_source.png",
        "exemplar_target": "exemplar_target.png",
        "query_source": "query_source.png",
        "synthetic": True,
    }
    (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {"example_id": example_id, "family": family, "task": task, "directory": f"{family}/{example_id}"}


def main():
    parser = argparse.ArgumentParser(description="Generate the 18 synthetic MagicMRI examples")
    parser.add_argument("--output-root", default="examples")
    parser.add_argument("--image-size", type=int, default=448)
    args = parser.parse_args()
    root = Path(args.output_root)
    rows = []
    for family, definitions in (
        ("translation", TRANSLATION),
        ("enhancement", ENHANCEMENT),
        ("segmentation", SEGMENTATION),
    ):
        for ordinal, (example_id, task) in enumerate(definitions, 1):
            rows.append(write_example(root, family, example_id, task, args.image_size, ordinal + len(rows) * 7))
    (root / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(rows)} synthetic visual-prompt examples in {root}")


if __name__ == "__main__":
    main()
