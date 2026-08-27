#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Convert authorized NIfTI volumes to MagicMRI slices")
    parser.add_argument("--source", required=True, help="Source NIfTI volume")
    parser.add_argument("--target", required=True, help="Target NIfTI volume or mask")
    parser.add_argument("--task-type", required=True, help="Task name stored in the training manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--slice-indices", default="all", help="Comma-separated indices or 'all'")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--target-is-mask", action="store_true")
    parser.add_argument(
        "--source-degradation",
        choices=("none", "blurx2", "gaussian", "salt_pepper"),
        default="none",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-empty", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    from magicmri.data.preprocessing import (
        degrade_image,
        load_volume,
        normalize_mri,
        to_uint8_image,
        volume_slice,
    )

    source_volume = normalize_mri(load_volume(args.source))
    target_raw = load_volume(args.target)
    target_volume = target_raw if args.target_is_mask else normalize_mri(target_raw)
    if source_volume.shape != target_volume.shape:
        raise ValueError(
            f"Source and target volumes must be aligned; got {source_volume.shape} and {target_volume.shape}"
        )
    if args.slice_indices == "all":
        indices = range(source_volume.shape[args.axis])
    else:
        indices = [int(value) for value in args.slice_indices.split(",") if value.strip()]

    output_dir = Path(args.output_dir)
    source_dir = output_dir / "source"
    target_dir = output_dir / "target"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in indices:
        source_slice = volume_slice(source_volume, index, args.axis)
        target_slice = volume_slice(target_volume, index, args.axis)
        if args.skip_empty and not (source_slice != 0).any():
            continue
        source_image = to_uint8_image(source_slice, args.image_size)
        source_image = degrade_image(source_image, args.source_degradation, args.seed + index)
        target_image = to_uint8_image(target_slice, args.image_size, args.target_is_mask)
        source_path = source_dir / f"slice_{index:04d}.png"
        target_path = target_dir / f"slice_{index:04d}.png"
        source_image.save(source_path)
        target_image.save(target_path)
        records.append(
            {
                "image_path": source_path.relative_to(output_dir).as_posix(),
                "target_path": target_path.relative_to(output_dir).as_posix(),
                "type": args.task_type,
            }
        )
    manifest = output_dir / args.manifest
    manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} pairs to {manifest}")


if __name__ == "__main__":
    main()
