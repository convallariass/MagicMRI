#!/usr/bin/env python3
"""Validate user-supplied Core-36 bindings and emit a local JSONL manifest."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magicmri.tasks import task_index  # noqa: E402


FIELDS = (
    "sample_id",
    "task_id",
    "patient_id",
    "slice_id",
    "slice_index",
    "exemplar_source",
    "exemplar_target",
    "query_source",
    "query_target",
)
PATH_FIELDS = ("exemplar_source", "exemplar_target", "query_source", "query_target")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a fail-closed Core-36 manifest without copying BraTS data"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-registry", type=Path, default=ROOT / "configs/core36_tasks.yaml")
    parser.add_argument(
        "--allow-task-subset",
        action="store_true",
        help="Allow a manifest that does not contain all 36 registered tasks",
    )
    return parser.parse_args()


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def main():
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root is not a directory: {data_root}")
    if not args.bindings.is_file():
        raise FileNotFoundError(f"Bindings file not found: {args.bindings}")
    tasks = task_index(args.task_registry)
    rows = []
    errors = []
    seen_samples = set()
    for line_number, line in enumerate(args.bindings.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"line {line_number}: invalid JSON ({error})")
            continue
        missing = [field for field in FIELDS if field not in row]
        if missing:
            errors.append(f"line {line_number}: missing fields {missing}")
            continue
        if row["task_id"] not in tasks:
            errors.append(f"line {line_number}: unknown task_id {row['task_id']}")
        if row["sample_id"] in seen_samples:
            errors.append(f"line {line_number}: duplicate sample_id {row['sample_id']}")
        seen_samples.add(row["sample_id"])
        if not isinstance(row["slice_index"], int) or row["slice_index"] < 0:
            errors.append(f"line {line_number}: slice_index must be a non-negative integer")
        normalized = {field: row[field] for field in FIELDS}
        for field in PATH_FIELDS:
            supplied = Path(str(row[field])).expanduser()
            candidate = supplied.resolve() if supplied.is_absolute() else (data_root / supplied).resolve()
            if not _inside(data_root, candidate):
                errors.append(f"line {line_number}: {field} escapes data root")
            elif not candidate.is_file():
                errors.append(f"line {line_number}: missing {field}: {candidate}")
            normalized[field] = str(candidate)
        rows.append(normalized)
    if not rows:
        errors.append("bindings contain no records")
    present = {row["task_id"] for row in rows}
    missing_tasks = sorted(set(tasks).difference(present))
    if missing_tasks and not args.allow_task_subset:
        errors.append(f"bindings omit {len(missing_tasks)} Core-36 tasks: {missing_tasks}")
    if errors:
        raise RuntimeError("Manifest validation failed:\n- " + "\n- ".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} validated records across {len(present)} task(s): {args.output}")


if __name__ == "__main__":
    main()
