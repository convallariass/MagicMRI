import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union


PathLike = Union[str, Path]
INFERENCE_FIELDS = (
    "sample_id",
    "task_id",
    "patient_id",
    "slice_id",
    "slice_index",
    "exemplar_source",
    "exemplar_target",
    "query_source",
)
PATH_FIELDS = ("exemplar_source", "exemplar_target", "query_source", "query_target")


def read_jsonl(path: PathLike) -> Iterable[Dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {manifest_path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Manifest row {line_number} is not an object")
        yield row


def load_inference_manifest(
    path: PathLike,
    task_ids: Iterable[str],
    require_target: bool = False,
) -> List[Dict[str, Any]]:
    manifest_path = Path(path).resolve()
    allowed_tasks = set(task_ids)
    required = set(INFERENCE_FIELDS)
    if require_target:
        required.add("query_target")
    rows = []
    seen = set()
    for line_number, row in enumerate(read_jsonl(manifest_path), 1):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Manifest row {line_number} is missing: {sorted(missing)}")
        if row["task_id"] not in allowed_tasks:
            raise ValueError(f"Manifest row {line_number} has unknown task_id: {row['task_id']}")
        if row["sample_id"] in seen:
            raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
        seen.add(row["sample_id"])
        for field in ("sample_id", "task_id", "patient_id", "slice_id"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"Manifest row {line_number} has invalid {field}")
        if not isinstance(row["slice_index"], int) or row["slice_index"] < 0:
            raise ValueError(f"Manifest row {line_number} has invalid slice_index")
        normalized = dict(row)
        for field in PATH_FIELDS:
            if field not in row:
                continue
            supplied = Path(str(row[field])).expanduser()
            candidate = supplied if supplied.is_absolute() else manifest_path.parent / supplied
            candidate = candidate.resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"Manifest row {line_number} missing {field}: {candidate}")
            normalized[field] = str(candidate)
        rows.append(normalized)
    if not rows:
        raise ValueError(f"Manifest contains no records: {manifest_path}")
    return rows
