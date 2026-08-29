from pathlib import Path
from typing import Dict, List, Union

import yaml


PathLike = Union[str, Path]
REQUIRED_TASK_FIELDS = {
    "task_id",
    "family",
    "source_modality",
    "output_type",
    "expected_exemplar_relation",
    "expected_query_input",
    "metric_family",
}


def load_task_registry(path: PathLike) -> List[Dict[str, str]]:
    registry_path = Path(path)
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "core36-1.0":
        raise ValueError(f"Unsupported task registry schema: {registry_path}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"Task registry has no task list: {registry_path}")
    identifiers = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Task {index} is not a mapping")
        missing = REQUIRED_TASK_FIELDS.difference(task)
        if missing:
            raise ValueError(f"Task {index} is missing fields: {sorted(missing)}")
        family_specific = {
            "translation": "target_modality",
            "enhancement": "degradation",
            "segmentation": "tumor_target",
        }
        family = task["family"]
        if family not in family_specific:
            raise ValueError(f"Unknown family for {task['task_id']}: {family}")
        required = family_specific[family]
        if required not in task:
            raise ValueError(f"{task['task_id']} is missing {required}")
        identifiers.append(task["task_id"])
    if len(tasks) != 36:
        raise ValueError(f"Expected 36 tasks, found {len(tasks)}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Task IDs are not unique")
    counts = {family: sum(task["family"] == family for task in tasks) for family in family_specific}
    if counts != {"translation": 12, "enhancement": 12, "segmentation": 12}:
        raise ValueError(f"Invalid Core-36 family counts: {counts}")
    return tasks


def task_index(path: PathLike) -> Dict[str, Dict[str, str]]:
    return {task["task_id"]: task for task in load_task_registry(path)}
