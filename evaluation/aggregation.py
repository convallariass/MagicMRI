"""Fail-closed aggregation for Core-36 per-slice metric records."""

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence


METRICS = {
    "generation": ("SSIM", "PSNR", "NMAE", "LPIPS"),
    "segmentation": ("Dice", "mIoU", "pACC", "HD95"),
}


def finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float("nan") if not finite else sum(finite) / len(finite)


def _group(rows: Iterable[Dict[str, Any]], keys: Sequence[str]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def patient_summaries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for (task_id, family, metric_family, patient_id), group in sorted(
        _group(rows, ("task_id", "family", "metric_family", "patient_id")).items()
    ):
        summary = {
            "task_id": task_id,
            "family": family,
            "metric_family": metric_family,
            "patient_id": patient_id,
            "n_slices": len(group),
        }
        for metric in METRICS[metric_family]:
            summary[metric] = finite_mean(row[metric] for row in group)
            summary[f"{metric}_n_finite"] = sum(
                math.isfinite(float(row[metric])) for row in group
            )
        output.append(summary)
    return output


def task_summaries(slice_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for (task_id, family, metric_family), group in sorted(
        _group(slice_rows, ("task_id", "family", "metric_family")).items()
    ):
        summary = {
            "task_id": task_id,
            "family": family,
            "metric_family": metric_family,
            "n_patients": len({row["patient_id"] for row in group}),
            "n_slices": len(group),
        }
        for metric in METRICS[metric_family]:
            summary[metric] = finite_mean(row[metric] for row in group)
            summary[f"{metric}_n_slices"] = sum(
                math.isfinite(float(row[metric])) for row in group
            )
        output.append(summary)
    return output


def family_summaries(task_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for (family, metric_family), group in sorted(
        _group(task_rows, ("family", "metric_family")).items()
    ):
        summary = {
            "family": family,
            "metric_family": metric_family,
            "n_tasks": len(group),
        }
        for metric in METRICS[metric_family]:
            summary[metric] = finite_mean(row[metric] for row in group)
            summary[f"{metric}_n_tasks"] = sum(
                math.isfinite(float(row[metric])) for row in group
            )
        output.append(summary)
    return output
