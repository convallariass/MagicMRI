#!/usr/bin/env python3
"""Core metric definitions retained from the verified project evaluators."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.metrics import structural_similarity


def load_image(path: Path, grayscale: bool = False) -> np.ndarray:
    mode = "L" if grayscale else "RGB"
    with Image.open(path) as image:
        return np.asarray(image.convert(mode))


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    """Grayscale SSIM with the fixed 8-bit range used by core evaluation."""
    if prediction.ndim == 3:
        prediction = np.asarray(Image.fromarray(prediction).convert("L"))
        target = np.asarray(Image.fromarray(target).convert("L"))
    return float(structural_similarity(prediction, target, data_range=255))


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    # Preserve the released paper evaluator's uint8 arithmetic exactly.
    mse = float(np.mean((prediction.astype(np.uint8) - target.astype(np.uint8)) ** 2))
    return 100.0 if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))


def nmae(prediction: np.ndarray, target: np.ndarray) -> float:
    difference = np.abs(prediction.astype(np.float64) - target.astype(np.float64))
    return float(np.mean(difference) / 255.0)


class LPIPSEvaluator:
    """Reuse one AlexNet LPIPS network for a dataset evaluation."""

    def __init__(self, device: str):
        import lpips

        print(
            "LPIPS uses pretrained AlexNet weights and may download them on first use; "
            "PyTorch cache is under torch.hub.get_dir()/checkpoints.",
            file=sys.stderr,
        )
        self.device = device
        self.model = lpips.LPIPS(net="alex").to(device).eval()

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        import torch

        tensors = []
        for image in (prediction, target):
            # The retained evaluator used cv2.imread arrays (BGR channel order).
            bgr = np.ascontiguousarray(image[..., ::-1])
            tensor = torch.from_numpy(bgr).permute(2, 0, 1)
            tensors.append(tensor.unsqueeze(0).float().div(255.0).to(self.device))
        with torch.no_grad():
            return float(self.model(tensors[0], tensors[1]).item())


def lpips_score(prediction: np.ndarray, target: np.ndarray, device: str = "cpu") -> float:
    return LPIPSEvaluator(device)(prediction, target)


def binary(array: np.ndarray, threshold: int = 127) -> np.ndarray:
    return np.asarray(array) > threshold


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = binary(prediction), binary(target)
    denominator = int(prediction.sum() + target.sum())
    return 1.0 if denominator == 0 else float(
        2 * np.logical_and(prediction, target).sum() / denominator
    )


def miou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = binary(prediction), binary(target)
    values = []
    for predicted_class, target_class in ((prediction, target), (~prediction, ~target)):
        union = int(np.logical_or(predicted_class, target_class).sum())
        intersection = int(np.logical_and(predicted_class, target_class).sum())
        values.append(0.0 if union == 0 else intersection / union)
    return float(np.mean(values))


def pacc(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = binary(prediction), binary(target)
    return float(np.mean(prediction == target))


def hd95(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = binary(prediction), binary(target)
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float("nan")
    prediction_surface = prediction ^ binary_erosion(prediction)
    target_surface = target ^ binary_erosion(target)
    distances = np.concatenate(
        (
            distance_transform_edt(~target_surface)[prediction_surface],
            distance_transform_edt(~prediction_surface)[target_surface],
        )
    )
    return float(np.percentile(distances, 95))


def generation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    lpips_evaluator: Optional[LPIPSEvaluator] = None,
) -> Dict[str, float]:
    result = {
        "SSIM": ssim(prediction, target),
        "PSNR": psnr(prediction, target),
        "NMAE": nmae(prediction, target),
    }
    if lpips_evaluator is not None:
        result["LPIPS"] = lpips_evaluator(prediction, target)
    return result


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    return {
        "Dice": dice(prediction, target),
        "mIoU": miou(prediction, target),
        "pACC": pacc(prediction, target),
        "HD95": hd95(prediction, target),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate one MagicMRI prediction-target pair")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--task", choices=("image", "segmentation"), required=True)
    parser.add_argument("--lpips", action="store_true", help="Also compute LPIPS")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    grayscale = args.task == "segmentation"
    prediction = load_image(args.prediction, grayscale)
    target = load_image(args.target, grayscale)
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} versus {target.shape}")
    if args.task == "segmentation":
        result = segmentation_metrics(prediction, target)
    else:
        evaluator = LPIPSEvaluator(args.device) if args.lpips else None
        result = generation_metrics(prediction, target, evaluator)
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
