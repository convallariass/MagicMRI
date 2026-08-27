#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.metrics import structural_similarity


def _image(path: Path, grayscale: bool = False) -> np.ndarray:
    mode = "L" if grayscale else "RGB"
    with Image.open(path) as image:
        return np.asarray(image.convert(mode))


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.ndim == 3:
        return float(structural_similarity(prediction, target, channel_axis=2, data_range=255))
    return float(structural_similarity(prediction, target, data_range=255))


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(difference**2))
    return float("inf") if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))


def nmae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction.astype(np.float64) - target.astype(np.float64))) / 255.0)


def lpips_score(prediction: np.ndarray, target: np.ndarray, device: str = "cpu") -> float:
    import lpips
    import torch

    model = lpips.LPIPS(net="alex").to(device).eval()
    tensors = []
    for image in (prediction, target):
        tensor = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0)
        tensors.append(tensor.permute(2, 0, 1).unsqueeze(0).to(device))
    with torch.no_grad():
        return float(model(tensors[0], tensors[1]).item())


def _binary(array: np.ndarray, threshold: int = 127) -> np.ndarray:
    return np.asarray(array) > threshold


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = _binary(prediction), _binary(target)
    denominator = int(prediction.sum() + target.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(prediction, target).sum() / denominator)


def miou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = _binary(prediction), _binary(target)
    foreground_union = np.logical_or(prediction, target).sum()
    background_union = np.logical_or(~prediction, ~target).sum()
    foreground = 1.0 if foreground_union == 0 else np.logical_and(prediction, target).sum() / foreground_union
    background = 1.0 if background_union == 0 else np.logical_and(~prediction, ~target).sum() / background_union
    return float((foreground + background) / 2.0)


def hd95(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = _binary(prediction), _binary(target)
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float("inf")
    prediction_surface = prediction ^ binary_erosion(prediction)
    target_surface = target ^ binary_erosion(target)
    distances = np.concatenate(
        (
            distance_transform_edt(~target_surface)[prediction_surface],
            distance_transform_edt(~prediction_surface)[target_surface],
        )
    )
    return float(np.percentile(distances, 95))


def main():
    parser = argparse.ArgumentParser(description="Evaluate MagicMRI outputs")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--task", choices=("image", "segmentation"), required=True)
    parser.add_argument("--lpips", action="store_true", help="Also compute LPIPS")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    grayscale = args.task == "segmentation"
    prediction, target = _image(args.prediction, grayscale), _image(args.target, grayscale)
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} versus {target.shape}")
    if args.task == "segmentation":
        result = {"Dice": dice(prediction, target), "mIoU": miou(prediction, target), "HD95": hd95(prediction, target)}
    else:
        result = {"SSIM": ssim(prediction, target), "PSNR": psnr(prediction, target), "NMAE": nmae(prediction, target)}
        if args.lpips:
            result["LPIPS"] = lpips_score(prediction, target, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
