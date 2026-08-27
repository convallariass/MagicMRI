from pathlib import Path
from typing import Optional, Tuple, Union

import nibabel as nib
import numpy as np
from PIL import Image, ImageFilter


PathLike = Union[str, Path]


def load_volume(path: PathLike) -> np.ndarray:
    """Load a NIfTI volume without retaining clinical header fields."""
    data = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, received shape {data.shape}")
    return data


def normalize_mri(volume: np.ndarray, lower: float = 0.5, upper: float = 99.5) -> np.ndarray:
    """Robustly map finite non-background MRI intensities to [0, 1]."""
    volume = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(volume)
    foreground = finite & (volume != 0)
    values = volume[foreground]
    if values.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    low, high = np.percentile(values, [lower, upper])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((np.where(finite, volume, low) - low) / (high - low), 0.0, 1.0)
    normalized[~foreground] = 0.0
    return normalized.astype(np.float32)


def volume_slice(volume: np.ndarray, index: int, axis: int = 2) -> np.ndarray:
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if not 0 <= index < volume.shape[axis]:
        raise IndexError(f"slice {index} is outside axis {axis} with length {volume.shape[axis]}")
    return np.take(volume, index, axis=axis)


def to_uint8_image(array: np.ndarray, size: int = 448, is_mask: bool = False) -> Image.Image:
    if is_mask:
        image = Image.fromarray(np.where(array > 0, 255, 0).astype(np.uint8), mode="L")
        return image.resize((size, size), Image.Resampling.NEAREST)
    image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L")
    return image.resize((size, size), Image.Resampling.BICUBIC)


def degrade_image(image: Image.Image, kind: str, seed: Optional[int] = None) -> Image.Image:
    """Create one of the degradation families used by the restoration tasks."""
    if kind == "none":
        return image
    if kind == "blurx2":
        reduced = image.resize((image.width // 2, image.height // 2), Image.Resampling.BICUBIC)
        return reduced.resize(image.size, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(1.0))
    rng = np.random.default_rng(seed)
    values = np.asarray(image, dtype=np.float32) / 255.0
    if kind == "gaussian":
        values = np.clip(values + rng.normal(0.0, 0.08, values.shape), 0.0, 1.0)
    elif kind == "salt_pepper":
        draw = rng.random(values.shape)
        values[draw < 0.015] = 0.0
        values[draw > 0.985] = 1.0
    else:
        raise ValueError("degradation must be one of: none, blurx2, gaussian, salt_pepper")
    return Image.fromarray((values * 255.0).astype(np.uint8), mode="L")
