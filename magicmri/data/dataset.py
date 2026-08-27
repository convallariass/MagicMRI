import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, RandomResizedCrop

from .masking import MaskingGenerator


PathLike = Union[str, Path]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest must be a non-empty JSON list: {path}")
    required = {"image_path", "target_path", "type"}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError(f"Manifest record {index} must contain {sorted(required)}")
    return records


def _load_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return image.convert("RGB")


def _shared_color_jitter(source: Image.Image, target: Image.Image) -> Tuple[Image.Image, Image.Image]:
    operations = torch.randperm(4).tolist()
    brightness = random.uniform(0.6, 1.4)
    contrast = random.uniform(0.6, 1.4)
    saturation = random.uniform(0.8, 1.2)
    hue = random.uniform(-0.1, 0.1)
    for operation in operations:
        if operation == 0:
            source, target = TF.adjust_brightness(source, brightness), TF.adjust_brightness(target, brightness)
        elif operation == 1:
            source, target = TF.adjust_contrast(source, contrast), TF.adjust_contrast(target, contrast)
        elif operation == 2:
            source, target = TF.adjust_saturation(source, saturation), TF.adjust_saturation(target, saturation)
        else:
            source, target = TF.adjust_hue(source, hue), TF.adjust_hue(target, hue)
    return source, target


def _paired_transform(
    source: Image.Image,
    target: Image.Image,
    image_size: int,
    training: bool,
    min_random_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if training:
        top, left, height, width = RandomResizedCrop.get_params(
            source, scale=(min_random_scale, 1.0), ratio=(0.75, 4.0 / 3.0)
        )
        source = TF.resized_crop(
            source, top, left, height, width, [image_size, image_size], InterpolationMode.BICUBIC
        )
        target = TF.resized_crop(
            target, top, left, height, width, [image_size, image_size], InterpolationMode.BICUBIC
        )
        if random.random() < 0.8:
            source, target = _shared_color_jitter(source, target)
        if random.random() < 0.5:
            source, target = TF.hflip(source), TF.hflip(target)
    else:
        source = TF.resize(source, [image_size, image_size], InterpolationMode.BICUBIC)
        target = TF.resize(target, [image_size, image_size], InterpolationMode.BICUBIC)
    source_tensor = TF.normalize(TF.to_tensor(source), MEAN, STD)
    target_tensor = TF.normalize(TF.to_tensor(target), MEAN, STD)
    return source_tensor, target_tensor


class VisualPairDataset(Dataset):
    """Task-balanced source-target pairs with an in-context exemplar pair.

    Each manifest record has ``image_path``, ``target_path``, and ``type``.
    A second record of the same type is sampled to form the 896 x 448 canvas.
    """

    def __init__(
        self,
        root: PathLike,
        manifests: Sequence[PathLike],
        image_size: int = 448,
        training: bool = True,
        half_mask_ratio: float = 0.1,
        min_random_scale: float = 0.3,
        num_mask_patches: int = 784,
        max_mask_patches_per_block: int = 392,
    ) -> None:
        self.root = Path(root)
        self.records: List[Dict[str, str]] = []
        for manifest in manifests:
            self.records.extend(_read_manifest(Path(manifest)))
        self.image_size = image_size
        self.training = training
        self.half_mask_ratio = half_mask_ratio
        self.min_random_scale = min_random_scale

        self.by_type: Dict[str, List[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            self.by_type[record["type"]].append(index)
        n_types = len(self.by_type)
        self.weights = [0.0] * len(self.records)
        for indices in self.by_type.values():
            weight = 1.0 / n_types / len(indices)
            for index in indices:
                self.weights[index] = weight

        patch_shape = (image_size * 2 // 16, image_size // 16)
        self.mask_generator = MaskingGenerator(
            patch_shape,
            num_masking_patches=num_mask_patches,
            min_num_patches=16,
            max_num_patches=max_mask_patches_per_block,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _pair(self, record: Dict[str, str]) -> Tuple[torch.Tensor, torch.Tensor]:
        source = _load_rgb(self.root / record["image_path"])
        target = _load_rgb(self.root / record["target_path"])
        return _paired_transform(
            source, target, self.image_size, self.training, self.min_random_scale
        )

    def __getitem__(self, index: int):
        primary = self.records[index]
        query_index = random.choice(self.by_type[primary["type"]])
        source_top, target_top = self._pair(primary)
        source_bottom, target_bottom = self._pair(self.records[query_index])
        source = torch.cat((source_top, source_bottom), dim=1)
        target = torch.cat((target_top, target_bottom), dim=1)
        valid = torch.ones_like(target)

        if random.random() < self.half_mask_ratio:
            mask = np.zeros(self.mask_generator.get_shape(), dtype=np.int32)
            mask[mask.shape[0] // 2 :, :] = 1
        else:
            mask = self.mask_generator()
        return source, target, torch.from_numpy(mask), valid
