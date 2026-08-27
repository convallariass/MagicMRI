from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PathLike = Union[str, Path]


def load_rgb(path: PathLike, size: int) -> np.ndarray:
    """Load one image as an RGB float array in [0, 1]."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size))
        return np.asarray(image, dtype=np.float32) / 255.0


def construct_visual_prompt(
    exemplar_source: PathLike,
    exemplar_target: PathLike,
    query_source: PathLike,
    size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the production 2H x H visual-prompt canvas.

    The source canvas contains the exemplar source above the query source. The
    target canvas contains the visible exemplar target in both halves; the
    lower half is then replaced by mask tokens inside the model.
    """
    source_exemplar = load_rgb(exemplar_source, size)
    target_exemplar = load_rgb(exemplar_target, size)
    query = load_rgb(query_source, size)
    source_canvas = np.concatenate((source_exemplar, query), axis=0)
    target_canvas = np.concatenate((target_exemplar, target_exemplar), axis=0)
    source_canvas = (source_canvas - IMAGENET_MEAN) / IMAGENET_STD
    target_canvas = (target_canvas - IMAGENET_MEAN) / IMAGENET_STD

    source = torch.from_numpy(source_canvas).permute(2, 0, 1).unsqueeze(0)
    target = torch.from_numpy(target_canvas).permute(2, 0, 1).unsqueeze(0)
    source = source.to(device=device, dtype=torch.float32)
    target = target.to(device=device, dtype=torch.float32)

    num_patches = (size * 2 * size) // (16**2)
    masked = torch.zeros(num_patches, dtype=torch.bool, device=device)
    masked[num_patches // 2 :] = True
    return source, target, masked.unsqueeze(0)


def resolve_example_inputs(example_dir: PathLike, config: Dict[str, str]) -> Dict[str, Path]:
    directory = Path(example_dir)
    paths = {
        "exemplar_source": directory / config["exemplar_source"],
        "exemplar_target": directory / config["exemplar_target"],
        "query_source": directory / config["query_source"],
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing visual-prompt input(s): " + ", ".join(missing))
    return paths
