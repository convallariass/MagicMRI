from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch

from .utils.visual_prompt import IMAGENET_MEAN, IMAGENET_STD, construct_visual_prompt


PathLike = Union[str, Path]


def infer(
    model: torch.nn.Module,
    paths: Dict[str, PathLike],
    device: torch.device,
    input_size: int = 448,
) -> np.ndarray:
    """Run one source-target visual exemplar and query through MagicMRI."""
    source, target, masked = construct_visual_prompt(
        paths["exemplar_source"],
        paths["exemplar_target"],
        paths["query_source"],
        input_size,
        device,
    )
    with torch.no_grad():
        latent = model.forward_encoder(source, target, masked.flatten(1))
        prediction = model.forward_decoder(latent)
    output = prediction.permute(0, 2, 3, 1).cpu().numpy()[0]
    output = output[output.shape[0] // 2 :]
    return np.clip((output * IMAGENET_STD + IMAGENET_MEAN) * 255.0, 0, 255).astype(np.uint8)
