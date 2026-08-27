from pathlib import Path
from typing import Any, Dict, Union

import torch

from magicmri.models import magicmri_vit_large_patch16_input896x448


PathLike = Union[str, Path]


def load_checkpoint(path: PathLike) -> Dict[str, torch.Tensor]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Download it and place it at "
            "checkpoints/magicmri_ckpt_release.pth."
        )
    checkpoint: Any = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        state = checkpoint
    else:
        raise RuntimeError("Unsupported checkpoint format; expected a model state dictionary.")
    return state


def load_pretrained_model(path: PathLike, device: torch.device):
    model = magicmri_vit_large_patch16_input896x448()
    result = model.load_state_dict(load_checkpoint(path), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {result}")
    return model.to(device).eval()
