#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magicmri.data.dataset import VisualPairDataset  # noqa: E402
from magicmri.models import magicmri_vit_large_patch16_input896x448  # noqa: E402
from magicmri.utils.checkpoint import load_checkpoint  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train the MagicMRI visual-prompt model")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--train-manifest", action="append")
    parser.add_argument("--val-manifest", action="append")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    return parser.parse_args()


def parameter_groups(model, weight_decay: float, layer_decay: float):
    groups: Dict[str, Dict] = {}
    num_layers = len(model.blocks) + 1
    scales = [layer_decay ** (num_layers - index) for index in range(num_layers + 1)]
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in {"pos_embed", "cls_token"} or name.startswith("patch_embed"):
            layer_id = 0
        elif name.startswith("blocks."):
            layer_id = int(name.split(".")[1]) + 1
        else:
            layer_id = num_layers
        decay = 0.0 if parameter.ndim == 1 or name in model.no_weight_decay() else weight_decay
        key = f"{layer_id}:{decay}"
        groups.setdefault(
            key,
            {"params": [], "weight_decay": decay, "lr_scale": scales[layer_id]},
        )["params"].append(parameter)
    return list(groups.values())


def cosine_lr(optimizer, progress: float, base_lr: float, min_lr: float, warmup: int, epochs: int):
    if progress < warmup:
        lr = base_lr * progress / max(warmup, 1)
    else:
        phase = (progress - warmup) / max(epochs - warmup, 1)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * phase))
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)


def move_batch(batch, device):
    return tuple(value.to(device, non_blocking=True) for value in batch)


@torch.no_grad()
def validate(model, loader, device, amp_enabled: bool) -> float:
    model.eval()
    losses: List[float] = []
    for batch in loader:
        images, targets, mask, valid = move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            loss, _, _ = model(images, targets, mask, valid)
        losses.append(float(loss.item()))
    return sum(losses) / max(len(losses), 1)


def main():
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data = config["data"]
    train_cfg = config["training"]
    data_root = Path(args.data_root or data["root"])
    train_manifests = args.train_manifest or data["train_manifests"]
    val_manifests = args.val_manifest or data["val_manifests"]
    output_dir = Path(args.output_dir or train_cfg["output_dir"])
    device = torch.device(args.device or train_cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(int(config["seed"]))
    train_set = VisualPairDataset(
        data_root,
        train_manifests,
        image_size=int(data["image_size"]),
        training=True,
        half_mask_ratio=float(data["half_mask_ratio"]),
        min_random_scale=float(data["min_random_scale"]),
        num_mask_patches=int(data["num_mask_patches"]),
        max_mask_patches_per_block=int(data["max_mask_patches_per_block"]),
    )
    val_set = VisualPairDataset(
        data_root,
        val_manifests,
        image_size=int(data["image_size"]),
        training=False,
        half_mask_ratio=1.0,
        num_mask_patches=int(data["num_mask_patches"]),
        max_mask_patches_per_block=int(data["max_mask_patches_per_block"]),
    )
    sampler = WeightedRandomSampler(train_set.weights, len(train_set), replacement=True)
    loader_args = {
        "batch_size": int(train_cfg["batch_size"]),
        "num_workers": int(train_cfg["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, sampler=sampler, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    model = magicmri_vit_large_patch16_input896x448().to(device)
    if args.init_checkpoint:
        model.load_state_dict(load_checkpoint(args.init_checkpoint), strict=False)

    groups = parameter_groups(
        model, float(train_cfg["weight_decay"]), float(train_cfg["layer_decay"])
    )
    optimizer = torch.optim.AdamW(groups, lr=float(train_cfg["learning_rate"]))
    amp_enabled = device.type == "cuda" and bool(train_cfg["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(train_cfg["epochs"])
    accumulation = int(train_cfg["accumulation_steps"])
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            progress = epoch + step / max(len(train_loader), 1)
            cosine_lr(
                optimizer,
                progress,
                float(train_cfg["learning_rate"]),
                float(train_cfg["min_learning_rate"]),
                int(train_cfg["warmup_epochs"]),
                epochs,
            )
            images, targets, mask, valid = move_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss, _, _ = model(images, targets, mask, valid)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["clip_grad"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.item())
        val_loss = validate(model, val_loader, device, amp_enabled)
        print(
            f"epoch={epoch + 1} train_loss={running / max(len(train_loader), 1):.6f} "
            f"val_loss={val_loss:.6f}"
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
            },
            output_dir / f"checkpoint_{epoch + 1:03d}.pth",
        )


if __name__ == "__main__":
    main()
