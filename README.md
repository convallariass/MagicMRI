# MagicMRI

MagicMRI is a visual-prompt MRI model for modality translation, enhancement/restoration, and tumor segmentation. A source-target exemplar pair specifies the requested operation, and the model predicts the corresponding target for a query source image.

## Installation

Python 3.8.18, PyTorch 2.0, and CUDA 11.8 were used for the reference implementation.

```bash
python -m pip install -r requirements.txt
```

## Data preparation

Obtain MRI data from its authorized source. Do not place restricted clinical data in this repository. Convert an aligned source-target NIfTI pair into normalized 2D slices and a generic training manifest with:

```bash
python scripts/prepare_data.py \
  --source /path/to/source.nii.gz \
  --target /path/to/target.nii.gz \
  --task-type modality_translation-t1cTOt1n \
  --output-dir /path/to/prepared/task
```

Use `--target-is-mask` for segmentation targets or `--source-degradation` for restoration tasks. A training manifest is a JSON list with `image_path`, `target_path`, and `type` fields. Paths are resolved relative to the configured data root.

The degradation options create training pairs from user-owned data; evaluation consumes saved predictions and targets and does not generate degradations.

## Training

Set the data paths in `configs/train.yaml`, then run:

```bash
python scripts/train.py --config configs/train.yaml
```

An initialization checkpoint can be supplied with `--init-checkpoint`; interrupted training can be continued with `--resume`.

## Inference

Each inference call consumes an exemplar source, an exemplar target, and a query source. Run a custom example directory with:

```bash
python scripts/inference.py \
  --example /path/to/example \
  --checkpoint checkpoints/magicmri_ckpt_release.pth
```

The example directory must contain `config.yaml` and the three images referenced by that file.

## Visual-prompt inference examples

The repository includes 18 deterministic synthetic examples: 6 modality-translation examples, 6 enhancement/restoration examples, and 6 tumor-segmentation examples. They contain no clinical data and are intended only to demonstrate the inference interface.

```bash
# Translation
python scripts/inference.py --family translation --checkpoint checkpoints/magicmri_ckpt_release.pth

# Enhancement/restoration
python scripts/inference.py --family enhancement --checkpoint checkpoints/magicmri_ckpt_release.pth

# Tumor segmentation
python scripts/inference.py --family segmentation --checkpoint checkpoints/magicmri_ckpt_release.pth
```

Validate all example files without a checkpoint using:

```bash
python scripts/inference.py --all-examples --validate-only
```

## Evaluation

The minimal evaluator provides SSIM, PSNR, NMAE, optional LPIPS, Dice, mIoU, and HD95 for MagicMRI outputs.

```bash
python evaluation/metrics.py \
  --prediction /path/to/prediction.png \
  --target /path/to/target.png \
  --task image
```

## Pretrained checkpoint

CHECKPOINT_DOWNLOAD_URL_TO_BE_ADDED

The pretrained checkpoint can be downloaded from the link above and placed at:
`checkpoints/magicmri_ckpt_release.pth`

The `checkpoints/` directory is ignored by Git.

## Citation

MagicMRI
