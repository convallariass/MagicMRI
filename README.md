# MagicMRI

MagicMRI is a 2D visual-prompt MRI model for modality translation, enhancement/restoration, and tumor segmentation. The public release provides:

1. the released model checkpoint;
2. the inference implementation;
3. machine-readable definitions for the 36 core tasks;
4. Core-36 inference and evaluation code; and
5. 18 synthetic smoke-test fixtures.

It does not include raw BraTS data, held-out project-subset reproduction, composite evaluation, downstream classification, private patient manifests, or full training-from-scratch reproduction.

## Release scope

A visual prompt consists of an exemplar source-target pair plus a query source. The requested operation is defined by the relationship between exemplar source and exemplar target. MagicMRI predicts the corresponding target for the query source.

The frozen [Core-36 registry](configs/core36_tasks.yaml) covers 12 modality translations, 12 enhancement/restoration tasks, and 12 tumor-segmentation tasks. This repository supports slice-level inference and core evaluation for those tasks only.

## Installation

The reference environment is Python 3.8.18, PyTorch 2.0.0, and CUDA 11.8. Create a fresh Python 3.8 environment, clone the repository, and choose one PyTorch installation.

CPU:

```bash
python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements.txt
python -m pip check
```

CUDA 11.8:

```bash
python -m pip install -r requirements-cuda118.txt
python -m pip install -r requirements.txt
python -m pip check
```

The CUDA wheel includes the PyTorch CUDA runtime but still requires a compatible NVIDIA driver. The code does not hard-code a GPU.

## Download checkpoint

The following command performs an anonymous download and fails unless both the exact byte size and SHA256 match:

```bash
python scripts/download_checkpoint.py --output checkpoints/magicmri_ckpt_release.pth
export MAGICMRI_CKPT="$PWD/checkpoints/magicmri_ckpt_release.pth"
```

- Size: `1,483,011,953` bytes
- SHA256: `7afcc73b8c829b96cb9276d1a7cc234a30d3182f6594f744083556db6f07e65e`

## 60-second smoke test

The interface-only validation needs no checkpoint:

```bash
python -m magicmri.infer \
  --manifest examples/smoke_manifest.jsonl \
  --task-registry configs/core36_tasks.yaml \
  --output-dir outputs/validation \
  --device cpu \
  --validate-only
```

After downloading the checkpoint, run one real CPU inference:

```bash
python scripts/inference.py \
  --example examples/translation/T01 \
  --checkpoint "$MAGICMRI_CKPT" \
  --output-dir outputs/cpu_smoke \
  --device cpu
```

GPU smoke test:

```bash
python scripts/inference.py \
  --example examples/translation/T01 \
  --checkpoint "$MAGICMRI_CKPT" \
  --output-dir outputs/gpu_smoke \
  --device cuda
```

`--device auto` selects CUDA when available and otherwise selects CPU. `--device cpu` always uses CPU. `--device cuda` fails clearly when CUDA is unavailable.

Run all 18 included fixtures with:

```bash
python scripts/inference.py \
  --all-examples \
  --checkpoint "$MAGICMRI_CKPT" \
  --output-dir outputs/all_synthetic \
  --device auto
```

## Reproduce the 36 core tasks

Obtain BraTS data through its authorized source. The repository neither downloads nor redistributes it. Set the local data root once:

```bash
export BRATS_ROOT=/absolute/location/of/BraTS
```

BraTS distributions and local preprocessing layouts are not sufficiently uniform for safe automatic guessing. Copy [the JSONL binding template](manifests/core36_bindings.template.jsonl) to an untracked local file and add one or more rows for every registered task. Paths may be relative to `BRATS_ROOT`. Each row explicitly binds an exemplar source, exemplar target, query source, evaluation target, pseudonymous patient, and ordered slice index. The [JSON schema](manifests/core36_manifest_schema.json) defines every field.

Build a validated local manifest:

```bash
python scripts/build_core36_manifest.py \
  --data-root "$BRATS_ROOT" \
  --bindings manifests/core36_bindings.local.jsonl \
  --task-registry configs/core36_tasks.yaml \
  --output manifests/core36_eval.jsonl
```

The builder fails on missing files, unknown tasks, duplicate sample IDs, paths outside the data root, or any missing Core-36 task. It never substitutes data and never copies source images.

Run Core-36 inference:

```bash
python -m magicmri.infer \
  --manifest manifests/core36_eval.jsonl \
  --checkpoint "$MAGICMRI_CKPT" \
  --task-registry configs/core36_tasks.yaml \
  --output-dir outputs/core36 \
  --device auto
```

Predictions are written under task-specific directories, and `outputs/core36/predictions.jsonl` preserves the exact patient/slice/task bindings needed by evaluation.

## Core-36 evaluation

Run all formal core metrics and aggregation:

```bash
python evaluation/core36_evaluator.py \
  --manifest outputs/core36/predictions.jsonl \
  --task-registry configs/core36_tasks.yaml \
  --output-dir outputs/core36_metrics \
  --device auto
```

Translation and enhancement/restoration report SSIM, PSNR, NMAE, and LPIPS. Segmentation reports Dice, mIoU, pACC, and HD95. Outputs include per-slice records, patient summaries, task summaries, and family-level task macros. The exact aggregation, threshold, empty-mask, metric-direction, and LPIPS rules are in [the evaluation protocol](docs/CORE36_EVALUATION_PROTOCOL.md).

LPIPS may anonymously download pretrained AlexNet weights on first use. The evaluator warns before initialization; weights are normally cached under `torch.hub.get_dir()/checkpoints`. Use `--skip-lpips` only for a software smoke test, not a complete Core-36 metric run.

The legacy single-PNG metric command remains available for diagnostics and is not the paper-level aggregation pipeline:

```bash
python evaluation/metrics.py \
  --prediction outputs/cpu_smoke/T01.png \
  --target examples/translation/T01/exemplar_target.png \
  --task image
```

## Repository structure

- `magicmri/`: model, prompt construction, checkpoint loading, and manifest inference
- `configs/core36_tasks.yaml`: sole Core-36 task registry
- `evaluation/`: core metrics and aggregation
- `manifests/`: public schema and binding template, not patient data
- `examples/`: synthetic smoke-test fixtures
- `scripts/`: checkpoint download, manifest construction, data preparation, and legacy fixture runner
- `tests/`: fast unit tests plus optional checkpoint integration checks

## Expected resources

The ViT-L checkpoint is approximately 1.48 GB. CPU inference is supported and is substantially slower than GPU inference. LPIPS adds an AlexNet-weight download on first use. Full Core-36 run time and storage depend on the number of locally bound slices.

## Synthetic fixture disclaimer

These synthetic fixtures are provided only for software smoke testing and are not samples from the study cohorts and are not used to reproduce paper-level quantitative results.

They exercise interfaces, prompt construction, checkpoint loading, inference-family paths, output shape/non-emptiness, and binary segmentation rendering. They are not representative MRI examples or scientific evidence.

## Citation

Citation metadata will be updated after publication. Until then, cite the versioned MagicMRI repository and the associated manuscript; no DOI is claimed here.

```bibtex
@misc{magicmri,
  title        = {MagicMRI},
  howpublished = {Versioned software repository},
  url          = {https://github.com/convallariass/MagicMRI},
  note         = {Citation metadata will be updated after publication}
}
```

## License and permitted use

MagicMRI is made publicly available to support scientific transparency and
reproduction of the published Core-36 experiments. MagicMRI-authored materials
are provided under the [PolyForm Strict License 1.0.0](LICENSE), except where
otherwise noted.

Permitted use is limited to non-commercial purposes such as academic research,
scientific reproduction, education, evaluation, personal study, and
non-commercial testing. Commercial use is not permitted without separate
written permission from the applicable copyright holder(s). The project license
does not grant permission to redistribute MagicMRI-authored materials or to
distribute modified or derivative versions. The released checkpoint is
provided under the same intended research/non-commercial release scope; see
[the model-use notice](docs/MODEL_USE_NOTICE.md).

Third-party components remain subject exclusively to their respective original
licenses and notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
`licenses/`. This summary is provided for convenience. The canonical
PolyForm Strict License 1.0.0 governs the covered materials.
