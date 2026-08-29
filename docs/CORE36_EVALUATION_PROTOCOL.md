# Core-36 evaluation protocol

## Scope and analysis unit

This evaluator covers only the 36 registered core tasks. MagicMRI inference is 2D and the analysis unit is one axial slice. Each manifest row binds a prediction and target to a pseudonymous patient, slice, and task. Held-out project subsets, composite tasks, downstream classification, and training analyses are outside this release.

## Metrics

Translation and enhancement/restoration use SSIM (higher is better), PSNR in dB (higher), NMAE (lower), and LPIPS-AlexNet (lower). SSIM is computed on 8-bit grayscale images with `data_range=255`. PSNR preserves the retained paper evaluator's unsigned 8-bit subtraction and squaring before the mean, including its exact-match value of 100 dB. NMAE casts to floating point before subtraction and divides mean absolute error by 255. These details are explicit because changing them would change parity with reported evaluation.

LPIPS uses the retained core evaluator preprocessing: PNGs were read through OpenCV, so channels are presented to LPIPS in BGR order and converted from unsigned 8-bit values to float values in `[0, 1]`. A single `lpips.LPIPS(net="alex")` network is reused for the dataset. The first invocation may download pretrained AlexNet weights to the PyTorch hub cache (`torch.hub.get_dir()/checkpoints`). The evaluator prints a warning before initialization.

Segmentation uses Dice (higher), two-class mean IoU (higher), pixel accuracy/pACC (higher), and HD95 in pixels (lower). Both prediction and target are converted to grayscale and thresholded with the foreground rule `value > 127`. Inference writes a binary PNG using `value > 128`, matching the release inference configuration.

## Empty masks and HD95

- Dice is 1 when both masks are empty and 0 when exactly one mask is empty.
- For mIoU, an absent class has IoU 0; foreground and background IoU are averaged.
- pACC is the fraction of exactly matching binary pixels.
- HD95 is 0 when both masks are empty, `NaN` when exactly one mask is empty, and the symmetric 95th-percentile surface distance otherwise.
- An undefined HD95 is serialized as JSON `null` in per-slice output and excluded from arithmetic means. The accompanying `HD95_n_finite` fields make this attrition explicit.

## Aggregation order

Metrics are computed per slice. Patient summaries preserve `(task, patient)` binding and average finite slice values for patient-level inspection. To preserve the paper evaluator's aggregation, each task summary is the arithmetic mean of its finite slice values; `n_patients` and per-metric finite-slice counts are reported alongside it. Finally, task means are macro-averaged within each family. Generation and segmentation metric families are never combined into a single cross-family number.

The evaluator fails on unknown tasks, missing files, shape mismatches, duplicate sample IDs, or an incomplete 36-task set unless `--allow-task-subset` is explicitly supplied for a smoke/parity run.

## Optional ordered stacking

No native 3D MagicMRI inference is implemented or claimed. If users perform ordered axial stacking for a separate volumetric analysis, that reconstruction occurs outside the model and is outside the released Core-36 slice-level evaluator.

## Provenance

The public definitions were extracted from the retained project generation evaluator (`denoise_transition_infer_mri_1226.py`) and segmentation evaluators (`segmentation_eval_36k1230plus_dice_hd95.py` and the later parameterized segmentation evaluator). Project-specific paths, distributed-launch code, development logging, and non-Core-36 metrics were intentionally excluded. Numerical parity is recorded in `CORE36_EVALUATOR_PARITY.md`.
