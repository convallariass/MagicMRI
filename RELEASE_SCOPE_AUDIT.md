# Release scope truth audit

Audit baseline: public `main` at `48e5219c57a5f9ca49d4c88b3a6271491b86a98f`, inspected before release changes on 2026-08-29.

## Current release-candidate scope

The completed candidate supplies Core-36 task definitions, fail-closed local
BraTS binding, manifest-driven inference, formal core evaluation, synthetic
smoke-test fixtures, tests, and the unchanged public release checkpoint. A
reader can run the intended workflow using environment variables, local data
paths and binding files, repository configs, command-line arguments, and the
released checkpoint. No documented Core-36 step requires editing
MagicMRI-authored Python source.

Project licensing is now the **PolyForm Strict License 1.0.0**. The project
release model is a restrictive non-commercial research/source-available
release, not OSI-approved open source. MagicMRI-authored materials are covered
by the project license except where otherwise noted. Third-party components
remain exclusively under their original licenses and notices. The publicly
downloadable checkpoint is provided under the same intended
research/non-commercial scope; public access does not grant commercial use,
redistribution, sublicensing, or distribution of modified or derived versions.

## A. What the baseline repository can do

- Construct the released ViT-L MagicMRI implementation and strictly load the anonymous release checkpoint.
- Run real 2D visual-prompt inference from one exemplar source-target pair and one query source image.
- Run 18 included deterministic synthetic fixtures: six translation, six enhancement/restoration, and six segmentation fixtures.
- Compute SSIM, PSNR, NMAE, optional LPIPS, Dice, mIoU, and HD95 for one prediction-target PNG pair.
- Validate the included fixture inventory without loading a checkpoint.

## B. What the baseline repository cannot do

- It has no complete machine-readable registry for the 36 core tasks.
- It has no fail-closed workflow for binding user-owned BraTS data to all 36 tasks.
- It has no manifest-driven `python -m magicmri.infer` interface.
- It has no dataset loop, patient binding, task aggregation, pACC implementation, or Core-36 macro summary.
- It cannot substantiate paper-level quantitative reproduction from the included synthetic fixtures.
- It does not reproduce held-out, composite, downstream-classification, reviewer-only, or training-from-scratch analyses.

## C. Claims that exceed the baseline repository

The baseline README calls its evaluator “minimal,” but an unqualified claim that the repository provides “evaluation code” can still be read as a paper evaluation pipeline. At baseline, only single-PNG metrics exist. Any current manuscript or response wording that says “all evaluation code,” calls the 18 current assets verified or representative study examples, or implies held-out/composite/classification reproduction exceeds the repository.

No authoritative current manuscript or response-letter source file was found in the accessible working tree during this audit. Accordingly, no manuscript file was changed. The wording supplied for this closeout is treated as the current alignment target: **core-task inference and evaluation code**.

## D. Core-36 registry at baseline

**Absent.** Task IDs occur in six synthetic examples per family, but the repository does not enumerate all 12 translation, 12 enhancement/restoration, and 12 segmentation tasks in one machine-readable source of truth.

## E. Baseline evaluation gaps

Compared with the retained project evaluator implementations, the baseline evaluator lacks:

- grayscale SSIM preprocessing used by the core generation evaluator;
- reusable LPIPS initialization and an explicit first-use download warning;
- segmentation pACC;
- a manifest/dataset loop and strict prediction-target matching;
- per-slice records with patient and task identifiers;
- patient-level aggregation, task summaries, and Core-36 macro aggregation;
- a documented empty-mask and HD95 rule;
- a documented segmentation threshold and LPIPS input range;
- a parity record against retained authoritative implementations.

The retained implementations also contain project-specific paths, distributed launch assumptions, development logging, and unrelated metrics. Those implementation artifacts are not suitable for direct public release. The release evaluator should preserve the relevant metric definitions while excluding private paths and non-Core-36 analyses.
