# Public release wording recommendations

## Manuscript Code Availability

Recommended wording after the Core-36 release is versioned:

> The main MagicMRI source code, core-task inference and evaluation code, inference configurations, Core-36 task definitions, release checkpoint, and runnable synthetic smoke-test examples are publicly available through the versioned project repository at https://github.com/convallariass/MagicMRI.

If appropriate, add that access to the underlying BraTS datasets remains governed by their data custodians and is not redistributed by the repository.

Optional licensing sentence:

> MagicMRI-authored release materials and the released checkpoint are source-available under the PolyForm Strict License 1.0.0 for non-commercial purposes; third-party components remain under their original licenses and notices.

## Response letter

Use the same bounded scope. State that the repository supplies 18 synthetic software smoke-test fixtures (six per family) and a machine-readable registry covering all 36 core tasks. Do not call the fixtures representative MRI examples, verified study examples, paper examples, modality-faithful examples, or clinically meaningful examples.

Do not claim that “all evaluation code” is public. The repository intentionally excludes held-out project subsets, composite evaluation, downstream classification, reviewer-only analyses, response figures, private patient manifests, raw BraTS data, and training-from-scratch reproduction.
