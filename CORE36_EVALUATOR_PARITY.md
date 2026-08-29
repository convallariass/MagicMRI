# Core-36 evaluator parity

Parity was run on 2026-08-29 using three retained non-public validation cases: one translation case (`T01`), one enhancement/restoration case (`E01`), and one segmentation case (`S01`). The same stored predictions and targets were evaluated by independent functions extracted directly from the retained authoritative project evaluators and by the public release implementation. No validation image is redistributed by this repository.

Targets were resized once with the authoritative evaluator's OpenCV linear interpolation before both arms so that both received identical arrays. LPIPS used the same cached AlexNet weights and CPU device in both arms. The single-slice task aggregate is identical to each corresponding per-slice result; a separate unit test verifies the retained slice-mean then task-macro aggregation order on multiple records.

| Case | Metric | Original result | Public result | Absolute difference | Tolerance | Status |
|---|---:|---:|---:|---:|---:|---:|
| T01 | SSIM | 0.9920863576536257 | 0.9920863576536257 | 0 | 1e-12 | PASS |
| T01 | PSNR | 42.09992852988187 | 42.09992852988187 | 0 | 1e-12 | PASS |
| T01 | NMAE | 0.0020048188546251833 | 0.0020048188546251833 | 0 | 1e-12 | PASS |
| T01 | LPIPS | 0.010379847139120102 | 0.010379847139120102 | 0 | 1e-8 | PASS |
| E01 | SSIM | 0.9973187428773119 | 0.9973187428773119 | 0 | 1e-12 | PASS |
| E01 | PSNR | 45.087382628886076 | 45.087382628886076 | 0 | 1e-12 | PASS |
| E01 | NMAE | 0.0010189427333433373 | 0.0010189427333433373 | 0 | 1e-12 | PASS |
| E01 | LPIPS | 0.007133874110877514 | 0.007133874110877514 | 0 | 1e-8 | PASS |
| S01 | Dice | 0.9699626562753694 | 0.9699626562753694 | 0 | 1e-12 | PASS |
| S01 | mIoU | 0.9688783850522757 | 0.9688783850522797 | 4.0e-15 | 1e-12 | PASS |
| S01 | pACC | 0.9963129783163265 | 0.9963129783163265 | 0 | 1e-12 | PASS |
| S01 | HD95 | 5.656854249492381 | 5.656854249492381 | 0 | 1e-12 | PASS |

**Metric parity: PASS. Aggregation parity: PASS.**

Authoritative provenance: the retained generation implementation `denoise_transition_infer_mri_1226.py`, segmentation implementation `segmentation_eval_36k1230plus_dice_hd95.py`, and its later fail-closed parameterized evaluator. Only Core-36 metric and aggregation behavior was restored; private paths, distributed-launch assumptions, unrelated analyses, and non-core metrics were not copied.
