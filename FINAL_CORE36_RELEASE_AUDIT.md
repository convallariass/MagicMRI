# Final Core-36 release audit

Audit date: 2026-08-30

**PUBLIC RELEASE VERDICT: READY**

This verdict applies to the author-approved Core-36 release contents inventoried
and tested immediately before the release commit.

## Source state

- Working branch: `release/core36-reproducibility`
- Base HEAD: `48e5219c57a5f9ca49d4c88b3a6271491b86a98f`
- Public fresh clone before applying the uncommitted release candidate: branch `main`, same HEAD, clean status.
- No model retraining, checkpoint change, paper-result change, force-push, or history rewrite was performed.

Because the user requested review before commit, final-candidate QA was performed by mechanically applying the uncommitted worktree contents over a second clean public clone. Its source changes are intentionally visible in `git status`; checkpoint and test outputs remain ignored.

## Acceptance results

| Gate | Result | Evidence |
|---|---|---|
| Python release syntax/import | PASS | `compileall` succeeded with external `PYTHONPATH` removed. |
| CPU clean install | PASS | Fresh Python 3.8.18 environment installed `torch==2.0.0+cpu`, `torchvision==0.15.1+cpu`, then pinned release dependencies. |
| `pip check` | PASS | `No broken requirements found.` |
| CUDA 11.8 install definition | PASS | Exact `2.0.0+cu118` / `0.15.1+cu118` wheel path is isolated in `requirements-cuda118.txt`; no CUDA GPU was available for a positive runtime test. |
| Device rules | PASS | CPU and auto-to-CPU paths ran; explicit CUDA on the CPU-only environment failed clearly with `torch.cuda.is_available() is false`. |
| Anonymous checkpoint access/download | PASS | The public Google Drive file completed an anonymous full-file download in the immediately preceding clean public-repository audit at the same base HEAD. Size and SHA matched. The final clone also reached the real anonymous confirmation/download stream. A second full transfer could not finish inside the audit harness's repeated 30-second process termination, so the already anonymously obtained byte-identical artifact was used for final-clone verification rather than misreported as a second completed transfer. |
| Checkpoint size/SHA gate | PASS | `1,483,011,953` bytes; `7afcc73b8c829b96cb9276d1a7cc234a30d3182f6594f744083556db6f07e65e`. The release downloader accepted the exact file and rejects mismatch. |
| Strict checkpoint load | PASS | Optional integration test ran with `MAGICMRI_CKPT` and strict state-dict loading passed with no missing/unexpected keys. |
| Core-36 registry | PASS | Exactly 36 unique tasks: 12 translation, 12 enhancement/restoration, 12 segmentation; exact modality/degradation/tumor Cartesian sets tested. |
| Core-36 data builder | PASS | A 36-task local binding set produced 36 validated records. Missing files/tasks, escapes, duplicate samples, and unknown tasks fail closed. |
| README interface validation | PASS | Local links resolved; module CLI validation accepted all 18 included records without loading a checkpoint. |
| README CPU smoke | PASS | The documented T01 command loaded the release checkpoint and wrote a non-empty 448×448 output. |
| 18 synthetic fixtures | PASS | Real CPU model inference completed 18/18: six translation, six enhancement/restoration, six segmentation. All outputs were non-empty 448×448 images; all six segmentation outputs contained only 0 and 255. |
| Core-36 evaluator smoke | PASS | Complete 36-task loop wrote 36 per-slice rows, 36 patient summaries, 36 task summaries, and three family summaries. |
| Metric parity | PASS | T01/E01/S01 comparison covered SSIM, PSNR, NMAE, LPIPS, Dice, mIoU, pACC, and HD95. Maximum absolute difference was `4.0e-15`; see `CORE36_EVALUATOR_PARITY.md`. |
| Aggregation parity | PASS | Retained per-slice task mean and task-level family macro order is implemented and unit tested. |
| `python -m unittest discover -v` | PASS | 12 tests ran and passed with the checkpoint integration variable set; zero skips in final acceptance. |
| Personal/internal path scan | PASS | No username, personal email, server IP, cluster/home-directory prefix, drive-letter path, account, or credential was found in release text/code. |
| Development-marker/internal-note scan | PASS | Matches were limited to deliberate scope/provenance statements and implementation terms such as threshold; no development reminder or mechanical-generation note remains. |
| Tracked-file hygiene | PASS | No tracked cache, editor state, log, output, temporary artifact, patient data, intermediate result, or checkpoint was added. Generated caches/outputs were ignored. |
| Required third-party attribution | PASS | Retained source notices and third-party license texts are byte-unchanged. `THIRD_PARTY_NOTICES.md` adds only an explicit project-license exclusion and continues to identify the BAAI/MIT and Meta/Apache components. |
| Top-level MagicMRI license | PASS | Root `LICENSE` is the verbatim canonical PolyForm Strict License 1.0.0: 3,593 bytes, SHA256 `9eb48619fbc193ab7bb327b090cfcc703000265b83e670f81f231d0b1c43c56e`, identical to the PolyForm Project official `1.0.0` tag. No copyright-holder identity was invented; the canonical text requires no filled copyright line. |
| Project release model | PASS | MagicMRI-authored materials are a restrictive non-commercial research/source-available release. The project is not described as OSI-approved open source. Commercial use, redistribution, sublicensing, and distribution of changes/new works are not granted. |
| Checkpoint notice | PASS | The publicly downloadable checkpoint is explicitly provided as a covered MagicMRI-authored release material under the same PolyForm Strict terms for the intended research/reproducibility scope. Public access is not represented as unrestricted reuse. |
| Third-party license separation | PASS | BAAI/MIT and Meta/Apache components remain exclusively under their original licenses/notices. Their source notices and license texts are byte-unchanged; the project license does not claim to override them. |
| No-source-edit reproduction | PASS | Clone, install, checkpoint download, local `BRATS_ROOT`/`MAGICMRI_CKPT`, binding/manifest construction, inference, and evaluation use existing configs and CLI parameters. No documented Core-36 step requires editing MagicMRI-authored Python source. |
| Authorship notice | PASS | The single approved comment `# MagicMRI implementation developed by Qing Zhao.` appears only in `magicmri/__init__.py`; it is not a copyright statement and no third-party-derived source file was changed for authorship. |

## License-closeout targeted QA

The license closeout changed only `LICENSE`, release-facing Markdown, and
licensing notices. The later approved authorship change added one comment to
`magicmri/__init__.py`. All executable/scientific content otherwise remained
byte-identical to the candidate that passed 18/18 CPU inference and evaluator
parity, so those expensive scientific checks are inherited.

In a fresh public clone with the candidate overlaid, a clean Python 3.8.18 CPU
environment again passed both requirements layers and `pip check`.
`python -m unittest discover -v` passed 12/12 with the actual release
checkpoint, including strict state-dict loading. Manifest validation accepted
18 fixtures against all 36 tasks. The actual checkpoint file remained
`1,483,011,953` bytes with SHA256
`7afcc73b8c829b96cb9276d1a7cc234a30d3182f6594f744083556db6f07e65e`.
All documented release commands exposed the expected CLI arguments. The final
authorship-comment candidate reran the same 12 tests successfully.

License, stale-license, private/internal-path, development-marker, tracked
artifact, and third-party-attribution scans passed. Runtime `__pycache__`
files created in the disposable audit clone were ignored and are not release
files.

## Remaining issues

### P0

None.

### P1

None.

### P2

None.

**Remaining blockers: none.**

## Exact modified/new files

Modified:

- `.gitignore`
- `README.md`
- `THIRD_PARTY_NOTICES.md`
- `configs/inference.yaml`
- `evaluation/metrics.py`
- `examples/README.md`
- `magicmri/__init__.py`
- `requirements.txt`
- `scripts/inference.py`

New:

- `CORE36_EVALUATOR_PARITY.md`
- `FINAL_CORE36_RELEASE_AUDIT.md`
- `LICENSE`
- `PUBLIC_RELEASE_WORDING_RECOMMENDATIONS.md`
- `RELEASE_BLOCKERS.md`
- `RELEASE_SCOPE_AUDIT.md`
- `configs/core36_tasks.yaml`
- `docs/CORE36_EVALUATION_PROTOCOL.md`
- `docs/MODEL_USE_NOTICE.md`
- `evaluation/aggregation.py`
- `evaluation/core36_evaluator.py`
- `examples/smoke_manifest.jsonl`
- `magicmri/infer.py`
- `magicmri/manifest.py`
- `magicmri/tasks.py`
- `manifests/core36_bindings.template.jsonl`
- `manifests/core36_manifest_schema.json`
- `requirements-cpu.txt`
- `requirements-cuda118.txt`
- `scripts/build_core36_manifest.py`
- `scripts/download_checkpoint.py`
- `tests/__init__.py`
- `tests/test_core36.py`
- `tests/test_metrics_and_interface.py`

## Final `git diff --stat`

The literal unstaged `git diff --stat` reports tracked-file edits only:

```text
 .gitignore             |  10 +++
 README.md              | 196 +++++++++++++++++++++++++++++++++++++++----------
 THIRD_PARTY_NOTICES.md |   6 ++
 configs/inference.yaml |   2 +-
 evaluation/metrics.py  | 125 ++++++++++++++++++++++---------
 examples/README.md     |   8 +-
 magicmri/__init__.py   |   2 +
 requirements.txt       |  17 ++---
 scripts/inference.py   |   7 +-
 9 files changed, 283 insertions(+), 90 deletions(-)
```

There are also 24 intentional new release files listed above. The final release
commit stages all 9 tracked modifications and all 24 new files explicitly.

## Pre-commit inventory

```text
intentional tracked modifications: 9
intentional new release files: 24
tracked checkpoint/output/cache/temp/private-data files: 0
```

## Public release verdict

**READY**

All Core-36 reproducibility gates pass. The canonical PolyForm Strict License
1.0.0, checkpoint/model-use notice, and third-party exclusions implement the
approved restrictive non-commercial research/source-available release model.
No technical or licensing blocker is currently identified. Git publication is
authorized under the normal, non-force release workflow.
