# Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/zoedepth-metric-depth-pipeline/blob/main/tutorials/zoedepth_metric_depth_colab.ipynb)

Notebook specification: **DIMER Notebook Specification 2.5** (isolated `uv` environment, no runtime restart, since the 2026-10-05 review fixes). The notebook is a generated, standalone carrier: edit `tools/notebook_template.py` or the package and regenerate with `python tools/build_notebook.py`; do not hand-edit the `.ipynb`.

| Notebook | Profile | Mode | Carrier | Default E2E path | Runtime | BYOD | Release status |
|---|---|---|---|---|---|---|---|
| `zoedepth_metric_depth_colab.ipynb` | `E2E` | `GUIDED` | standalone (generated) | 24 deterministic RGB/metric-depth scenes; 18/6 train/held-out split; pinned pretrained and constant-median baselines; two real gradient epochs on the metric head; held-out `abs_rel` and `delta1`; unseen-scene prediction; SafeTensors adapter export; fresh pinned-base reload equivalence | CUDA required (Section 1 stops on a CPU runtime); Colab T4 or Kaggle T4 supported; recorded hosted runs: Kaggle T4 of the previous blob (one restart) and a one-pass Colab T4 run of blob `441111a9df11` (2026-10-10, no restart); the current blob `444f0542` (DAT20 refusal display) passed its default path and a scripted DAT20 refusal on a hosted Colab T4 on 2026-10-29 | ZIP of 8–128 paired `images/` and `depth/*.npy` files (at the root or inside one folder; finite positive metres), by `BYOD_ZIP_PATH` or one upload; off by default; no BYOD run recorded | **Candidate** — the recorded 11/11-cell Kaggle T4 run (previous blob) needed a manual restart, so it is not one-pass evidence; see `../docs/release-verification.md`; blob `441111a9df11` (commit `13c283a`) passed a one-pass Colab T4 run with no restart on 2026-10-10; the current blob `444f0542` (DAT20 refusal display) passed its default path and a scripted DAT20 refusal on a hosted Colab T4 on 2026-10-29; remaining gates REL12 and REL2; earlier blobs are history (REL14) |

## Conformance notes

- The carrier embeds the exact pipeline module, immutable model identity, snapshot manifest, and runtime pins. Generator parity is enforced by `tests/test_notebook_parity.py` and `tools/validate_release_assets.py`.
- `validate_depth_dataset` rejects malformed, duplicate, mismatched, non-finite, non-positive, out-of-range, or oversized records before training. The default sample and split are deterministic.
- `freeze_for_adaptation` freezes the pinned base except `metric_head.*`; `finetune` records nonzero gradient-driven weight movement and validation history.
- `evaluate_adaptation` reports held-out `abs_rel` and `delta1` beside a constant training-median baseline. The synthetic result is workflow evidence, not an NYU/KITTI benchmark.
- `save_artifact` writes only `adapter.safetensors` plus a closed manifest containing the pinned base identity, trainable prefixes, digest, and adaptation metadata. `from_artifact` constructs a fresh pinned base, verifies the artifact, and attaches it without pickle or remote code.
- The unseen-scene depth map must match after fresh reload. Static checks and local runs are not promotion evidence; see `../docs/release-verification.md`.

## AI Assistance Disclosure

This repository’s code and documentation were developed with generative AI assistance under maintainer direction. The maintainer remains responsible for review, validation, and release decisions.
