# Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/zoedepth-metric-depth-pipeline/blob/main/tutorials/zoedepth_metric_depth_colab.ipynb)

Notebook specification: **DIMER Notebook Specification 2.0**. The notebook is a generated, standalone carrier: edit `tools/notebook_template.py` or the package and regenerate with `python tools/build_notebook.py`; do not hand-edit the `.ipynb`.

| Notebook | Profile | Mode | Carrier | Default E2E path | Runtime | BYOD | Release status |
|---|---|---|---|---|---|---|---|
| `zoedepth_metric_depth_colab.ipynb` | `E2E` | `GUIDED` | standalone (generated) | 24 deterministic RGB/metric-depth scenes; 18/6 train/held-out split; pinned pretrained and constant-median baselines; two real gradient epochs on the metric head; held-out `abs_rel` and `delta1`; unseen-scene prediction; SafeTensors adapter export; fresh pinned-base reload equivalence | CUDA required; Colab T4 or Kaggle T4 supported | ZIP containing paired `images/` and `depth/*.npy` files, off by default | **Candidate** — verified 11/11-cell clean Kaggle T4 execution for the exact committed blob is recorded in `../docs/release-verification.md`; awaiting reviewer/integrator promotion |

## Conformance notes

- The carrier embeds the exact pipeline module, immutable model identity, snapshot manifest, and runtime pins. Generator parity is enforced by `tests/test_notebook_parity.py` and `tools/validate_release_assets.py`.
- `validate_depth_dataset` rejects malformed, duplicate, mismatched, non-finite, non-positive, out-of-range, or oversized records before training. The default sample and split are deterministic.
- `freeze_for_adaptation` freezes the pinned base except `metric_head.*`; `finetune` records nonzero gradient-driven weight movement and validation history.
- `evaluate_adaptation` reports held-out `abs_rel` and `delta1` beside a constant training-median baseline. The synthetic result is workflow evidence, not an NYU/KITTI benchmark.
- `save_artifact` writes only `adapter.safetensors` plus a closed manifest containing the pinned base identity, trainable prefixes, digest, and adaptation metadata. `from_artifact` constructs a fresh pinned base, verifies the artifact, and attaches it without pickle or remote code.
- The unseen-scene depth map must match after fresh reload. Static checks and local runs are not promotion evidence; see `../docs/release-verification.md`.

## AI Assistance Disclosure

This repository’s code and documentation were developed with generative AI assistance under maintainer direction. The maintainer remains responsible for review, validation, and release decisions.
