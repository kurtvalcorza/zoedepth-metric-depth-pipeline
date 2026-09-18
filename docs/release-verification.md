# Release verification

`tutorials/zoedepth_metric_depth_colab.ipynb` is an `E2E`, standalone Candidate carrier under DIMER Notebook Specification 2.0. Review remediation changed the executable source after the v4 run, so a fresh exact-blob clean Kaggle T4 run is required. The v4 executor pass remains recorded below as historical workflow evidence but does not satisfy the gate for the corrected revision.

## Automatic coverage

CI and the local validator must establish that:

- notebook JSON and every code cell are valid, generated content is byte-for-byte current, and no outputs or execution counts are committed;
- the embedded pipeline, immutable model identity, snapshot manifest, and dependency pins match repository source;
- the default path uses a deterministic 24-record RGB/metric-depth dataset and disjoint 18/6 train/held-out split;
- the pinned pretrained model and constant training-median baseline run before real gradient adaptation;
- only `metric_head.*` is trainable, its weights move, and held-out `abs_rel` plus `delta1` are recorded beside the baseline;
- an unseen-scene prediction is exported, the adapter uses SafeTensors with a closed digest-bound manifest, and a fresh pinned base reload reproduces the depth map;
- the status remains Candidate until the recorded exact-blob evidence is reviewed and an integrator promotes it.

## Supported execution procedure

1. Commit the generated notebook and resolve its Git blob ID.
2. Start a new Colab T4 or Kaggle T4 runtime with no repository checkout or warm model cache.
3. Run all cells at their defaults with `USE_BYOD = False`; do not edit implementation cells.
4. Confirm the runtime reports CUDA, the immutable model revision, and the same repository revision recorded in notebook metadata.
5. Confirm dataset validation and the 18/6 split, pretrained evaluation, two training epochs, nonzero `weight_delta_l2`, held-out evaluation, unseen inference, artifact digest verification, and fresh-base reload all complete.
6. Confirm the output directory contains the dataset manifest, evaluation report, result JSON, and exactly the two-file adapter directory.
7. Record runtime versions, device, notebook commit/blob, wall time, metrics, output inventory, warnings, and clean-cache status below. Record no secrets.

A failed default path, missing gradient update, altered split, unsafe artifact, or reload mismatch blocks promotion.

## Recorded executions

### Current E2E carrier

| Date (UTC) | Commit / notebook blob | Executor | Path | Outcome |
|---|---|---|---|---|
| Awaiting rerun | Corrected review-remediation commit/blob to be recorded | Kaggle T4 | Default generated dataset | Required before promotion |

### Local E2E pre-flight (not promotion evidence)

| Date | Source identity | Executor | Outcome |
|---|---|---|---|
| 2026-09-18 | Uncommitted generated carrier; metadata base revision `5fb52be92696` | Windows RTX 5070 Ti Laptop GPU, Python 3.14.2, torch 2.11.0+cu128, transformers 5.8.1; pinned install skipped | **PASS** — 11/11 code cells; 18 train + 6 held-out; 36 optimizer steps; 1,761,754 trainable / 343,311,553 frozen; weight delta 0.094411; held-out `abs_rel` 0.431046 → 0.454265 and `delta1` 0.211141 → 0.211249 versus constant-median 0.452605 / 0.257758; unseen `abs_rel` 0.427272; SafeTensors artifact and fresh-base reload max difference 0.0. This proves workflow execution, not quality improvement. |

### Historical superseded carriers

| Date (UTC) | Commit / notebook blob | Executor | Profile | Outcome |
|---|---|---|---|---|
| 2026-09-14 | `4cd28d1` / `3bcf19a2b16e` | Kaggle CPU | Earlier inference-only carrier | Passed its prior 8-cell path; it is not evidence for the current E2E notebook |
| 2026-09-17 | `7bb7db6f401e7ffc9791d879949d56cc5fa9abaf` / `c5d97244af45b6522695ba5700a0bb1182edeebb` | Kaggle T4 (`kurtvalcorza/dimer-nb2-zoedepth-metric-depth` v2) | Earlier E2E carrier | Passed 11/11 cells, 36 optimizer steps, and reload equivalence, but the target depth bypassed the processor padding geometry; retained as workflow history only |
| 2026-09-18 | `b1f91a13621959e7f41f72b4ab69cfda7ba62b28` / `810883ed15211487e09faa6a6b6d5acbd811e339` | Kaggle T4 (`kurtvalcorza/dimer-nb2-zoedepth-metric-depth` v4) | Superseded E2E carrier | Passed 11/11 cells and reload, but subsequent review found nine data-boundary, metric-aggregation, artifact, and provenance defects; retained as workflow history only |

## Current status

The corrected E2E implementation and generated carrier remain **Candidate**, and the clean-runtime gate is open. Requalification must verify the exact corrected notebook blob, a clean Hugging Face cache, the declared pins, the digest-bound base snapshot, all 11 code cells, bounded input memory, finite gradient movement, pixel-weighted held-out evaluation, and fresh reload equivalence. Synthetic held-out scores remain workflow evidence only and cannot establish quality improvement.
