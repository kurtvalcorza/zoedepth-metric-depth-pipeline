# Release verification

`tutorials/zoedepth_metric_depth_colab.ipynb` is an `E2E`, standalone Candidate carrier under DIMER Notebook Specification 2.0. Exact carrier commit `ac175a534c57a9d9ddd7dc4bad282bc99f56051c`, embedded source revision `f0cedd2892eebaf99e6b598aa60cecf669ba50f4`, and notebook blob `3d3254ea89e196d9f0bd484da8a59275ac5589e4` passed 11/11 cells on Kaggle T4 only after a manual restart following the old in-kernel install cell. That run is workflow history, not one-pass clean-runtime (REL2) evidence and not promotion evidence; the regenerated notebook (2026-10-05 review fixes, below) — blob `441111a9df11` (commit `13c283a`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-10 (Colab CLI 0.7.4 sequential execution, 13/13 code cells, 144.7 s; held-out AbsRel 0.431096 → 0.466203 and δ1 0.211222 → 0.212009, the same as the Kaggle v9 run; reload max difference 0.0) — is the current one-pass record; status stays Candidate.

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
| 2026-09-18 | `ac175a534c57a9d9ddd7dc4bad282bc99f56051c` / `3d3254ea89e196d9f0bd484da8a59275ac5589e4` | Kaggle T4 (`kurtvalcorza/dimer-nb2-zoedepth-metric-depth` v9) | Default generated dataset | **PASS** — 11/11 cells after one expected install restart; 265.5 s; clean cache; 10 staged files / 1,380 MB; 36 optimizer steps; pixel-weighted held-out AbsRel 0.431096 → 0.466203 and delta1 0.211222 → 0.212009; reload max difference 0.0; workflow passed without a synthetic quality-improvement claim |
| 2026-10-10 (09:43:18 UTC start) | `13c283a2397b0fd70d77c2003e6709e492c71848` / `441111a9df11057f25ec70de236f7d1e36ec4090` (`NOTEBOOK_SOURCE.repository_revision` `fe41b1d2cb19`, embedded `module_sha256` `1107ff98322c…`, generator `build_notebook.py/2.1-swc-zoe`, `notebook_spec` 2.2) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-zoedepth-13c283a-4905`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, isolated CPython 3.12.12 (47 locked packages, setup 56 s), torch 2.14.0+cu130, transformers 4.57.6, safetensors 0.8.0, `cuda:0` | Default generated dataset, every form field at its default (`USE_BYOD = False`) | **PASS** — one pass, no restart, 0 errors; 13/13 code cells in order (cell 4, the carried pipeline module, prints nothing); 144.7 s wall; 4 snapshot files / 1,380,379,860 bytes digest-verified at `f364d4c7936e`; 18/6 split, no overlap; pretrained AbsRel 0.431096 / δ1 0.211222 vs constant-median 0.452605 / 0.257758; 1,761,754 trainable / 343,311,553 frozen; val log-L1 0.428735 → 0.425855; 36 optimizer steps; weight delta 0.098800; adapted AbsRel 0.466203 / δ1 0.212009; row correlation −0.841 / −0.842; unseen AbsRel 0.433712; reload `PASSED`, max difference 0.0. Held-out metrics and history equal the Kaggle v9 run. Evidence in `docs/execution-evidence/2026-10-10-13c283a/`: executed notebook SHA-256 `dc47d8675d95cc666cc3a19302d55388f95b0b749e59bb9107237148bc8391ba`, `run_summary.json` `583f5b990090cf7f16d567f25cb270dea95132d8c4543e88d61a58f4a51791ce`, `exec.log` `bdd5a9f61cac8a3886869c6e12a4ede4aad4c59181100b2f82fe64f4ce8b2a41`. Not exercised: BYOD, the Section 12 activity, a browser Run all, form re-runs |
| 2026-10-25 (07:51:08 UTC start) | `63ee8cc` (PR #8 head) / `b0aa8183a8d9` (tier C review fixes) | Colab CLI sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4 (session `suite-zoedepth-63ee8cc-4905`) | Default generated dataset | **PASSED** — 136.1 s wall; 13/13 code cells in order in `exec.log`, one pass, no restart, 0 error outputs, input notebook unchanged; 18 train / 6 held-out records; 1,761,754 trainable of 345 M parameters; 36 optimizer steps; validation log-L1 0.428735 after epoch 1 → 0.425855 after epoch 2; pixel-weighted held-out AbsRel 0.431096 → 0.466203 (constant-median baseline 0.452605) and δ1 0.211222 → 0.212009 (baseline 0.257758); the notebook printed "Held-out AbsRel got worse after training"; unseen scene depth range 1.976–2.362 m; reload `PASSED`, max abs difference 0.0. Every metric equals the 2026-10-10 run and the numbers the notebook quotes. Evidence: `docs/execution-evidence/2026-10-25-63ee8cc/` — SHA-256 captured → committed: `zoedepth_metric_depth_colab_63ee8cc_colab-cli-t4_default.ipynb` `f470ab49dc1e97c8bb5b24b0da62349fa9d8666f9e288050593697423e7f154c` → unchanged (no local path); `exec.log` `4b357f61d3825aff38493a306e45a3eacc7a20c24132f5fe1e3125e5fcffe8f6` → `cfdb37e0d64ad82548d94598fe2a5e54e0f3342b2f751096eb507510f5994ecd` (2 local-path replacements); `run_summary.json` `544c0c9d74c532a0067fb7653472f41ab4b0eb55f2e6b230f7fe2ab39381aaaa` → unchanged (no local path). |

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
| 2026-09-18 | `1cfd4977d330cc04de628d6bc327e577d1228ccc` / `0e4c86304b1102dda3d30900eb32e94e585b0768` | Kaggle T4 (`kurtvalcorza/dimer-nb2-zoedepth-metric-depth` v5) | Superseded E2E carrier | Passed 11/11 cells and reload, but final review found BYOD archive, adapter-size, and reload-to-refine defects; retained as workflow history only |
| 2026-09-18 | `2dc73837422b0751fea79414f3de71f61ac800ec` / `ee943c717635036ae64974e38ae15c71aecd8fb0` | Kaggle T4 (`kurtvalcorza/dimer-nb2-zoedepth-metric-depth` v6) | Provenance-rejected E2E carrier | Passed 11/11 cells and reload, but the notebook still labelled its embedded source as `f11c579`; retained as workflow history only |
| 2026-09-18 | `cd2abb1e780e7707210b33dc2d5369157a4060db` / `6afc6e78ca16fa49803e8cc37e83a26e19759dce` | Kaggle T4 (`kurtvalcorza/dimer-nb2-zoedepth-metric-depth` v7) | Superseded E2E carrier | Passed 11/11 cells and reload, but final review found manifest-size, evidence-index, variability-guidance, and BYOD-report provenance gaps; retained as workflow history only |
| 2026-09-18 | `8db7fe8d8f955d370291a1aa568a92de167e1b6c` / `e9f6ab0fe13622dd9ea3ad58a110d36ede7a2d6c` | Kaggle T4 (`kurtvalcorza/dimer-nb2-zoedepth-metric-depth` v8) | Superseded E2E carrier | Passed 11/11 cells and reload, but final review found adaptation-wide and BYOD cumulative resource ceilings plus stale model-card evidence; retained as workflow history only |

### Hosted Colab T4 runs of blob `444f0542` (2026-10-29)

These runs executed the PR head `e51670b61c40c512e4bc06f9c56d594e6584ae9f`, notebook `tutorials/zoedepth_metric_depth_colab.ipynb` blob `444f0542b88a55c7d7c96ce7ae2d423e7f3e86e1`, fetched at the 40-character SHA and Git-blob verified, with the Colab CLI (`colab exec -f`) on a fresh Colab Tesla T4 session (not a browser `Run all`). Each `run_summary.json` records `verdict` `PASS`, `exec_exit_code` 0, `input_unchanged` true and `stop_verified` true (the VM was released), and the SHA-256 of each executed notebook equals the summary's `executed_sha256`.

**Default path (2026-10-29 14:52:10 UTC start).** Default settings, no overrides; session `suite-zoedepth-e51670b-4905`; wall 149.3 s (`wall_s`); `code_cells` 13, `executed_cells` 12, 0 errors. Retained evidence: [`docs/execution-evidence/2026-10-29-e51670b/`](execution-evidence/2026-10-29-e51670b/).

Printed by the default run (verbatim lines from the executed outputs):

```text
{'verified_files': 4, 'revision': 'f364d4c7936e91f465abba182208dd68142bf0ca'}
  "baseline_pretrained": {
    "records": 6,
    "valid_pixels": 73728,
    "abs_rel": 0.4310957682873357,
    "delta1": 0.21122233072916666,
    "constant_median_baseline_abs_rel": 0.45260539228211055,
    "constant_median_baseline_delta1": 0.2577582465277778
  },
  "verdict": "sample-sanity",
  "adapted": {
    "records": 6,
    "valid_pixels": 73728,
    "abs_rel": 0.46620341110134805,
    "delta1": 0.2120090060763889,
    "constant_median_baseline_abs_rel": 0.45260539228211055,
    "constant_median_baseline_delta1": 0.2577582465277778,
    "abs_rel_delta_vs_baseline": 0.0135980188192375,
    "delta1_delta_vs_baseline": -0.045749240451388895
  },
  "interpretation": {
    "pretrained_beats_constant_abs_rel": true,
    "pretrained_beats_constant_delta1": false,
    "adapted_beats_constant_abs_rel": false,
    "abs_rel_change_after_training": 0.0351,
    "delta1_change_after_training": 0.0008,
    "row_correlation_pretrained": -0.841,
    "row_correlation_adapted": -0.842
  }
Held-out AbsRel got worse after training: the frozen features still read the layout upside down, and the head can only rescale.
{'verification': 'PASSED', 'rtol': 1e-05, 'atol': 1e-05, 'max_abs_diff': 0.0, 'artifact_dir': 'outputs/zoedepth-metric-head-adapter-v1'}
```

**DAT20 refusal record (2026-10-29, blob `444f0542`).** Overrides, set by `--set` on a copy of the blob: `USE_BYOD` `False` → `True` and `BYOD_ZIP_PATH` `''` → `'/etc/hostname'` (the run copy differs from the blob in exactly those 2 source lines). The suite cut the run copy right after the refusing cell `zoedepth_metric_depth-16`, dropping 15 cells (7 of them code cells), which reproduces `Run all` stopping at the first error; the cut copy is `run_copy.ipynb`, SHA-256 `7ff85fa9a526ea599bec0a660557eda70a4c5620036a49f35de0422ecb241e6a` (Git blob `d31cbad6`), equal to the summary's `run_copy_sha256`. The cut copy's earlier code cells completed without an error output; the refusing cell's only error output has `ename` `IsolatedCellRefusal` and no traceback, and its output holds exactly one `Refused:` line:

```text
Refused: hostname: File is not a zip file; fix the input and run this cell again (the earlier results of this session and their outputs/ files were removed).
```

This run says nothing about the 15 cut cells, which did not run in it. Session `suite-zoedepth-e51670b-4905`, wall 117.3 s. Retained evidence: [`docs/execution-evidence/2026-10-29-e51670b-refusal/`](execution-evidence/2026-10-29-e51670b-refusal/).

**Local-path redaction.** Local paths were replaced byte-wise by `<local-run-dir>` (and the Colab CLI install path by `<colab-cli-tool-dir>/`) before the first commit; each executed notebook and run copy was checked against its `run_summary.json` before any edit.

| File (under `docs/execution-evidence/2026-10-29-e51670b/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `zoedepth_metric_depth_colab_e51670b_colab-cli-t4_default.ipynb` | `10f3de4eae27de2ae6d13cbbfa0364cf714a2460dd16aab424afd6e376fcfc69` | `10f3de4eae27de2ae6d13cbbfa0364cf714a2460dd16aab424afd6e376fcfc69` (unchanged, no local path) |
| `exec.log` | `2a739e1978c7c38d47badd500875df9e90519687ed7c109bcd2d9b5bdac2a24e` | `fdd2f286c26677f6fa5836ac4bbe23e64eb7bf3023d90005cb7c64a04d446fb0` (2 local-path replacements) |
| `run_summary.json` | `d181f0894d212290214c5a6639ea53c2caea47968b6e424bcab152088646767d` | `d181f0894d212290214c5a6639ea53c2caea47968b6e424bcab152088646767d` (unchanged, no local path) |

| File (under `docs/execution-evidence/2026-10-29-e51670b-refusal/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `zoedepth_metric_depth_colab_e51670b_colab-cli-t4_refusal.ipynb` | `62815793c0d317535e7a3ddc361da9fd37c6617c50b6797473f32e3fae0459b8` | `62815793c0d317535e7a3ddc361da9fd37c6617c50b6797473f32e3fae0459b8` (unchanged, no local path) |
| `exec.log` | `21f7fa91e2fec7c44f0f8596d5f2a1f6faf35fc23ab3ab75a0a0ef93428955df` | `95b3129855ad012e057b4206d7a4adbc2140cdeb1f77f18f76769040eeccf813` (2 local-path replacements) |
| `run_summary.json` | `dae6e50d57f9de8cb8d23a2c8d1e0cb59425293276feefe7b807e3fb51ac9d37` | `dae6e50d57f9de8cb8d23a2c8d1e0cb59425293276feefe7b807e3fb51ac9d37` (unchanged, no local path) |
| `run_copy.ipynb` | `7ff85fa9a526ea599bec0a660557eda70a4c5620036a49f35de0422ecb241e6a` | `7ff85fa9a526ea599bec0a660557eda70a4c5620036a49f35de0422ecb241e6a` (unchanged, no local path) |

Not exercised by these runs: an accepted BYOD input (REL12) and a browser `Run all` of the interactive journeys (REL2).

## Current status

**Current status (2026-10-25): Candidate.** `tutorials/zoedepth_metric_depth_colab.ipynb` blob `b0aa8183a8d95a24e24d0afd9828e0e7d5e9c245` (PR #8 head `63ee8cc`, tier C review fixes: markdown, the use notice, `metadata.dimer` and the `NOTEBOOK_SOURCE` record only) passed a Colab CLI sequential execution on a fresh Colab Tesla T4 at `63ee8cc` on 2026-10-25 (13/13 code cells, one pass, no restart, 0 errors, 136.1 s; see the 2026-10-25 row under "Recorded executions"; evidence in `docs/execution-evidence/2026-10-25-63ee8cc/`). This covers the **default path** only. Remaining gates: REL12 (a BYOD run, one input accepted and one refused) and REL2 (a browser run of the interactive journeys). Under REL14 the runs of earlier blobs recorded here are history for those blobs.

The current blob `444f0542` (code commit `a745e98`) shows a refused BYOD input as one `Refused: <file>: <reason>` line with no traceback, removes the earlier run's results and `outputs/` files and stops Run all at that cell, while genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20); on 2026-10-29 its default path (13 code cells, exit 0, 0 errors) and its scripted DAT20 refusal (`USE_BYOD=True;BYOD_ZIP_PATH=/etc/hostname`: exactly one `Refused:` line at cell `zoedepth_metric_depth-16`, no traceback, earlier cells clean, VM released) both passed on a hosted Colab Tesla T4 (Colab CLI `colab exec -f`, not a browser `Run all`). The remaining gates are REL12 (a hosted BYOD run with an accepted input) and REL2 (a browser run of the interactive journeys); earlier blobs' runs are history (REL14).

The corrected E2E implementation and generated carrier remain **Candidate**; the v9 run below needed a manual restart (corrected 2026-10-10 per the 2026-10-05 review, ZOE-M1), so it is not one-pass evidence; the one-pass clean-runtime record is the 2026-10-10 Colab T4 run of blob `441111a9df11` (commit `13c283a`) in the table above, with no restart. Promotion remains a reviewer/integrator decision; no BYOD run is recorded. The serial executor verified the Git blob, began with a clean Hugging Face cache, installed the declared pins, restarted once as designed after dependency replacement, downloaded and digest-verified the pinned base snapshot, completed all 11 code cells, and preserved the v9 output hashes under `docs/verification/2026-09-18-kaggle-t4/`. The synthetic held-out scores do not demonstrate a quality improvement, so the run qualifies the workflow only; promotion remains a separate reviewer/integrator decision.

## 2026-10-05 review fixes — verification pending

The 2026-10-05 review (`docs/reviews/2026-10-05-notebook-review/zoedepth_metric_depth_colab_Review.md`) found two problems with how the recorded run is presented:
- The "clean-runtime gate satisfied" statement rests on a run that restarted once after the in-kernel install cell, so it is not one-pass (REL2) evidence.
- The default scenes are laid out upside down compared with photographs. This explains why the pretrained model loses to the constant on δ1 and why held-out AbsRel gets worse after the head-only fine-tune.

The notebook was then regenerated (`zoedepth_metric_depth_colab_Fixes.md`):
- An isolated, hash-locked `uv` environment replaces the in-kernel install (spec 2.2), and CUDA is checked in Section 1, before the checkpoint download.
- The inverted layout is stated in Section 4 and shown as a figure, and Section 8 prints an interpretation against the constant. The generator and the recorded numbers are unchanged.
- The guided layer was added.
- BYOD gains a path field, a single-folder ZIP layout, a stated 8-pair minimum and a named sparse-depth refusal.
- Sections 6–7 restore the pretrained head, so re-runs do not accumulate training.

**The regenerated blob `441111a9df11` (commit `13c283a`) ran once on a fresh Colab Tesla T4 on 2026-10-10 with no restart** (13/13 code cells, 0 errors, 144.7 s; see Recorded executions). Status stays Candidate until reviewer/integrator promotion.
