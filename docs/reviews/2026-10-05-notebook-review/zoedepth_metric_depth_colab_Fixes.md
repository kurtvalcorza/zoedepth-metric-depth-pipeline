# ZoeDepth metric-depth fine-tuning notebook — review fixes

**Review:** `zoedepth_metric_depth_colab_Review.md` (5 October 2026, findings ZOE-M1 to ZOE-M3 and ZOE-m1 to ZOE-m3).

**Fixed in:**
- `tools/build_notebook.py`, now the isolated-runtime generator, vendored as `build_notebook.py/2.1-swc-zoe`. It adds an optional `runtime_check` key, which this template uses for the CUDA preflight.
- `tools/notebook_template.py`.
- `tutorials/requirements-colab.lock.txt`, a new hash lock.
- The validator, `tests/conftest.py`, `README.md`, `tutorials/README.md`, `docs/release-verification.md` and the tests.

The notebook was regenerated and `--check` passes. The carried `pipeline.py` and the sample generator are **unchanged**.

**Readiness:** **Verification pending** until the hosted gates below are recorded. `STATUS.md` and every release label are unchanged, including the "clean-runtime gate satisfied" wording; the record now carries an annotation next to it.

## Findings

| ID | Status | Change | Cells / files | Evidence |
|---|---|---|---|---|
| ZOE-M1 | Fixed in source; run and gate pending | The isolated `uv` runtime replaces the in-kernel install: pinned `uv` 0.12.15, managed CPython 3.12.12 and a hash lock of 47 packages. There is no kernel `pip` and no restart guard, and the notebook declares spec 2.2. The record gains a 2026-10-05 section stating that the satisfied gate rests on a restarted run and needs re-qualification with `restarted_after_install_cell: false`. The status wording itself is left for the maintainer. | Sections 1–3; validator; record | `test_zoe_M1_*` |
| ZOE-M2 | Fixed as disclosure and interpretation (generator unchanged, as instructed) | **Generator unchanged.** The sample was **not** re-laid out: that would change every recorded baseline, and the new numbers could not be reproduced here without the model.<br>**Disclosure.** Section 4 states that the scenes are upside down compared with photographs (≈0.9 m at the top row, ≈3.9 m at the bottom) and that the sample is kept as a domain-shift lesson. Before the baseline, a Predict prompt asks which way the pretrained prediction will slope.<br>**Section 8.** It now renders one held-out scene as four panels (RGB, reference, pretrained, adapted; written to `outputs/…_heldout_example.png`, using Pillow only) and prints an interpretation block. The block compares the model with the constant before and after training, gives the change in AbsRel/δ1, and gives the row correlation of each prediction with the reference. It also prints plain-language explanations when the slope is inverted or AbsRel worsened.<br>**Collapsed answers.** These quote the recorded numbers and explain them: AbsRel 0.431 vs constant 0.453; δ1 0.211 vs 0.258; AbsRel 0.466 after training while the validation loss fell; unseen-scene range 1.98–2.36 m vs 0.72–4.38 m.<br>**Closing.** The "Interpretation and limits" section now says what the numbers describe. | Sections 4, 6, 7, 8; closing | `test_zoe_M2_*` (real generated scenes: the top row is near and the bottom row far; the Section 5–8 cells run with a stand-in model that slopes the wrong way like the real one, so the figure and the interpretation are produced) |
| ZOE-M3 | Fixed | **Orientation:** How to use, an I/M/O table, a roadmap and a glossary (AbsRel, δ1, constant baseline, backbone/neck/head, log-L1, domain shift, row correlation, adapter).<br>**Active learning:** predictions with collapsed answers in Sections 4, 6, 7 and 8, plus a Section 12 activity (train longer) built on the ZOE-m2 fix.<br>**Support:** a troubleshooting table and a conclusion template.<br>**Infrastructure:** cells are collapsed. | template | `test_zoe_M3_*` (12 tests) |
| ZOE-m1 | Fixed | **Path and upload.** New field `BYOD_ZIP_PATH`; an upload must contain exactly one file.<br>**Layout.** A ZIP with a single top-level folder is accepted, and a ZIP with no pairs is refused with a message naming the layout.<br>**Minimum.** The 8–128-pair minimum is stated before upload and enforced in the reader with a message that names it.<br>**Sparse depth.** It is refused by name ("zero, negative or non-finite … sparse sensor depth marks missing pixels with 0; fill or crop them"). Masking it in the loss would change the carried module and the recorded adaptation, so it is left for the maintainer. | Section 4; Prerequisites; BYOD declaration | `test_zoe_m1_*` (real ZIPs: folder-zipped by path, 4 pairs, sparse depth, no pairs; cancelled or two-file upload) |
| ZOE-m2 | Fixed | **Pretrained head saved.** Section 6 captures `PRETRAINED_HEAD` once, from the freshly verified model, and restores it before scoring, so `base_eval` is always the pretrained score.<br>**Clean re-runs.** Section 7 restores it before every fine-tune, so re-running trains once from the pretrained head. `EPOCHS` and `LEARNING_RATE` are now form fields with the old defaults. | Sections 6–7 | `test_zoe_m2_rerunning_sections_6_and_7_does_not_accumulate_training` |
| ZOE-m3 | Fixed | **CUDA check moved.** It now runs in the Section 1 record-runtime cell, before the 1.38 GB download. The Section 7 check stays.<br>**README.** It no longer claims a BYOD run; it says the run needed a restart.<br>**`tutorials/README.md`.** It says the only hosted run is Kaggle T4, of the previous blob. | `runtime_check`; README files | `test_zoe_m3_*` |

**Suggestions:**
- S4 applied (spec 2.2).
- S1 partly: Section 4 says the held-out split is the last six scenes with no shuffle.
- S2 and S3 not taken.

## User-visible changes

- **Section 1:**
  - builds an isolated environment (Linux x86_64 only; the CUDA `torch` is several GB);
  - stops on a CPU runtime before the checkpoint download.
- **Sections 4–7:**
  - Section 4 has the new field `BYOD_ZIP_PATH`, and BYOD accepts a ZIP wrapped in one top-level folder.
  - Section 6 keeps the pretrained head.
  - Section 7 has `EPOCHS` and `LEARNING_RATE` fields and restores the head first.
- **Section 8 and outputs:**
  - Section 8 shows a four-panel figure.
  - `evaluation_report.json` gains `interpretation`, and a new output `…_heldout_example.png` is written.
  - The final status line now points to the Section 8 interpretation.
- **Unchanged:** default numbers are expected to stay the same, since the generator, seed, split and training are unchanged and the head restore is a no-op on the first run.
- **Tests:** without torch, `test_adaptation.py` is skipped at collection instead of erroring.

## Verification (offline, not clean-runtime evidence)

- **Checks:** `build_notebook.py --check` OK; `validate_release_assets.py` PASS; `ruff check src tests tools` clean.
- **pytest without torch:**
  - Before: one collection error. With `test_adaptation.py` ignored, 31 passed.
  - After: **54 passed** (23 new tests). CI installs torch and also runs `test_adaptation.py`.
- **Real inputs:** the generated scenes and the BYOD ZIPs are real. Model behaviour is played by a stand-in, so these tests prove the plumbing and the interpretation logic, not ZoeDepth.

## Remaining gates

1. A hosted one-pass **Run all** on a fresh T4, recorded with `restarted_after_install_cell: false`. Confirm that the default numbers match v9, then re-run the export cell.
2. The REL12 BYOD journey: one ZIP by path and one through the dialog, plus one refusal.
3. **Maintainer decisions:**
   - re-lay out the sample like photographs (this changes every recorded number);
   - mask sparse depth in the loss;
   - update `STATUS.md` and the gate wording after the run.
