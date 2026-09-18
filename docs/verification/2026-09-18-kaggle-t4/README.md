# ZoeDepth E2E Kaggle T4 verification

This directory retains the serial-suite evidence for the standalone ZoeDepth bounded fine-tuning E2E notebook.

## Immutable identity

- Target commit: `7bb7db6f401e7ffc9791d879949d56cc5fa9abaf`
- Embedded source revision: `2a196989a90489dd615855ebdfa4d669d0027617`
- Notebook: `tutorials/zoedepth_metric_depth_colab.ipynb`
- Git blob: `c5d97244af45b6522695ba5700a0bb1182edeebb`
- Kernel: `kurtvalcorza/dimer-nb2-zoedepth-metric-depth`, version 2
- Kaggle image: `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`

## Result

- Outcome: **PASS**
- Runtime: Python 3.12.13, PyTorch 2.14.0+cu130, Transformers 4.57.6, Tesla T4 15,360 MiB
- Fresh Hugging Face cache: yes
- Wall time: 263.8 seconds
- Cells: 11/11 successful after one expected restart following dependency replacement
- Model snapshot: four manifest entries downloaded and digest-verified at revision `f364d4c7936e91f465abba182208dd68142bf0ca`
- Training: 18 records, two epochs, 36 optimizer steps, 1,761,754 trainable parameters, weight delta 0.095016
- Held-out evaluation: pretrained AbsRel 0.431096 and delta1 0.211222; adapted AbsRel 0.454204 and delta1 0.211277; constant-median baseline 0.452605 / 0.257758
- Unseen generated scene: AbsRel 0.427249 and delta1 0.214600
- Artifact: SafeTensors adapter 7,064,048 bytes with a closed digest-bound manifest
- Reload: passed; maximum absolute depth difference 0.0

The metrics are synthetic sample-sanity evidence and do not demonstrate quality improvement or reproduce an NYU/KITTI benchmark.

## Preserved output hashes

| File | Bytes | SHA-256 |
|---|---:|---|
| `zoedepth_metric_depth_dataset_manifest.json` | 703 | `4a01bb35915e08486df2d1fdb0682364d157f6af959923d1beecaedaf9d4cb2b` |
| `zoedepth_metric_depth_evaluation_report.json` | 749 | `4b2e9e6e55d476b83b0bc04e81fb336f22256cb42b537498401e53120fa159fa` |
| `zoedepth_metric_depth_result.json` | 4,211 | `f15a55466937a5052f9ad067eb0e4fe3a0dd2e9b4b5948ef43f4ce5e214b10ad` |
| `zoedepth_metric_depth_unseen_depth.npy` | 49,280 | `a61bf018761dc02a7141ac18339331d12694d52d0cb93af52697b0a676a27418` |
| `zoedepth-metric-head-adapter-v1/adapter.safetensors` | 7,064,048 | `aab9119382c324b8a1b4f629a4e17a3864bc90224aae892ec6b020b64d0e9383` |
| `zoedepth-metric-head-adapter-v1/manifest.json` | 2,347 | `c62ad23006a4477dae83afb934d6d88aa908248e0c3315aaba196c264725d9ee` |

`suite/` contains the generated executor, kernel metadata, first-pass restart record, successful executed notebook, `run_summary.json`, and preserved outputs. `LEDGER.md` is the serial-suite audit table.
