# ZoeDepth E2E Kaggle T4 verification

This directory retains the serial-suite evidence for the standalone ZoeDepth bounded fine-tuning E2E notebook.

## Immutable identity

- Target commit: `ac175a534c57a9d9ddd7dc4bad282bc99f56051c`
- Embedded source revision: `f0cedd2892eebaf99e6b598aa60cecf669ba50f4`
- Notebook: `tutorials/zoedepth_metric_depth_colab.ipynb`
- Git blob: `3d3254ea89e196d9f0bd484da8a59275ac5589e4`
- Kernel: `kurtvalcorza/dimer-nb2-zoedepth-metric-depth`, version 9
- Kaggle image: `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`

## Result

- Outcome: **PASS**
- Runtime: Python 3.12.13, PyTorch 2.14.0+cu130, Transformers 4.57.6, Tesla T4 15,360 MiB
- Fresh Hugging Face cache: yes
- Wall time: 265.5 seconds
- Cells: 11/11 successful after one expected restart following dependency replacement
- Model snapshot: four manifest entries downloaded and digest-verified at revision `f364d4c7936e91f465abba182208dd68142bf0ca`
- Training: 18 records, two epochs, 36 optimizer steps, 1,761,754 trainable parameters, weight delta 0.098801
- Held-out evaluation: pretrained AbsRel 0.431096 and delta1 0.211222; adapted AbsRel 0.466203 and delta1 0.212009; constant-median baseline 0.452605 / 0.257758
- Unseen generated scene: AbsRel 0.433712 and delta1 0.215007
- Artifact: SafeTensors adapter 7,064,048 bytes with a closed digest-bound manifest
- Reload: passed; maximum absolute depth difference 0.0

The metrics are synthetic sample-sanity evidence and do not demonstrate quality improvement or reproduce an NYU/KITTI benchmark.

## Preserved output hashes

| File | Bytes | SHA-256 |
|---|---:|---|
| `zoedepth_metric_depth_dataset_manifest.json` | 703 | `aeafa65db08d19608a42c35f376461fd1b008b8ce4b3bf0dfd1da7c391c6504e` |
| `zoedepth_metric_depth_evaluation_report.json` | 799 | `e6aef1a2fd807a98d105aec8556a8dbf49960389fb839b8590c24fdf3938de08` |
| `zoedepth_metric_depth_result.json` | 4,418 | `46ffd43c89e7c65cdb6570bd6a8c686cec9fdb3abf6ff92d138fbf57c8aea76b` |
| `zoedepth_metric_depth_unseen_depth.npy` | 49,280 | `bf567631c415c99d7b16a99b893bfa3e5439a57eea66600bca2b40a8db4a2e5a` |
| `zoedepth-metric-head-adapter-v1/adapter.safetensors` | 7,064,048 | `c16d4600249b26db8f79fff0bfae6db00b75d3feb4ef57fb731acd2a0e2f9cac` |
| `zoedepth-metric-head-adapter-v1/manifest.json` | 2,492 | `fb44a9058d9b4e7e300f165dcb5c392d10709d86cac16aa1700c580a453219c5` |

`suite/` contains the historical and rejected bundles plus the corrected v9 generated executor, kernel metadata, first-pass restart record, successful executed notebook, `run_summary.json`, and preserved outputs. `LEDGER.md` records every executor pass and qualification rejection.
