# ZoeDepth E2E Kaggle T4 verification

This directory retains the serial-suite evidence for the standalone ZoeDepth bounded fine-tuning E2E notebook.

## Immutable identity

- Target commit: `1cfd4977d330cc04de628d6bc327e577d1228ccc`
- Embedded source revision: `4ddf37ea1f002497dbfc238cf44aae74c2dc47b1`
- Notebook: `tutorials/zoedepth_metric_depth_colab.ipynb`
- Git blob: `0e4c86304b1102dda3d30900eb32e94e585b0768`
- Kernel: `kurtvalcorza/dimer-nb2-zoedepth-metric-depth`, version 5
- Kaggle image: `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`

## Result

- Outcome: **PASS**
- Runtime: Python 3.12.13, PyTorch 2.14.0+cu130, Transformers 4.57.6, Tesla T4 15,360 MiB
- Fresh Hugging Face cache: yes
- Wall time: 278.1 seconds
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
| `zoedepth_metric_depth_evaluation_report.json` | 800 | `04445643a3d2926ca94ca4b5110dcb7bc5f73aa910677533e789596a9be8f692` |
| `zoedepth_metric_depth_result.json` | 4,420 | `6b9d5034c3133d0044a120e8ca58966013a4636959b86b47c8e8920f1dd88239` |
| `zoedepth_metric_depth_unseen_depth.npy` | 49,280 | `1783efa230c47a75840d9e46d3f8d0338854dfa00d71b0dcd031a9f3b0ffddf4` |
| `zoedepth-metric-head-adapter-v1/adapter.safetensors` | 7,064,048 | `33cba221a74c2f4f93ed6889e1d3f8a288df713e5585f2d86019c8e2cd0d8460` |
| `zoedepth-metric-head-adapter-v1/manifest.json` | 2,493 | `34f88b58a474904f2d02f99e265e2686c18e5de968884baf619c3079d18f4fad` |

`suite/` contains the historical v2 bundle, rejected v3 and v4 bundles, and the corrected v5 generated executor, kernel metadata, first-pass restart record, successful executed notebook, `run_summary.json`, and preserved outputs. `LEDGER.md` records the executor passes and qualification rejections.
