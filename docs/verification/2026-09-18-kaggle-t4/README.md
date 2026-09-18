# ZoeDepth E2E Kaggle T4 verification

This directory retains the serial-suite evidence for the standalone ZoeDepth bounded fine-tuning E2E notebook.

## Immutable identity

- Target commit: `8db7fe8d8f955d370291a1aa568a92de167e1b6c`
- Embedded source revision: `71a2ee3ef019dc3da3730a62e0b0906608391493`
- Notebook: `tutorials/zoedepth_metric_depth_colab.ipynb`
- Git blob: `e9f6ab0fe13622dd9ea3ad58a110d36ede7a2d6c`
- Kernel: `kurtvalcorza/dimer-nb2-zoedepth-metric-depth`, version 8
- Kaggle image: `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`

## Result

- Outcome: **PASS**
- Runtime: Python 3.12.13, PyTorch 2.14.0+cu130, Transformers 4.57.6, Tesla T4 15,360 MiB
- Fresh Hugging Face cache: yes
- Wall time: 283.4 seconds
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
| `zoedepth_metric_depth_evaluation_report.json` | 800 | `705a16a0e9a610a007b1ab28b2e9ea035d7083ca988bb763cf5ecd93c3788768` |
| `zoedepth_metric_depth_result.json` | 4,420 | `fa591a84e1fb9ed273d7d2570e72313df0953c91caded8de857a85edc743cf7f` |
| `zoedepth_metric_depth_unseen_depth.npy` | 49,280 | `ec47675788b0c97053844c14257131297de50c691aceb770888b01cbd994d72f` |
| `zoedepth-metric-head-adapter-v1/adapter.safetensors` | 7,064,048 | `cd61880d9804e4f502820336171b7a416980f5f739482c67881275e323ee458c` |
| `zoedepth-metric-head-adapter-v1/manifest.json` | 2,492 | `9f490cb9bb276e9c5ab39a0f29051f5fd5e375b5b10a7b8ba1828c346cdbc526` |

`suite/` contains the historical and rejected bundles plus the corrected v8 generated executor, kernel metadata, first-pass restart record, successful executed notebook, `run_summary.json`, and preserved outputs. `LEDGER.md` records every executor pass and qualification rejection.
