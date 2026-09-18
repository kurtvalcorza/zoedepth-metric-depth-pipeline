# ZoeDepth E2E Kaggle T4 verification

This directory retains the serial-suite evidence for the standalone ZoeDepth bounded fine-tuning E2E notebook.

## Immutable identity

- Target commit: `b1f91a13621959e7f41f72b4ab69cfda7ba62b28`
- Embedded source revision: `e59149e41f93db06d7b67687be293d14ad0b9994`
- Notebook: `tutorials/zoedepth_metric_depth_colab.ipynb`
- Git blob: `810883ed15211487e09faa6a6b6d5acbd811e339`
- Kernel: `kurtvalcorza/dimer-nb2-zoedepth-metric-depth`, version 4
- Kaggle image: `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`

## Result

- Executor outcome: **PASS**
- Qualification outcome: **REJECTED** — subsequent review found nine defects; this v4 bundle is retained as historical evidence and does not qualify the corrected carrier.
- Runtime: Python 3.12.13, PyTorch 2.14.0+cu130, Transformers 4.57.6, Tesla T4 15,360 MiB
- Fresh Hugging Face cache: yes
- Wall time: 254.6 seconds
- Cells: 11/11 successful after one expected restart following dependency replacement
- Model snapshot: four manifest entries downloaded and digest-verified at revision `f364d4c7936e91f465abba182208dd68142bf0ca`
- Training: 18 records, two epochs, 36 optimizer steps, 1,761,754 trainable parameters, weight delta 0.098800
- Held-out evaluation: pretrained AbsRel 0.431096 and delta1 0.211222; adapted AbsRel 0.466203 and delta1 0.212009; constant-median baseline 0.452605 / 0.257758
- Unseen generated scene: AbsRel 0.433712 and delta1 0.215007
- Artifact: SafeTensors adapter 7,064,048 bytes with a closed digest-bound manifest
- Reload: passed; maximum absolute depth difference 0.0

The metrics are synthetic sample-sanity evidence and do not demonstrate quality improvement or reproduce an NYU/KITTI benchmark.

## Preserved output hashes

| File | Bytes | SHA-256 |
|---|---:|---|
| `zoedepth_metric_depth_dataset_manifest.json` | 703 | `4a01bb35915e08486df2d1fdb0682364d157f6af959923d1beecaedaf9d4cb2b` |
| `zoedepth_metric_depth_evaluation_report.json` | 747 | `d0ceec7db624956a25c3cb5c676b892d4950e4b59ab1f6e54d7f3b380c6c01ba` |
| `zoedepth_metric_depth_result.json` | 4,215 | `51d3c985799ee39075ab3c789747af08ec08259eecd4aa2519ba308873acd212` |
| `zoedepth_metric_depth_unseen_depth.npy` | 49,280 | `26639929607193cae286c0c3fc7956e1dbc5cae2778357b35c2ebe37fd474baa` |
| `zoedepth-metric-head-adapter-v1/adapter.safetensors` | 7,064,048 | `a958f03752dadfc995cb8231558d291e22dedf48d9a74f30fd06858babff4ebf` |
| `zoedepth-metric-head-adapter-v1/manifest.json` | 2,344 | `1adaa6499e7e05f2fbb0168d494fef5b64b930199a52cfe89ae8c64ba6a5657f` |

`suite/` contains the historical v2 bundle and the corrected v4 generated executor, kernel metadata, first-pass restart record, successful executed notebook, `run_summary.json`, and preserved outputs. `LEDGER.md` also records the v3 executor pass that was rejected for qualification because its embedded source revision was stale.
