"""E2E notebook template for ZoeDepth metric-head adaptation."""

# ruff: noqa: E501

TEMPLATE = {
    "package": "zoedepth_metric_depth_pipeline",
    "repo_name": "zoedepth-metric-depth-pipeline",
    "stem": "zoedepth_metric_depth",
    "notebook_name": "zoedepth_metric_depth_colab.ipynb",
    "profile": "E2E",
    "run_all": (
        "Selecting **Run all** on a fresh CUDA runtime installs pinned dependencies, verifies the exact base snapshot, "
        "generates and validates paired RGB/depth records, freezes the backbone and decoder neck, fine-tunes the metric "
        "head, evaluates the held-out split against a training-median baseline, predicts a new image, exports a safe "
        "adapter, reloads it over a fresh base, verifies numeric equivalence, and writes provenance without a clone, "
        "credential, upload, or configuration edit."
    ),
    "byod": (
        "Set `USE_BYOD = True` to upload a bounded ZIP containing paired `images/` files and `depth/<id>.npy` metric "
        "depth arrays. BYOD follows the same validation, split, adaptation, evaluation, export, and reload path."
    ),
    "pipeline_class": "ZoeDepthMetricPipeline",
    "weights_key": "zoedepth-nyu-kitti",
    "runtime_imports": ["torch", "transformers", "safetensors"],
    "title": "ZoeDepth NYU+KITTI — DIMER end-to-end metric-depth fine-tuning",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/zoedepth-metric-depth-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/zoedepth-metric-depth-pipeline/blob/main/tutorials/zoedepth_metric_depth_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Intel%2Fzoedepth--nyu--kitti-ffcc4d?style=flat",
            "https://huggingface.co/Intel/zoedepth-nyu-kitti",
        ),
    ],
    "capability": "pinned ZoeDepth metric inference plus bounded metric-head gradient adaptation, held-out AbsRel/δ1 evaluation, safe adapter export, and fresh reload",
    "intro": (
        "This notebook freezes the 304M-parameter BEiT backbone and 39M-parameter DPT neck, then trains only the 1.76M "
        "parameter metric head with mean absolute log-depth error. The default data are 24 deterministic generated "
        "RGB/depth pairs split 18/6 before loading the model. Evaluation reports AbsRel and δ1 against a constant "
        "training-median baseline. A SafeTensors metric-head adapter is integrity-bound to the exact pinned base and "
        "verified after fresh reconstruction. Generated-scene scores are sample-sanity, not a depth benchmark."
    ),
    "learning_objectives": (
        "validate aligned RGB and positive metric-depth targets, preserve a held-out split, measure the pretrained "
        "baseline, perform bounded metric-head adaptation, interpret AbsRel and δ1 against a trivial baseline, run "
        "new-image inference, export a base-bound SafeTensors adapter, and verify fresh reload equivalence."
    ),
    "exclusions": (
        "full-model or backbone training, camera-intrinsic estimation, point-cloud generation, benchmark claims, or "
        "production calibration. The adapter modifies the existing NYU/KITTI metric head only."
    ),
    "prerequisites": [
        "- **Runtime:** fresh Python 3.12 with an NVIDIA T4-class GPU or better. CUDA is required. The pinned 1.38 GB checkpoint is acquired and digest-verified automatically.",
        "- **Knowledge:** Python, metric depth, train/validation separation, AbsRel, δ1, and adapter/base dependencies.",
        "- **Data:** the default path generates 24 paired scenes. Optional BYOD accepts images and positive finite `.npy` depth maps in metres, paired by stem and capped by the validator. Do not upload confidential or restricted data unless you are authorized to use it in the hosted runtime.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Generate or upload paired RGB and metric-depth data\n\n"
                "The default 24 deterministic 128×96 scenes encode sloped surfaces plus foreground panels with known "
                "depth in metres. Records 0–17 train and 18–23 remain held out. Optional BYOD requires matching "
                "`images/<id>` and `depth/<id>.npy` members and rejects traversal before decoding."
            ),
            "code": (
                "import io\n"
                "import zipfile\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n\n"
                "def generated_depth_records(start=0, count=24):\n"
                "    records = []\n"
                "    height, width = 96, 128\n"
                "    yy, xx = np.mgrid[:height, :width]\n"
                "    for index in range(start, start + count):\n"
                "        phase = index * 0.19\n"
                "        depth = 0.9 + 2.6 * yy / (height - 1) + 0.7 * xx / (width - 1)\n"
                "        depth += 0.18 * np.sin(xx / 13.0 + phase)\n"
                "        x0, x1 = 18 + index % 9, 62 + index % 13\n"
                "        y0, y1 = 22 + index % 7, 66 + index % 11\n"
                "        depth[y0:y1, x0:x1] = 1.15 + 0.02 * (index % 5)\n"
                "        red = np.clip((depth - 0.7) / 4.0 * 255.0, 0, 255)\n"
                "        green = np.clip(yy / (height - 1) * 255.0 + index * 2, 0, 255)\n"
                "        blue = np.clip(xx / (width - 1) * 210.0 + 25.0, 0, 255)\n"
                "        rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)\n"
                "        rgb[y0:y1, x0:x1] = np.array([45 + index * 3, 185, 75], dtype=np.uint8)\n"
                "        records.append({{'id': f'generated-depth-{{index:02d}}', 'image': Image.fromarray(rgb), 'depth_m': depth.astype(np.float32)}})\n"
                "    return records\n\n"
                "def records_from_zip(blob):\n"
                "    if len(blob) > 256 * 1024 * 1024:\n"
                "        raise ValueError('BYOD ZIP exceeds the 256 MiB upload ceiling')\n"
                "    with zipfile.ZipFile(io.BytesIO(blob)) as archive:\n"
                "        infos = archive.infolist()\n"
                "        names = archive.namelist()\n"
                "        normalized = [name.replace('\\\\', '/') for name in names]\n"
                "        max_archive_entries = 2 * MAX_ADAPTATION_RECORDS + 2  # paired files plus directory entries\n"
                "        if len(names) > max_archive_entries or sum(info.file_size for info in infos) > 512 * 1024 * 1024:\n"
                "            raise ValueError('unsafe or oversized BYOD archive')\n"
                "        if any(info.flag_bits & 1 for info in infos) or any(name.startswith('/') or '..' in name.split('/') for name in normalized):\n"
                "            raise ValueError('unsafe or oversized BYOD archive')\n"
                "        image_entries = [(name.split('/')[-1].rsplit('.', 1)[0], original) for name, original in zip(normalized, names) if name.startswith('images/') and name.lower().endswith(('.png', '.jpg', '.jpeg'))]\n"
                "        depth_entries = [(name.split('/')[-1].rsplit('.', 1)[0], original) for name, original in zip(normalized, names) if name.startswith('depth/') and name.lower().endswith('.npy')]\n"
                "        images, depths = dict(image_entries), dict(depth_entries)\n"
                "        if len(images) != len(image_entries) or len(depths) != len(depth_entries):\n"
                "            raise ValueError('BYOD archive contains duplicate image or depth stems')\n"
                "        if set(images) != set(depths):\n"
                "            raise ValueError('BYOD image and depth stems must match exactly')\n"
                "        if not 2 <= len(images) <= MAX_ADAPTATION_RECORDS:\n"
                "            raise ValueError(f'BYOD record count must be in 2..{{MAX_ADAPTATION_RECORDS}}')\n"
                "        records = []\n"
                "        total_image_pixels = 0\n"
                "        for key in sorted(images):\n"
                "            image = Image.open(io.BytesIO(archive.read(images[key])))\n"
                "            width, height = image.size\n"
                "            short_side, long_side = min(width, height), max(width, height)\n"
                "            if short_side < MIN_IMAGE_SIDE or long_side > MAX_IMAGE_SIDE or long_side / short_side > MAX_ASPECT_RATIO:\n"
                "                raise ValueError(f'{{key}} image dimensions are outside the public image ceilings')\n"
                "            total_image_pixels += width * height\n"
                "            if total_image_pixels > MAX_ADAPTATION_PIXELS:\n"
                "                raise ValueError(f'BYOD images exceed MAX_ADAPTATION_PIXELS {{MAX_ADAPTATION_PIXELS}}')\n"
                "            image.load()\n"
                "            depth = np.load(io.BytesIO(archive.read(depths[key])), allow_pickle=False).astype(np.float32)\n"
                "            records.append({{'id': key, 'image': image.convert('RGB'), 'depth_m': depth}})\n"
                "        return records\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    dataset_records = records_from_zip(next(iter(uploaded.values())))\n"
                "    dataset_kind = 'BYOD'\n"
                "else:\n"
                "    dataset_records = generated_depth_records()\n"
                "    dataset_kind = 'generated'\n"
                "if len(dataset_records) < 8:\n"
                "    raise ValueError('E2E adaptation requires at least 8 paired records')\n"
                "split_at = max(2, int(len(dataset_records) * 0.75))\n"
                "train_records, val_records = dataset_records[:split_at], dataset_records[split_at:]\n"
                "print({{'dataset_kind': dataset_kind, 'records': len(dataset_records), 'train': len(train_records), 'held_out': len(val_records)}})"
            ),
        },
        {
            "md": "## 5. Validate alignment and split integrity\n\nThe validator checks unique IDs, image ceilings, exact depth/image shape, finite positive metres, the 80 m ceiling, and content fingerprints. Cross-split IDs are forbidden.",
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "train_manifest = validate_depth_dataset(train_records)\n"
                "val_manifest = validate_depth_dataset(val_records)\n"
                "overlap = set(r['id'] for r in train_records) & set(r['id'] for r in val_records)\n"
                "if overlap:\n"
                "    raise RuntimeError(f'train/validation leakage: {{sorted(overlap)}}')\n"
                "dataset_manifest = {{'kind': dataset_kind, 'train': train_manifest, 'validation': val_manifest, 'overlap_ids': []}}\n"
                "with open('outputs/{stem}_dataset_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(dataset_manifest, handle, indent=2)\n"
                "print(json.dumps(dataset_manifest, indent=2))"
            ),
        },
        {
            "md": "## 6. Measure the pretrained baseline\n\nScore the untouched holdout with AbsRel and δ1. The trivial comparator predicts the training split's median depth at every pixel.",
            "code": (
                "def score_records(model_pipe, records, median_depth):\n"
                "    model_abs_sum = base_abs_sum = 0.0\n"
                "    model_delta_hits = base_delta_hits = valid_pixels = 0\n"
                "    for record in records:\n"
                "        prediction = np.asarray(model_pipe.predict(record['image'])['depth'], dtype=np.float64)\n"
                "        reference = np.asarray(record['depth_m'], dtype=np.float64)\n"
                "        baseline = np.full_like(reference, median_depth)\n"
                "        valid = _valid_mask(prediction, reference)\n"
                "        count = int(valid.sum()); valid_pixels += count\n"
                "        model_abs_sum += float(np.sum(np.abs(prediction[valid] - reference[valid]) / reference[valid]))\n"
                "        base_abs_sum += float(np.sum(np.abs(baseline[valid] - reference[valid]) / reference[valid]))\n"
                "        model_ratio = np.maximum(prediction[valid] / reference[valid], reference[valid] / prediction[valid])\n"
                "        base_ratio = np.maximum(baseline[valid] / reference[valid], reference[valid] / baseline[valid])\n"
                "        model_delta_hits += int(np.sum(model_ratio < DELTA_THRESHOLD))\n"
                "        base_delta_hits += int(np.sum(base_ratio < DELTA_THRESHOLD))\n"
                "    return {{'records': len(records), 'valid_pixels': valid_pixels, 'abs_rel': model_abs_sum / valid_pixels, 'delta1': model_delta_hits / valid_pixels, 'constant_median_baseline_abs_rel': base_abs_sum / valid_pixels, 'constant_median_baseline_delta1': base_delta_hits / valid_pixels}}\n\n"
                "training_median = train_manifest['depth_median_m']\n"
                "base_eval = score_records(pipe, val_records, training_median)\n"
                "print(json.dumps(base_eval, indent=2))"
            ),
        },
        {
            "md": "## 7. Freeze the base and run bounded metric-head fine-tuning\n\nOnly `metric_head.*` is trainable. AdamW runs two epochs at batch size one with mean absolute log-depth error. The run must produce a non-zero weight delta.",
            "code": (
                "if not torch.cuda.is_available():\n"
                "    raise RuntimeError('The canonical ZoeDepth E2E path requires a CUDA GPU')\n"
                "parameter_counts = pipe.freeze_for_adaptation()\n"
                "history = pipe.finetune(train_records, val_records, epochs=2, learning_rate=1e-5, seed=42)\n"
                "print({{'parameters': parameter_counts, 'history': history, 'weight_delta_l2': pipe.adaptation_config['weight_delta_l2']}})"
            ),
        },
        {
            "md": "## 8. Evaluate the adapted model\n\nEvaluate the same untouched holdout against the training-median baseline. Lower AbsRel and higher δ1 are better, but generated-scene values remain sample-sanity.",
            "code": (
                "adapted_eval = pipe.evaluate_adaptation(val_records)\n"
                "evaluation_report_e2e = {{'task': 'monocular-metric-depth-adaptation', 'verdict': 'sample-sanity', 'estimation': f'fixed {{dataset_kind}} held-out split', 'baseline_pretrained': base_eval, 'adapted': adapted_eval, 'weight_delta_l2': pipe.adaptation_config['weight_delta_l2']}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(evaluation_report_e2e, handle, indent=2)\n"
                "print(json.dumps(evaluation_report_e2e, indent=2))"
            ),
        },
        {
            "md": "## 9. Infer on an unseen generated image\n\nA newly generated record outside the train/validation index range exercises adapted serving and produces a metric-depth array.",
            "code": (
                "unseen = generated_depth_records(start=24, count=1)[0]\n"
                "unseen_result = pipe.predict(unseen['image'])\n"
                "unseen_abs_rel = abs_rel(unseen_result['depth'], unseen['depth_m'])\n"
                "unseen_delta1 = delta1(unseen_result['depth'], unseen['depth_m'])\n"
                "np.save('outputs/{stem}_unseen_depth.npy', unseen_result['depth'])\n"
                "print({{'id': unseen['id'], 'abs_rel_sample_sanity': unseen_abs_rel, 'delta1_sample_sanity': unseen_delta1, 'depth_range_m': [unseen_result['depth_min'], unseen_result['depth_max']]}})"
            ),
        },
        {
            "md": "## 10. Export and verify a fresh reload\n\nExport the metric head as SafeTensors with a closed manifest. A fresh base verifies the artifact and must reproduce the unseen depth array within explicit tolerances.",
            "code": (
                "artifact_dir = pipe.save_artifact('outputs/zoedepth-metric-head-adapter-v1', producer_revision=NOTEBOOK_SOURCE['repository_revision'])\n"
                "reloaded_pipe = ZoeDepthMetricPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR)\n"
                "reloaded_result = reloaded_pipe.predict(unseen['image'])\n"
                "max_abs_diff = float(np.max(np.abs(unseen_result['depth'] - reloaded_result['depth'])))\n"
                "if not np.allclose(unseen_result['depth'], reloaded_result['depth'], rtol=1e-5, atol=1e-5):\n"
                "    raise RuntimeError(f'reloaded adapter depth mismatch: max_abs_diff={{max_abs_diff}}')\n"
                "reload_summary = {{'verification': 'PASSED', 'rtol': 1e-5, 'atol': 1e-5, 'max_abs_diff': max_abs_diff, 'artifact_dir': str(artifact_dir)}}\n"
                "print(reload_summary)"
            ),
        },
        {
            "md": "## 11. Export provenance and terminal summary\n\nBind dataset fingerprints, split sizes, hyperparameters, metrics, weight activity, base identity, runtime, artifact, and reload evidence without retaining source records.",
            "code": (
                "payload = {{'dataset': dataset_manifest, 'adaptation': pipe.adaptation_config, 'evaluation': evaluation_report_e2e, 'unseen': {{'id': unseen['id'], 'abs_rel': unseen_abs_rel, 'delta1': unseen_delta1, 'depth_file': 'outputs/{stem}_unseen_depth.npy'}}, 'artifact': reload_summary, 'notebook_source': NOTEBOOK_SOURCE, 'repository_revision': NOTEBOOK_SOURCE['repository_revision'], 'base_model': {{'id': MODEL_ID, 'revision': MODEL_REVISION}}, 'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device}}}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2)\n"
                "print({{'status': 'E2E COMPLETE', 'train_records': len(train_records), 'held_out_records': len(val_records), 'optimizer_steps': sum(item['optimizer_steps'] for item in history), 'weight_delta_l2': pipe.adaptation_config['weight_delta_l2'], 'reload': reload_summary['verification'], 'outputs': sorted(os.listdir('outputs'))}})"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "This run proves that the exact notebook can validate aligned RGB/depth pairs, update only the declared metric "
        "head, score a held-out generated split, serialize the adapter, attach it to the exact pinned base, and reproduce "
        "inference after reload. Generated gradients and panels are not photographs or sensor depth. Real use requires "
        "rights-cleared RGB-D data from the target camera and domain, leakage-safe splits, missing-depth policy, and "
        "calibration across depth ranges and scene types.\n\n"
        "Successful execution proves that the recorded repository revision can complete this bounded tutorial without the repository being reachable at runtime. It does **not** establish benchmark superiority or production fitness.\n\n"
        "## References\n\n"
        "- Repository: https://github.com/kurtvalcorza/zoedepth-metric-depth-pipeline\n"
        "- Repository model card: https://github.com/kurtvalcorza/zoedepth-metric-depth-pipeline/blob/main/MODEL_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- ZoeDepth paper: https://arxiv.org/abs/2302.12288"
    ),
}
