"""E2E notebook template for ZoeDepth metric-head adaptation (NOTEBOOK_SPEC 2.5 §4 standalone carrier)."""

# ruff: noqa: E501

TEMPLATE = {
    "package": "zoedepth_metric_depth_pipeline",
    "repo_name": "zoedepth-metric-depth-pipeline",
    "stem": "zoedepth_metric_depth",
    "notebook_name": "zoedepth_metric_depth_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    # NOTEBOOK_SPEC 2.2 §5 (ZOE-M1): the pins are installed into an isolated uv environment and every later cell runs in
    # a persistent worker there, so a hosted runtime's preloaded packages never force a restart. The lock is compiled with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    # ZOE-m3: fail closed on the wrong runtime before the 1.38 GB checkpoint is downloaded.
    "runtime_check": (
        "if not torch.cuda.is_available():\n"
        "    raise RuntimeError('This notebook needs an NVIDIA GPU exposed as CUDA (Colab: Runtime > Change runtime type > T4 GPU); "
        "it stops here, before the 1.38 GB checkpoint download.')"
    ),
    "run_all": (
        "Selecting **Run all** on a fresh CUDA runtime builds an isolated Python environment from the hash-locked pins (the "
        "kernel's own packages are left alone, so no runtime restart is needed; a CPU runtime stops in Section 1, before the "
        "checkpoint download), verifies the exact base snapshot, "
        "generates and validates paired RGB/depth records, freezes the backbone and decoder neck, fine-tunes the metric "
        "head, evaluates the held-out split against a training-median baseline, predicts a new image, exports a safe "
        "adapter, reloads it over a fresh base, verifies numeric equivalence, and writes provenance without a clone, "
        "credential, upload, configuration edit or restart (NOTEBOOK_SPEC 2.5 §5). Hosted runs are recorded per notebook blob in `docs/release-verification.md`: the previous blob `441111a9` "
        "completed one pass on a Colab T4 on 2026-10-10 (144.7 s), and the Kaggle T4 run of an earlier notebook (2026-09-18) needed a restart."
    ),
    "byod": (
        "Set `USE_BYOD = True` and either give a ZIP path in `BYOD_ZIP_PATH` or upload exactly one ZIP containing paired "
        "`images/<id>.png|jpg` files and `depth/<id>.npy` metric-depth arrays (8–128 pairs; `images/` and `depth/` at the archive "
        "root or inside one top-level folder; every depth value finite and positive, in metres). BYOD follows the same "
        "validation, split, adaptation, evaluation, export, and reload path."
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
        "baseline, perform bounded metric-head adaptation, interpret AbsRel and δ1 against a trivial baseline (and recognise when "
        "the data, not the model, explains the result), run "
        "new-image inference, export a base-bound SafeTensors adapter, and verify fresh reload equivalence."
    ),
    "exclusions": (
        "full-model or backbone training, camera-intrinsic estimation, point-cloud generation, benchmark claims, or "
        "production calibration. The adapter modifies the existing NYU/KITTI metric head only."
    ),
    "guided_opening": [
        (
            "## How to use this notebook\n\n"
            "**Who this notebook is for.** Learners who can open a hosted notebook (Google Colab or Kaggle), run cells in order and read short "
            "Python, and who want to see how a pretrained metric-depth model is evaluated against a trivial baseline, adapted, exported and "
            "reloaded — and how to read the result when it is not the one you hoped for. No prior experience with ZoeDepth is assumed: each term "
            "is explained where it is first needed and again in the glossary below. A T4 GPU is required. The **Prerequisites** give the details.\n\n"
            "**Running it.** In Colab choose *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. The default path needs no edit, "
            "no upload, no account, no token and no runtime restart. Section 1 builds the isolated environment (the CUDA build of `torch` is "
            "several GB, so it is the slowest step) and stops at once on a CPU runtime; a second Run all in the same runtime reuses it.\n\n"
            "**Where the code runs.** The first two code cells run in the notebook kernel: they build the environment and start one Python process "
            "inside it. Every later cell is sent to that process, so the pinned `torch` and `transformers` are used without replacing anything "
            "the hosted runtime had already loaded. Printed output, images and errors come back to the notebook as usual, and variables persist.\n\n"
            "**Two kinds of cell.** *Infrastructure cells* (Sections 1–3: the isolated install and router, the carried pipeline module and the "
            "pinned-checkpoint staging) are collapsed and labelled **Infrastructure**; you may run them without studying their implementation. "
            "*Learner cells* (Sections 4–11) are the machine-learning workflow.\n\n"
            "**Form controls.** `USE_BYOD` and `BYOD_ZIP_PATH` (Section 4); `EPOCHS` and `LEARNING_RATE` (Section 7). Leave them at their defaults "
            "for the first run. After a change, select the changed cell and choose *Runtime → Run after*.\n\n"
            "**Section tags.** Each learner heading carries one tag. **[Concept]** — what the model does and why. **[Evaluation practice]** — how "
            "the evidence is produced and how to read it. **[Engineering]** — reproducibility, provenance and packaging.\n\n"
            "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit to an expectation; after "
            "it, **What to notice** describes normal output and a collapsed **Check your reasoning** block gives a worked answer. Write your own "
            "answer first, then open it. The numbers the answers quote come from the Colab T4 run of blob `441111a9` (commit `13c283a`, 2026-10-10), "
            "which printed the same metrics as the Kaggle T4 run of the previous version (18 September 2026); neither was re-run for this revision."
        ),
        (
            "## The task: Input → Model/System → Output\n\n"
            "| Stage | Input | Model / system | Output |\n"
            "|---|---|---|---|\n"
            "| **Validate and split** | 24 generated RGB images with metric depth (or your ZIP) | the carried `validate_depth_dataset` | manifests; 18 train / 6 held out |\n"
            "| **Baseline** | the held-out images | ZoeDepth NYU+KITTI (BEiT backbone, DPT neck, metric head), and a constant at the training median depth | AbsRel and δ1 for both |\n"
            "| **Adapt** | the 18 training pairs | only the 1.76 M-parameter metric head trains; log-depth L1 loss, AdamW | an adapted head and its loss history |\n"
            "| **Evaluate, infer, export** | held-out and one unseen image | the adapted model; a SafeTensors adapter reloaded on a fresh base | AbsRel/δ1 after training, a four-panel figure, an identical reload |\n\n"
            "## Roadmap\n\n"
            "| Section | Tag | What happens | What you read |\n"
            "|---|---|---|---|\n"
            "| 1. Install the pinned runtime | [Engineering] | isolated environment; CUDA check | versions, CUDA |\n"
            "| 2. Pipeline code | [Engineering] | the repository's module, carried verbatim | nothing to run by hand |\n"
            "| 3. Pin, stage and verify the model | [Engineering] | 1.38 GB snapshot downloaded and digest-checked | the verified files |\n"
            "| 4. Data | [Evaluation practice] | generated scenes (or your ZIP) — upside down on purpose | counts |\n"
            "| 5. Validate | [Evaluation practice] | alignment, ceilings, no leakage | the manifests |\n"
            "| 6. Baseline | [Evaluation practice] | pretrained model and constant on the holdout | AbsRel, δ1 |\n"
            "| 7. Fine-tune | [Concept] | two epochs on the metric head | loss history |\n"
            "| 8. Evaluate and look | [Evaluation practice] | scores after training, one scene as four panels | the interpretation block |\n"
            "| 9. Unseen image | [Concept] | adapted inference | depth range |\n"
            "| 10. Export and reload | [Engineering] | SafeTensors adapter on a fresh base | `PASSED` |\n"
            "| 11. Provenance | [Engineering] | JSON outputs | the file list |\n"
            "| 12. Activity (optional) | [Concept] | train longer and compare | your comparison |\n"
            "| Troubleshooting, Conclusion | — | recovery, your notes | when needed |\n\n"
            "**Fast path.** Short on time? Run all, then read Sections 4, 6 and 8 and the conclusion."
        ),
        (
            "<details>\n<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
            "| Term | Meaning in this notebook |\n"
            "|---|---|\n"
            "| **Metric depth** | Distance to the camera in metres at every pixel (not just nearer/farther ordering). |\n"
            "| **AbsRel** | Mean of \\|prediction − reference\\| / reference over valid pixels; 0 is perfect, lower is better. |\n"
            "| **δ1** | Share of pixels whose prediction is within a factor of 1.25 of the reference; higher is better. |\n"
            "| **Constant baseline** | Predicting the training split's median depth at every pixel; any useful model should beat it. |\n"
            "| **Backbone / neck / metric head** | The BEiT feature extractor (frozen), the DPT decoder that mixes scales (frozen), and the small head that turns features into metres (trained here). |\n"
            "| **Log-depth L1 loss** | Mean \\|log prediction − log reference\\|, the training objective; a scale error costs the same at 1 m and at 4 m. |\n"
            "| **Domain shift** | Test data that differ systematically from training data, here scenes laid out upside down compared with photographs. |\n"
            "| **Row correlation** | How the prediction's average depth per image row follows the reference's: +1 same slope, −1 opposite. |\n"
            "| **Adapter** | The SafeTensors file of the trained metric head, bound to the exact pinned base by digest and reloaded onto it. |\n"
            "| **Isolated environment** | A separate Python built from hash-locked pins, in which every learner cell runs. |\n\n"
            "</details>"
        ),
    ],
    "prerequisites": [
        "- **Runtime:** a fresh **Linux x86_64** runtime with an NVIDIA T4-class GPU or better (Colab or Kaggle T4). CUDA is required, and Section 1 stops on a CPU runtime before anything large is downloaded. The kernel's own Python version does not matter: Section 1 builds a separate environment with CPython 3.12.12 from the hash-locked pins and every later cell runs there. The pinned 1.38 GB checkpoint is acquired and digest-verified automatically. Timing: the previous version of this notebook ran in 88.1 s on a Kaggle T4 once installed (18 September 2026), after a first pass of 177.4 s that stopped for a restart in the old install cell; the previous blob of this version took 144.7 s on a Colab T4 (2026-10-10).",
        "- **Knowledge:** Python, metric depth, train/validation separation, AbsRel, δ1, and adapter/base dependencies.",
        "- **Overlap with the training data:** the ZoeDepth NYU+KITTI checkpoint was trained on NYU Depth v2 and KITTI after relative-depth "
        "pretraining [1] (model card). The default scenes are generated in code when the notebook runs, so they cannot be in any training set "
        "(verified by construction). A BYOD set of indoor or driving photographs could overlap NYU Depth v2 or KITTI, which would make the "
        "pretrained baseline look better than it is.",
        "- **Data:** the default path generates 24 paired scenes. **Their depth layout is upside down compared with photographs** (near at the top, far at the bottom), which Section 4 explains and uses. Optional BYOD accepts a ZIP of 8–128 image/depth pairs: `images/<id>.png|jpg` and `depth/<id>.npy` (finite positive metres, same height and width as the image), at the archive root or inside one top-level folder, by `BYOD_ZIP_PATH` or one upload. Sparse sensor depth with zeros for missing returns is refused with a message saying so; fill or crop those pixels first. Do not upload confidential or restricted data unless you are authorized to use it in the hosted runtime.",
        "- **Credentials:** none. The pinned model is public.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Generate or upload paired RGB and metric-depth data · [Evaluation practice]\n\n"
                "The default 24 deterministic 128×96 scenes encode sloped surfaces plus foreground panels with known "
                "depth in metres. Records 0–17 train and 18–23 remain held out (the last six generated scenes; there is no shuffle).\n\n"
                "**Read this sample for what it is: its depth layout is upside down.** In each scene depth grows from about 0.9 m at "
                "the top row to about 3.9 m at the bottom (plus a left-to-right slope and a nearer panel). Photographs — and the "
                "NYU indoor and KITTI road scenes ZoeDepth's metric heads were fitted to — are the other way round: the floor or "
                "road is nearest at the bottom of the frame and distance grows towards the horizon. The pretrained model reads "
                "these scenes as photographs, so its prediction slopes the wrong way. The sample is kept as it is on purpose, "
                "because every recorded number below depends on it and because it is a clean **domain-shift lesson**: watch "
                "what a strong pretrained model, a constant and a short head-only fine-tune each do when the data break the "
                "model's assumptions. Section 8 shows one scene as a picture and spells the result out.\n\n"
                "Optional BYOD: set `USE_BYOD = True` and put a ZIP path in `BYOD_ZIP_PATH` (any Jupyter runtime), or leave it "
                "empty and upload exactly one ZIP in Colab. The ZIP holds 8–128 pairs, `images/<id>.png|jpg` and `depth/<id>.npy`, "
                "at the archive root or inside one top-level folder (as when a folder is compressed). Traversal is rejected before "
                "decoding, and depth with zeros or negative values for missing pixels is refused with a message naming that rule.\n\n"
                "**Predict before running:** if a model trained on photographs looks at a scene that is nearest at the top, will its "
                "predicted depth increase towards the bottom like the reference, or towards the top?"
            ),
            "code": (
                "import io\n"
                "import zipfile\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_ZIP_PATH = ''  # @param {{type:\"string\"}}\n\n"
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
                "        # ZOE-m1: a ZIP made by compressing a folder puts everything under one top-level folder; accept that layout.\n"
                "        tops = {{name.split('/', 1)[0] for name in normalized if '/' in name}}\n"
                "        if len(tops) == 1 and not any(name.startswith(('images/', 'depth/')) for name in normalized):\n"
                "            prefix = next(iter(tops)) + '/'\n"
                "            normalized = [name[len(prefix):] if name.startswith(prefix) else name for name in normalized]\n"
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
                "        if not images:\n"
                "            raise ValueError('BYOD ZIP has no images/<id>.png|jpg and depth/<id>.npy pairs at its root or inside one top-level folder')\n"
                "        if not 8 <= len(images) <= MAX_ADAPTATION_RECORDS:\n"
                "            raise ValueError(f'BYOD ZIP has {{len(images)}} image/depth pairs; the notebook needs 8..{{MAX_ADAPTATION_RECORDS}} (8 so the held-out split has at least 2)')\n"
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
                "            if np.any(~np.isfinite(depth)) or np.any(depth <= 0):\n"
                "                raise ValueError(f'{{key}}: depth has zero, negative or non-finite values (sparse sensor depth marks missing pixels with 0); fill or crop them, the loss trains on every pixel')\n"
                "            records.append({{'id': key, 'image': image.convert('RGB'), 'depth_m': depth}})\n"
                "        return records\n\n"
                "if USE_BYOD:\n"
                "    if BYOD_ZIP_PATH:\n"
                "        from pathlib import Path\n\n"
                "        dataset_records = records_from_zip(Path(BYOD_ZIP_PATH).read_bytes())\n"
                "    else:\n"
                "        from google.colab import files\n"
                "        uploaded = files.upload()\n"
                "        if len(uploaded) != 1:\n"
                "            raise RuntimeError(f'Upload exactly one ZIP (received {{len(uploaded)}}); or set BYOD_ZIP_PATH and run this cell again.')\n"
                "        dataset_records = records_from_zip(next(iter(uploaded.values())))\n"
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
            "md": "**What to notice (Section 4):** 24 records, 18 train and 6 held out.\n\n<details><summary>Check your reasoning</summary>Towards the top. A model trained on photographs has learned that the bottom of a frame is usually nearest, so it predicts the opposite slope to these references. In the Colab T4 run of blob `441111a9` (commit `13c283a`, 2026-10-10) Section 8 printed `row_correlation_pretrained` −0.841: the prediction's rows run against the reference's. Section 8 shows this as a picture.</details>\n\n## 5. Validate alignment and split integrity · [Evaluation practice]\n\nThe validator checks unique IDs, image ceilings, exact depth/image shape, finite positive metres, the 80 m ceiling, and content fingerprints. Cross-split IDs are forbidden.",
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
            "md": "## 6. Measure the pretrained baseline · [Evaluation practice]\n\nScore the untouched holdout with AbsRel and δ1. The six held-out scenes come from the same generator as the training scenes, with only the sine phase and panel offsets changed, so they are near-copies: this is a sample-sanity check of the pipeline, not an independent test. The trivial comparator predicts the training split's median depth at every pixel. The cell also keeps a copy of the pretrained metric head (`PRETRAINED_HEAD`), taken the first time it runs, and restores it before scoring, so the baseline is always the pretrained model's even if you re-run this cell after Section 7.\n\n**Predict before running:** will the pretrained model beat the constant on AbsRel? On δ1?",
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
                "import torch\n\n"
                "# ZOE-m2: the pretrained head is captured once from the verified snapshot and restored before every baseline.\n"
                "if 'PRETRAINED_HEAD' not in globals():\n"
                "    PRETRAINED_HEAD = {{name: value.detach().cpu().clone() for name, value in pipe.model.metric_head.state_dict().items()}}\n"
                "pipe.model.metric_head.load_state_dict(PRETRAINED_HEAD)\n"
                "training_median = train_manifest['depth_median_m']\n"
                "base_eval = score_records(pipe, val_records, training_median)\n"
                "example = val_records[0]\n"
                "pretrained_example = np.asarray(pipe.predict(example['image'])['depth'], dtype=np.float64)\n"
                "print(json.dumps(base_eval, indent=2))"
            ),
        },
        {
            "md": "**What to notice (Section 6):** the model's AbsRel and δ1 beside the constant's.\n\n<details><summary>Check your reasoning</summary>Barely, and no. The Colab T4 run of blob `441111a9` (commit `13c283a`, 2026-10-10) measured AbsRel 0.431 for the model against 0.453 for the constant, but δ1 0.211 against 0.258: a single number repeated at every pixel puts more pixels within 25 % of the truth than the pretrained model does. That is the inverted layout at work (Section 4), not a weak model.</details>\n\n## 7. Freeze the base and run bounded metric-head fine-tuning · [Concept]\n\nOnly `metric_head.*` is trainable. AdamW runs two epochs at batch size one with mean absolute log-depth error. The run must produce a non-zero weight delta. The cell first restores the pretrained head (`PRETRAINED_HEAD`), so re-running it after changing `EPOCHS` or `LEARNING_RATE` trains once from the pretrained model, never on top of an earlier run. The backbone and neck stay frozen, so the fine-tune can rescale depth but cannot change *where* the frozen features put near and far.\n\n**Predict before running:** the validation loss will fall a little. Will held-out AbsRel fall with it?",
            "code": (
                "EPOCHS = 2  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-5  # @param {{type:\"number\"}}\n\n"
                "if not torch.cuda.is_available():\n"
                "    raise RuntimeError('The canonical ZoeDepth E2E path requires a CUDA GPU')\n"
                "pipe.model.metric_head.load_state_dict(PRETRAINED_HEAD)  # ZOE-m2: start from the pretrained head every time\n"
                "parameter_counts = pipe.freeze_for_adaptation()\n"
                "history = pipe.finetune(train_records, val_records, epochs=EPOCHS, learning_rate=LEARNING_RATE, seed=42)\n"
                "print({{'parameters': parameter_counts, 'history': history, 'weight_delta_l2': pipe.adaptation_config['weight_delta_l2']}})"
            ),
        },
        {
            "md": "**What to notice (Section 7):** the parameter counts (1.76 M trainable of 345 M), two history rows and a small `weight_delta_l2`.\n\n<details><summary>Check your reasoning</summary>No. In the Colab T4 run of blob `441111a9` (commit `13c283a`, 2026-10-10) the validation log-L1 went from 0.4287 after epoch 1 to 0.4259 after epoch 2 (no pre-training validation loss is printed), while held-out AbsRel rose from 0.431 to 0.466 — worse than the constant's 0.453 — and δ1 stayed flat (0.211 → 0.212). The loss is a log-scale average that the head can lower by shifting the overall scale; AbsRel weighs near pixels heavily, and with the frozen features still reading the scenes upside down, rescaling helps the loss but hurts AbsRel. A falling training or validation loss is optimisation evidence, not task evidence.</details>\n\n## 8. Evaluate the adapted model, and look at one scene · [Evaluation practice]\n\nEvaluate the same untouched holdout against the training-median baseline. Lower AbsRel and higher δ1 are better, but generated-scene values remain sample-sanity. The cell then draws the first held-out scene as four panels — RGB, reference depth, pretrained prediction, adapted prediction (dark = near, light = far, one shared scale) — and prints an interpretation: model against constant before and after training, and whether the predicted depth increases down the image like the reference (`row_correlation`, from −1 to +1).",
            "code": (
                "adapted_eval = pipe.evaluate_adaptation(val_records)\n"
                "evaluation_report_e2e = {{'task': 'monocular-metric-depth-adaptation', 'verdict': 'sample-sanity', 'estimation': f'fixed {{dataset_kind}} held-out split', 'baseline_pretrained': base_eval, 'adapted': adapted_eval, 'weight_delta_l2': pipe.adaptation_config['weight_delta_l2']}}\n"
                "adapted_example = np.asarray(pipe.predict(example['image'])['depth'], dtype=np.float64)\n"
                "reference_example = np.asarray(example['depth_m'], dtype=np.float64)\n\n\n"
                "def row_correlation(depth):\n"
                "    # +1: depth increases down the image exactly as in the reference row profile; -1: the opposite.\n"
                "    return float(np.corrcoef(depth.mean(axis=1), reference_example.mean(axis=1))[0, 1])\n\n\n"
                "interpretation = {{\n"
                "    'pretrained_beats_constant_abs_rel': base_eval['abs_rel'] < base_eval['constant_median_baseline_abs_rel'],\n"
                "    'pretrained_beats_constant_delta1': base_eval['delta1'] > base_eval['constant_median_baseline_delta1'],\n"
                "    'adapted_beats_constant_abs_rel': adapted_eval['abs_rel'] < base_eval['constant_median_baseline_abs_rel'],\n"
                "    'abs_rel_change_after_training': round(adapted_eval['abs_rel'] - base_eval['abs_rel'], 4),\n"
                "    'delta1_change_after_training': round(adapted_eval['delta1'] - base_eval['delta1'], 4),\n"
                "    'row_correlation_pretrained': round(row_correlation(pretrained_example), 3),\n"
                "    'row_correlation_adapted': round(row_correlation(adapted_example), 3),\n"
                "}}\n"
                "evaluation_report_e2e['interpretation'] = interpretation\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(evaluation_report_e2e, handle, indent=2)\n"
                "print(json.dumps(evaluation_report_e2e, indent=2))\n"
                "if interpretation['row_correlation_pretrained'] < 0:\n"
                "    print('The pretrained prediction slopes the opposite way to the reference: these scenes are nearest at the top, photographs at the bottom.')\n"
                "if interpretation['abs_rel_change_after_training'] > 0:\n"
                "    print('Held-out AbsRel got worse after training: the frozen features still read the layout upside down, and the head can only rescale.')\n\n"
                "low, high = float(reference_example.min()), float(reference_example.max())\n\n\n"
                "def depth_panel(depth):\n"
                "    scaled = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)\n"
                "    return Image.fromarray((scaled * 255).astype(np.uint8)).convert('RGB')\n\n\n"
                "panels = [example['image'].convert('RGB'), depth_panel(reference_example), depth_panel(pretrained_example), depth_panel(adapted_example)]\n"
                "figure = Image.new('RGB', (4 * 132, 100), 'white')\n"
                "for position, panel in enumerate(panels):\n"
                "    figure.paste(panel.resize((128, 96)), (position * 132, 2))\n"
                "figure.save('outputs/{stem}_heldout_example.png')\n"
                "print('panels, left to right: RGB | reference depth | pretrained prediction | adapted prediction')\n"
                "figure"
            ),
        },
        {
            "md": "**What to notice (Section 8):** the interpretation block and the four panels.\n\n<details><summary>Check your reasoning</summary>In the reference panel the bottom is light (far); in both prediction panels the light band sits higher and the range is narrow (the recorded unseen prediction spans only 1.98–2.36 m, `depth_range_m`, while the generated reference depths span about 0.73–4.38 m, the `depth_min_m` / `depth_max_m` Section 4 prints), and `row_correlation_pretrained` is negative. Training changed the scale a little, not the slope. So the honest reading of this run is: the plumbing works, the pretrained model cannot read upside-down scenes, and two epochs on the head cannot fix that. On data laid out like photographs — or your own RGB-D data — the same cells measure a real adaptation.</details>\n\n## 9. Infer on an unseen generated image · [Concept]\n\nA newly generated record outside the train/validation index range exercises adapted serving and produces a metric-depth array.",
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
            "md": "## 10. Export and verify a fresh reload · [Engineering]\n\nExport the metric head as SafeTensors with a closed manifest. A fresh base verifies the artifact and must reproduce the unseen depth array within explicit tolerances.",
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
            "md": "## 11. Export provenance and terminal summary · [Engineering]\n\nBind dataset fingerprints, split sizes, hyperparameters, metrics, weight activity, base identity, runtime, artifact, and reload evidence without retaining source records.",
            "code": (
                "payload = {{'dataset': dataset_manifest, 'adaptation': pipe.adaptation_config, 'evaluation': evaluation_report_e2e, 'unseen': {{'id': unseen['id'], 'abs_rel': unseen_abs_rel, 'delta1': unseen_delta1, 'depth_file': 'outputs/{stem}_unseen_depth.npy'}}, 'artifact': reload_summary, 'notebook_source': NOTEBOOK_SOURCE, 'repository_revision': NOTEBOOK_SOURCE['repository_revision'], 'base_model': {{'id': MODEL_ID, 'revision': MODEL_REVISION}}, 'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device}}}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2)\n"
                "print({{'status': 'E2E COMPLETE (pipeline stages; read the Section 8 interpretation for what the numbers mean)', 'train_records': len(train_records), 'held_out_records': len(val_records), 'optimizer_steps': sum(item['optimizer_steps'] for item in history), 'weight_delta_l2': pipe.adaptation_config['weight_delta_l2'], 'reload': reload_summary['verification'], 'outputs': sorted(os.listdir('outputs'))}})"
            ),
        },
    ],
    "closing": (
        "## 12. Activity: change one thing — train longer · [Concept]\n\n"
        "Optional; **Predict → Change one thing → Run → Observe → Explain**. It changes nothing unless you do it.\n\n"
        "1. **Predict:** with `EPOCHS = 6` instead of 2, will held-out AbsRel move back below the constant's, or further above it? What will "
        "`row_correlation_adapted` do?\n"
        "2. **Change:** in Section 7 set `EPOCHS = 6`. Change nothing else.\n"
        "3. **Run:** select the Section 7 cell and choose *Runtime → Run after*. Section 7 restores the pretrained head first, so this is one "
        "clean six-epoch run, not six epochs on top of two. Sections 8–11 then overwrite the outputs and the adapter (download `outputs/` "
        "first if you want to keep them).\n"
        "4. **Observe:** compare the history, the interpretation block and the four panels with your first run.\n"
        "5. **Explain:** in one sentence, why can more training of the head alone not turn the predicted slope the right way?\n\n"
        "<details><summary>Check your reasoning</summary>This setting has not been recorded, so your run is the measurement. Expect the "
        "validation loss to keep falling a little and the row correlation to stay negative: the head maps frozen features to metres, and those "
        "features place near and far by where things sit in a photograph. More epochs can fit the overall scale and spread better, which may "
        "move AbsRel either way, but reordering the layout would need the backbone or neck to learn, or data laid out like photographs.</details>\n\n"
        "## Interpretation and limits\n\n"
        "The default run measures a strong pretrained model on scenes laid out upside down compared with its training data. In the recorded "
        "run it barely beat a constant on AbsRel (0.431 against 0.453), lost to it on δ1 (0.211 against 0.258), and two epochs of head-only "
        "training lowered the validation loss while making held-out AbsRel worse (0.466). Those numbers describe the mismatch between this "
        "sample and photographs, not the quality of ZoeDepth, and the Section 8 figure shows why. What the run does prove is the workflow: the "
        "notebook can validate aligned RGB/depth pairs, keep a held-out split, update only the declared metric head, score it against a trivial "
        "baseline, serialize the adapter, attach it to the exact pinned base, and reproduce inference after reload. Generated gradients and "
        "panels are not photographs or sensor depth. Real use requires rights-cleared RGB-D data from the target camera and domain, leakage-safe "
        "splits, a missing-depth policy, and calibration across depth ranges and scene types.\n\n"
        "Successful execution proves that the recorded repository revision can complete this bounded tutorial without the repository being reachable at runtime. It does **not** establish benchmark superiority or production fitness.\n\n"
        "## Troubleshooting\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1 stops with `This notebook needs an NVIDIA GPU exposed as CUDA` | a CPU runtime | Choose *Runtime → Change runtime type → T4 GPU* and *Runtime → Run all*. |\n"
        "| You want the isolated environment somewhere other than `dimer_isolated_env/` | Section 1 builds it in the working directory by default | Set the environment variable `DIMER_ISOLATED_ENV` to a path before running Section 1. |\n"
        "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a local Windows or macOS kernel, or an ARM machine | Use Google Colab or Kaggle; the lock holds manylinux x86_64 wheels. |\n"
        "| Section 1 fails while downloading, or `The pinned uv wheel failed its size/SHA-256 check` | a network failure, or an altered download | Run the install cell again; a repeated mismatch means the download is being altered — never edit the digest. |\n"
        "| `The isolated environment's Python process exited` | the worker ran out of memory | Restart the session and choose *Runtime → Run all*. |\n"
        "| A Hub download fails in Section 3, or `sha256 … != manifest` | a transient Hugging Face failure, or a changed file | Run Section 3 again; a digest mismatch is never loaded. |\n"
        "| `RuntimeError: Upload exactly one ZIP` in Section 4 | the dialog was cancelled, or several files were chosen | Run the cell again and choose one ZIP, or set `BYOD_ZIP_PATH`. |\n"
        "| `BYOD ZIP has no images/<id>… pairs` or `has N image/depth pairs` | wrong layout, or fewer than 8 pairs | Put `images/` and `depth/` at the root (or inside one folder) and supply 8–128 pairs with matching stems. |\n"
        "| `depth has zero, negative or non-finite values` | sparse sensor depth with zeros for missing returns | Fill or crop the missing pixels before zipping; the loss trains on every pixel. |\n"
        "| `exceeds the 80 m model ceiling` | depth stored in millimetres or beyond the model's range | Convert to metres; crop far regions. |\n"
        "| `reloaded adapter depth mismatch` in Section 10 | the adapter did not round-trip | Re-run from Section 7; do not ship the adapter if it repeats. |\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional. Fill in from your own run, one sentence each:\n\n"
        "1. On the held-out scenes the pretrained model scored AbsRel ___ and δ1 ___, against ___ and ___ for the constant.\n"
        "2. After training, AbsRel was ___ and δ1 ___; the validation loss went from ___ to ___.\n"
        "3. The four panels showed ___, and `row_correlation_pretrained` was ___, which means ___.\n"
        "4. With `EPOCHS = 6` (Section 12), ___.\n"
        "5. What I would need before trusting an adapted depth model on my own camera: ___.\n\n"
        "## References\n\n"
        "[1] S. F. Bhat, R. Birkl, D. Wofk, P. Wonka, and M. Müller, \"ZoeDepth: Zero-shot transfer by combining relative and metric depth,\" arXiv:2302.12288, 2023. [Online]. Available: https://arxiv.org/abs/2302.12288\n\n"
        "[2] Upstream model checkpoint. [Online]. Available: https://huggingface.co/{MODEL_ID}\n\n"
        "[3] K. Valcorza, \"zoedepth-metric-depth-pipeline,\" source repository; model card: https://github.com/kurtvalcorza/zoedepth-metric-depth-pipeline/blob/main/MODEL_CARD.md. [Online]. Available: https://github.com/kurtvalcorza/zoedepth-metric-depth-pipeline"
    ),
}
