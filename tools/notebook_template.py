"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "zoedepth_metric_depth_pipeline",
    "repo_name": "zoedepth-metric-depth-pipeline",
    "stem": "zoedepth_metric_depth",
    "notebook_name": "zoedepth_metric_depth_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "ZoeDepthMetricPipeline",
    "weights_key": "zoedepth-nyu-kitti",
    "runtime_imports": ["torch", "transformers"],
    "title": "ZoeDepth NYU+KITTI — DIMER monocular metric depth estimation tutorial (standalone)",
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
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-isl--org%2FZoeDepth-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/isl-org/ZoeDepth",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2302.12288-b31b1b.svg", "https://arxiv.org/abs/2302.12288"),
    ],
    "capability": "monocular metric depth estimation — one RGB image → a float32 depth map in metres at the input resolution — using the pinned `Intel/zoedepth-nyu-kitti` weights",
    "intro": (
        "At inference the ZoeDepth model (a BEiT-large DPT encoder–decoder for relative depth, 24 layers, hidden size "
        "1024, with two metric bin heads — NYU indoor, 0.001–10 m, and KITTI outdoor, 0.001–80 m — chosen per image by a "
        "latent domain classifier; about 345M parameters) turns one RGB image into a depth map in metres, which the carried "
        "module interpolates back to the input resolution through the pinned processor's post-processing; an optional "
        "horizontal-flip test-time augmentation (the upstream evaluation setting) averages two passes. **No adaptation "
        "occurs:** no training, fine-tuning, in-context conditioning, or preprocessing fitting happens in this notebook — "
        "the upstream checkpoint supplies the weights and processor, and the carried module adds snapshot verification, "
        "the input contract (side ceilings, aspect-ratio ceiling, a boolean flip flag), a fixed output contract (metres, "
        "float32, minimum/median/maximum), and the `abs_rel`, `delta1`, `validate_inputs` and `evaluation_report` "
        "helpers. The default sample is a flat cartoon room drawn in code with **no reference depth**, so the evaluation "
        "report is `not-measurable` by design (the fleet matrix scores this row only with a reference) and the printed "
        "ordering check — is the near box estimated nearer than the far cabinet? — is an observation, not a metric."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, draw a synthetic room (or upload your own photograph and, optionally, a metric "
        "reference depth map) and validate it into an input manifest, choose whether to use flip augmentation, run the "
        "supported task, read the depth map correctly (metres, but an estimate with no confidence and a scale that depends "
        "on the model recognising the scene), exercise an optional BYOD path, produce an evaluation report that is "
        "`sample-sanity` with `abs_rel` and `delta1` only when a reference exists and `not-measurable` otherwise, and export "
        "the depth array, a preview and provenance."
    ),
    "exclusions": (
        "Relative or affine-invariant depth (this checkpoint claims metres; for scale-free depth see the sibling Depth "
        "Anything pipeline), camera intrinsics or point-cloud reconstruction (a depth map is not a 3D model without them), "
        "video or multi-view consistency, batch throughput, evaluation on NYU Depth v2 or KITTI (not bundled; only a drawn "
        "room is run here, and it is scored only if you bring a reference), and any training. The model was fine-tuned on "
        "indoor NYU and outdoor KITTI photographs; flat drawings, documents, medical, aerial and underwater imagery and "
        "unusual cameras are outside what this notebook measures, and a smooth-looking depth map carries no signal."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate: the repository's model card records 5.3 s to load and 1.5 s per 640×480 image (2.7 s with flip augmentation) in the Windows venv (Intel Core Ultra 9 275HX). The pinned `torch==2.14.0` install and the 1.38 GB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python, NumPy and PIL; what metric depth in metres means and why a monocular estimate of it depends on the model guessing the scene scale; what absolute relative error and δ1 measure and why one drawn image is not a benchmark.",
        "- **Data:** the default sample is a deterministic 640×480 cartoon room drawn in code with Pillow (a wall, a floor, a far cabinet on the wall, a near box on the floor and a window; no text rendering, so its digest is stable across Pillow builds) with **no reference depth**, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image decodable by Pillow (PNG/JPEG/WebP and similar), any colour mode, sides between 32 and 4096 px, aspect ratio at most 4:1, plus optionally a `.npy` float array of metric depth in metres with the same height and width (non-positive pixels are ignored). Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Draw the synthetic room or optional BYOD\n\n"
                "The default sample is **synthetic**: a flat cartoon room — a grey wall, a brown floor meeting it at a "
                "visible edge, a blue cabinet high on the wall, a red box low on the floor and a pale window — is drawn "
                "with Pillow at 640×480, the same drawing the repository's smoke run used. It has **no reference depth**: a "
                "drawing has no metres in it, and the notebook does not invent any, so the evaluation report will be "
                "`not-measurable` and the only check is an ordering observation (the box, drawn low and large, should come "
                "out nearer than the cabinet, drawn high and small). The image digest is printed for the record. BYOD is "
                "optional and disabled by default; when enabled, upload one photograph and, if you have one, a `.npy` depth "
                "map in metres with the same height and width — then the report becomes `sample-sanity` with `abs_rel` and "
                "`delta1`.\n\n"
                "Flip augmentation is a **caller-owned request parameter**: `flip_augmentation` runs the image and its "
                "mirror and averages the two depth maps (the upstream evaluation setting; twice the cost, and the smoke run "
                "saw a mean difference of 0.024 m on this room). Nothing is validated in this cell — the next section hands "
                "the image to the pipeline's own validation stage, which is the only checker. Look for a dictionary naming "
                "the sample kind, the image size and digest, the flip flag and whether a reference exists."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "flip_augmentation = False  # @param {{type:\"boolean\"}}\n\n\n"
                "def synthetic_room(width=640, height=480):\n"
                "    \"\"\"A flat cartoon room drawn with Pillow (no text): wall, floor, far cabinet, near box, window.\"\"\"\n"
                "    image = Image.new('RGB', (width, height), (200, 200, 190))  # wall\n"
                "    d = ImageDraw.Draw(image)\n"
                "    d.rectangle([0, 300, width, height], fill=(140, 110, 80))  # floor\n"
                "    d.polygon([(0, 300), (width, 300), (width, 320), (0, 320)], fill=(120, 95, 70))  # skirting edge\n"
                "    d.rectangle([80, 120, 200, 260], fill=(90, 120, 200))  # far cabinet, high on the wall\n"
                "    d.rectangle([380, 250, 560, 420], fill=(200, 60, 60))  # near box, low on the floor\n"
                "    d.rectangle([390, 260, 550, 300], fill=(230, 100, 100))  # box lid\n"
                "    d.ellipse([250, 60, 330, 140], fill=(255, 240, 150))  # window / light\n"
                "    probes = {{'near box': (470, 340), 'far cabinet': (140, 190), 'floor near the camera': (320, 470), 'wall top': (320, 30)}}\n"
                "    return image, probes\n\n\n"
                "reference_depth = None\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(name for name in uploaded if not name.lower().endswith('.npy'))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    depth_files = [name for name in uploaded if name.lower().endswith('.npy')]\n"
                "    if depth_files:\n"
                "        reference_depth = np.load(io.BytesIO(uploaded[depth_files[0]])).astype(np.float64)  # metres, H x W\n"
                "    probes = {{}}\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic drawing: no randomness and no text rendering, so no seed is needed and the digest is stable.\n"
                "    image, probes = synthetic_room()\n"
                "    image_name = 'synthetic_room_640x480.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256, 'flip_augmentation': flip_augmentation, 'has_reference_depth': reference_depth is not None}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `predict` applies — "
                "image type, sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px, aspect ratio at most `MAX_ASPECT_RATIO`, and a "
                "boolean flip flag — and returns an **input manifest** naming the schema (including the 384×512 aspect-preserving "
                "resize, the padding and the post-processing), the input's observed mode, size and aspect ratio, the flip "
                "flag and the verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To show what rejection "
                "looks like, the cell also validates a 5:1 panorama and records the pipeline's own error message as a finding. "
                "Inside the pipeline the image is converted to RGB and resized to fit 384×512 with its aspect ratio kept; "
                "nothing else is dropped or altered. The pipeline cannot tell whether the image is a photograph, which "
                "camera took it, or whether the scene is indoors or outdoors — the model guesses the last from the image, and "
                "that guess sets the metres."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_ASPECT_RATIO': MAX_ASPECT_RATIO, 'DEPTH_KIND': DEPTH_KIND, 'DEPTH_UNIT': DEPTH_UNIT, 'DEPTH_RANGES_M': DEPTH_RANGES_M, 'FLIP_AUGMENTATION': FLIP_AUGMENTATION, 'DELTA_THRESHOLD': DELTA_THRESHOLD}}}})\n"
                "input_manifest = validate_inputs(image, flip_augmentation=flip_augmentation, names=[image_name])\n"
                "# Demonstrate rejection on a request that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(Image.new('RGB', (2000, 400)))\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'panorama-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Estimate depth and read the output correctly\n\n"
                "`predict` returns `depth` (a float32 H×W array in metres at the input resolution; larger is farther), "
                "`depth_kind` and `depth_unit`, the minimum, median and maximum, the flip flag, the image size and the model "
                "identity. **The metres are an estimate, not a measurement**: the model picks its indoor or outdoor head from "
                "the image itself, the scale follows that choice and the camera it assumes, no confidence is attached, and a "
                "blank image still yields a depth map. The forward pass is deterministic on a fixed device and dtype; CUDA "
                "kernels can shift values slightly, so GPU and CPU maps need not match to the millimetre. Each call costs one "
                "BEiT-large pass at up to 384×512 (about 1.5 s on the reference CPU, 2.7 s with flip). As recorded in the "
                "model card, the repository's CPU smoke on this same room estimated 1.59–1.90 m with the near box at 1.69 m "
                "and the far cabinet at 1.87 m — the right ordering — and estimated 1.43–2.33 m for a blank white image and "
                "1.39–1.82 m for uniform noise: the model always produces plausible-looking metres. The cell prints the depth "
                "at the drawn probe points as an observation."
            ),
            "code": (
                "import time\n\n"
                "t0 = time.time()\n"
                "result = pipe.predict(image, flip_augmentation=flip_augmentation)\n"
                "elapsed = round(time.time() - t0, 2)\n"
                "depth = result['depth']\n"
                "print({{'device': pipe.device, 'seconds': elapsed, 'shape': depth.shape, 'dtype': str(depth.dtype), 'unit': result['depth_unit'], 'min_m': round(result['depth_min'], 3), 'median_m': round(result['depth_median'], 3), 'max_m': round(result['depth_max'], 3), 'flip_augmentation': result['flip_augmentation']}})\n"
                "probe_depths = {{name: round(float(depth[y, x]), 3) for name, (x, y) in probes.items()}}\n"
                "if probe_depths:\n"
                "    print('depth at drawn probes (m):', probe_depths)\n"
                "    print('near box nearer than far cabinet:', probe_depths['near box'] < probe_depths['far cabinet'])"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No accuracy is "
                "reported by default: metric depth needs a reference depth map in metres from a sensor, LiDAR, stereo or an "
                "RGB-D benchmark, and this repository ships none (NYU Depth v2 and KITTI are not bundled). When a reference is "
                "supplied the report carries `abs_rel` (mean |pred − ref| / ref) and `delta1` (the fraction of pixels whose "
                "ratio is within 1.25), both computed **without any scale or shift alignment** because the model claims "
                "metres, plus a constant-median-depth baseline, with the verdict `sample-sanity`. On the synthetic path no "
                "reference exists — a drawing has no metres — so the verdict is `not-measurable` by design and the report "
                "states what would make the task measurable; the probe ordering from the previous section is attached to the "
                "report file under `observations` for the record. The report is written to "
                "`outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, reference_depth, sample_kind=sample_kind)\n"
                "report['observations'] = {{'probe_depths_m': probe_depths}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k not in ('metrics', 'baselines', 'observations')}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    print(f\"{{metric['id']:10}} {{metric['value']:.4f}}  ({{metric['estimation']}})\")\n"
                "for baseline in report['baselines']:\n"
                "    print(f\"baseline {{baseline['id']}}: abs_rel {{baseline['abs_rel']:.4f}}, delta1 {{baseline['delta1']:.4f}}  ({{baseline['note']}})\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No metric reference depth exists for this image, so nothing is scored; the metres are an unverified estimate.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the depth statistics, the flip flag, the evaluation report with the probe "
                "observations, the input manifest, the sample identity and digest, the notebook's source (repository, revision, "
                "embedded module digest, generator), the model identifier, the immutable model revision, the model licence, "
                "and the runtime identity (Python, `torch`, `transformers`, device); the depth map itself is written as a "
                "float32 `.npy` in metres (the array intended for downstream use), because arrays do not belong in JSON. A "
                "side-by-side preview PNG shows the image next to the depth map rendered on a fixed grey ramp between its own "
                "minimum and maximum (a supplement to, not a replacement for, the array — the ramp is per-image and says "
                "nothing about absolute scale). No credentials are recorded."
            ),
            "code": (
                "np.save('outputs/{stem}_depth.npy', depth)\n"
                "lo, hi = float(depth.min()), float(depth.max())\n"
                "ramp = ((depth - lo) / max(hi - lo, 1e-6) * 255.0).round().astype(np.uint8)\n"
                "depth_preview = Image.fromarray(ramp).convert('RGB')\n"
                "preview = Image.new('RGB', (image.width * 2, image.height), 'white')\n"
                "preview.paste(image.convert('RGB'), (0, 0))\n"
                "preview.paste(depth_preview, (image.width, 0))\n"
                "preview.save('outputs/{stem}_preview.png')\n"
                "payload = {{\n"
                "    'prediction': {{k: v for k, v in result.items() if k != 'depth'}},\n"
                "    'depth_file': 'outputs/{stem}_depth.npy',\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'has_reference_depth': reference_depth is not None}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The depth map is the model's estimate of metres for an image whose camera and scene it has never seen; nothing in "
        "the output scores that estimate, the scale follows the model's own indoor/outdoor guess, and it produces metres for "
        "any input — a blank white image came out at 1.4–2.3 m in the smoke run. On the drawn room the evaluation report is "
        "`not-measurable` by design and the probe ordering (near box 1.69 m before far cabinet 1.87 m in the smoke run) is an "
        "observation about a flat cartoon, not evidence of metric accuracy; it says nothing about photographs, cameras with "
        "unusual focal lengths, outdoor scale, reflective or transparent surfaces, thin structures, or anything beyond 10 m "
        "indoors and 80 m outdoors, and a BYOD result without a reference is a single-image observation with the same "
        "verdict. **A smooth depth map is not a correct one**: bring a metric reference (a sensor, LiDAR, stereo or an RGB-D "
        "benchmark) and read `abs_rel` and `delta1` against the constant-median baseline before trusting any number. The "
        "pipeline provides no camera intrinsics, no point cloud, no confidence, no benchmark evaluation and no training "
        "capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** set `flip_augmentation` to `True` and compare the two maps; move the red box up the wall in "
        "`synthetic_room` and watch its estimated depth grow; enable `USE_BYOD` with an indoor photograph, then an outdoor one, "
        "and compare the ranges the model chose; if you own an RGB-D capture, upload its depth as a `.npy` in metres and see "
        "the verdict switch to `sample-sanity`.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/zoedepth-metric-depth-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/zoedepth-metric-depth-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/zoedepth-metric-depth-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/isl-org/ZoeDepth\n"
        "- ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth (Bhat et al., 2023): https://arxiv.org/abs/2302.12288\n"
        "- Vision Transformers for Dense Prediction — DPT (Ranftl, Bochkovskiy, Koltun, 2021): https://arxiv.org/abs/2103.13413\n"
        "- Indoor Segmentation and Support Inference from RGBD Images — NYU Depth v2 (Silberman et al., 2012): https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html"
    ),
}
