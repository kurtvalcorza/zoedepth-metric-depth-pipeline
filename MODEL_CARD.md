---
license: mit
model_card_spec: "1.1"
pipeline_tag: depth-estimation
task: "Depth Estimation - Metric"
base_model: Intel/zoedepth-nyu-kitti
date_published: "2024-04-30"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt` 2024-04-30T20:22:35Z, https://huggingface.co/api/models/Intel/zoedepth-nyu-kitti — the Transformers-format conversion); the ZoeDepth paper and original checkpoints are from 2023-02 (arXiv:2302.12288), and the pinned revision is the Hub's `main` as of 2026-09-14"
---

# ZoeDepth NYU+KITTI — Monocular Metric Depth Estimation (Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Intel%2Fzoedepth--nyu--kitti-ffcc4d?style=flat)](https://huggingface.co/Intel/zoedepth-nyu-kitti)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-isl--org%2FZoeDepth-181717?style=flat&logo=github&logoColor=white)](https://github.com/isl-org/ZoeDepth)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2302.12288-b31b1b.svg)](https://arxiv.org/abs/2302.12288)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — stage and verify the pinned upstream revision in a fresh runtime, validate an input, run the task, and inspect and export the outputs:

- **End-to-End Fine-Tuning Tutorial**:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/zoedepth-metric-depth-pipeline/blob/main/tutorials/zoedepth_metric_depth_colab.ipynb) [`zoedepth_metric_depth_colab.ipynb`](https://github.com/kurtvalcorza/zoedepth-metric-depth-pipeline/blob/main/tutorials/zoedepth_metric_depth_colab.ipynb)  
  *A standalone E2E path over deterministic RGB/metric-depth scenes: pinned pretrained and constant-median baselines, bounded metric-head gradient adaptation, held-out `abs_rel` and `delta1`, SafeTensors export, unseen inference, and fresh-model reload.*

---

#### Description

`Intel/zoedepth-nyu-kitti` is the Transformers-format release of the ZoeDepth model fine-tuned jointly on NYU Depth v2 and KITTI (the "ZoeD-M12-NK" configuration), from "ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth" (Bhat, Birkl, Wofk, Wonka, Müller, arXiv:2302.12288), converted by the Hugging Face team and pinned here to revision `f364d4c7936e91f465abba182208dd68142bf0ca` (the Hub's `main` on 2026-09-14). The snapshot declares a BEiT-large encoder with a DPT fusion decoder and two metric bins heads selected by a latent domain classifier. At inference the processor encodes the image, the model emits depth in metres, and post-processing restores the input size. This repository adds verified packaging and an optional bounded adaptation route: `freeze_for_adaptation` leaves only `metric_head.*` trainable; `finetune` uses caller-owned RGB/metric-depth records; `evaluate_adaptation` reports held-out `abs_rel` and `delta1` beside a constant training-median baseline; and `save_artifact`/`from_artifact` export and reload only that surface through a digest-bound SafeTensors artifact.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is monocular metric depth estimation: input one RGB image (`PIL.Image.Image`, any mode, converted to RGB) and an optional flip flag; output a float32 depth map in metres at the input resolution with its minimum, median and maximum. The optional adaptation task accepts 2–128 uniquely identified RGB records with exact-size finite positive metric-depth arrays capped at 80 m, trains only the metric head, and requires a separate held-out split for evaluation. Within DIMER the pipeline is an inference component and bounded adaptation baseline for metric depth, not a rangefinder and not a certified sensor substitute.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision and robotics developers, and researchers integrating single-image depth into research prototypes, internal tooling, or the DIMER workbench. A user is expected to understand that the metres are an *estimate with no confidence* whose scale follows the model's own guess of the scene domain (indoor or outdoor head) and of the camera it assumes, that a monocular model cannot know the true scale of an unfamiliar camera or scene and will still answer, that the model was fine-tuned on NYU indoor and KITTI driving photographs so drawings, other cameras, aerial, medical and underwater imagery are distribution shifts (the tutorial's cartoon receives plausible metres anyway), that flip augmentation doubles the cost for a small change, that the forward pass is deterministic on a fixed device but GPU and CPU maps can differ slightly, and that accuracy can only be measured against a metric reference they capture. Users who need relative depth, camera intrinsics, point clouds, video consistency or batch throughput are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** no confidence or uncertainty map, no abstention (a blank image yields metres), no camera intrinsics, no point cloud or 3D reconstruction, no video or multi-view consistency, no relative-depth output (see the sibling Depth Anything pipeline), and no batching across images.
2. **Input boundary:** `predict` rejects non-PIL images (`TypeError`), sides below `MIN_IMAGE_SIDE = 32` px or above `MAX_IMAGE_SIDE = 4096` px, aspect ratios above `MAX_ASPECT_RATIO = 4.0`, and a non-boolean flip flag (`ValueError`/`TypeError`). Every image is resized to fit 384×512 with its aspect ratio kept, so a 4096×4096 input carries no more information than a 384×384 one and fine structures are lost; wide panoramas are refused because the working width grows with the aspect ratio.
3. **Input boundary:** the metric heads were fine-tuned on NYU Depth v2 (Kinect indoor scenes, 640×480, up to 10 m) and KITTI (a car-mounted camera with LiDAR, up to 80 m). Photographs from other focal lengths and viewpoints, close-ups, macro, aerial, medical, underwater or rendered imagery, scenes beyond either range, and non-photographic inputs (the tutorial's drawing) fall outside what the upstream authors evaluated and what this repository measured; the metres on them are undefined, not merely degraded (see §Risks and harms).
4. **Decision boundary:** not for autonomous decisions that act on the metres — collision avoidance, navigation, safety distances, dimensional measurement for construction, medical or legal purposes, insurance or forensic claims — without a real range measurement, and a locally measured `abs_rel`/`delta1` on the deployment's own camera and scenes.

#### Factors

###### Groups

This pipeline is not human-centric by design: it estimates depth for a scene and never classifies, identifies or scores people. The fine-tuning data (NYU Depth v2 apartments and offices in New York; KITTI streets in Karlsruhe) contains no evaluation groups in the demographic sense, and neither the upstream authors nor this repository audited it for anything of the kind. What does vary is the scene population: NYU and KITTI are North-American indoor and German road scenes captured with two specific rigs, so homes, workplaces, streets, vehicles and vegetation from other regions, other cameras and phones, night scenes, rain and fog, and cluttered or reflective interiors are the groups whose depth accuracy is unknown, not known to be equal. Where images contain people, the depth to a person is estimated like any other surface, with no per-person audit; an operator deploying on their own camera and environment is responsible for a metric evaluation stratified by scene type and lighting before relying on the output.

###### Instrumentation

The upstream "instruments" are a Microsoft Kinect (NYU: 640×480 RGB with structured-light depth, holes in-painted) and a stereo/LiDAR rig on a car (KITTI: 1242×375 RGB with sparse Velodyne depth), plus the twelve MiDaS relative-depth sources behind the backbone. Inference images arrive from whatever produced them — a phone, a DSLR, a webcam, a drawing library — and focal length, sensor size, resolution, exposure, lens distortion and compression all change what a pixel's appearance says about its distance; the model has no access to the intrinsics and guesses the scale from content alone, while the fixed fit-to-384×512 resize discards detail regardless of the source. The pipeline validates type, size and aspect ratio only; it cannot detect a non-photographic image, an unusual camera, or a scene outside both heads' ranges. The synthetic tutorial room (flat Pillow shapes, no perspective cues beyond placement, no texture) is itself a rendering instrument unlike any NYU or KITTI frame, which is why its metres are an observation and not a measurement.

###### Environment

The declared notebook environment is Python 3.12 with `torch==2.14.0`, `transformers==4.57.6`, and the exact pins in `pyproject.toml`. Historical inference timings were measured on CPU. The current E2E carrier also completed a local CUDA pre-flight on an RTX 5070 Ti Laptop GPU, but that run used Python 3.14.2, torch 2.11.0+cu128, and transformers 5.8.1, so it is compatibility evidence rather than release qualification. Cost is dominated by the BEiT-large pass at the processor's working resolution and grows with aspect ratio, not source pixel count. The model assumes a camera and scene like NYU or KITTI; other cameras, scene types, and non-photographic inputs are unmeasured, and the pipeline reports no distribution-shift signal.

#### Metrics

###### Performance Measures

The depth map is an estimate in metres with no confidence; its minimum, median and maximum do not measure correctness. `abs_rel(pred, ref)` and `delta1(pred, ref)` score predictions without scale or shift alignment. `evaluate_adaptation(records, baseline_depth_m=...)` applies both to a validated labelled set and scores a constant training-median baseline beside the model. The E2E tutorial reports the pinned pretrained and adapted results on the same held-out split with the verdict `sample-sanity`. Synthetic results are workflow evidence, not an NYU/KITTI benchmark; deployment claims require representative sensor references that were not used for optimization.

###### Decision thresholds

No score threshold exists in the model: it emits a depth value for every pixel and nothing is filtered or abstained; the only threshold in the repository is `DELTA_THRESHOLD = 1.25` inside the `delta1` helper, the depth-estimation convention, not a value tuned here. The request parameter is **flip augmentation** `flip_augmentation`, default `FLIP_AUGMENTATION = False` (a repository choice for cost; the upstream evaluation used flipping): when true the image and its mirror are both predicted and averaged, doubling the cost — on the drawn room the two maps differed by 0.024 m on average and the median moved from 1.82 m to 1.85 m. The model's own decision — which metric head to use — is made internally from the image by its domain classifier and is not exposed or controllable; a deployment that knows its scenes are indoors cannot force the NYU head through this package. A deployment owns the flip choice and decides how a depth value is verified before it is used.

###### Approaches to uncertainty and variability

This repository reports no central metric value and therefore no dispersion: the smoke run records timings and the depth statistics of one drawn room, not accuracy. Run-to-run variability comes only from floating-point kernel selection across CPU builds and accelerators; there is no sampling and no seed to set, so a fixed input on fixed hardware is repeatable, but CPU and CUDA maps can differ in the low decimals; the drawn room uses no text rendering, so its bytes do not depend on the Pillow build. On the drawn room the model estimated 1.59–1.90 m (median 1.82 m) with the near box at 1.69 m, the far cabinet at 1.87 m, the floor nearest the camera at 1.68 m and the wall top at 1.86 m — the drawn ordering, one observation on one flat cartoon whose true metres do not exist; on a blank white image it estimated 1.43–2.33 m (median 1.93 m) and on uniform noise 1.39–1.82 m (median 1.68 m), which is what an estimator with no abstention looks like — every input receives plausible indoor metres. A caller who needs an accuracy estimate must capture metric references and compute `abs_rel`/`delta1` over many images or bootstrap resamples themselves; a caller who needs a per-pixel uncertainty has none from this model.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The upstream paper describes pretraining the relative-depth backbone on the MiDaS mixture of twelve datasets (including web-scraped stereo photographs and 3D movies whose licensing the MiDaS authors document per source) and fine-tuning the metric heads on NYU Depth v2 (indoor scenes recorded by the dataset authors) and KITTI (street scenes recorded from a car in Karlsruhe, in which pedestrians, licence plates and house fronts are visible); personal data in the training corpora is therefore present, and the web-sourced portions were not audited. This repository distributes code, tests, and documentation; it does not distribute the 1,380,374,404-byte `model.safetensors`, which is staged locally under `weights/zoedepth-nyu-kitti/` and git-ignored, and it ships no photographs — the tutorial room is drawn in code. The operator must audit the images they submit for personal, proprietary, or otherwise restricted content; the pipeline performs no such check and will estimate the distance to a person as readily as to a wall.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Foreseeable but unintended sensitive uses — obstacle avoidance for robots, drones or assistive devices, distance estimates in driving or industrial safety, dimensional measurement for construction, medical or forensic purposes, surveillance range-finding — would be admissible only with a real range sensor in the loop (the model produces metres for any image and gives no signal when they are wrong), a locally measured `abs_rel`/`delta1` on the deployment's own camera and scenes against a constant-depth baseline, a documented flip and resolution policy, and whatever regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks all 4 listed files' byte sizes and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True`, always passes `trust_remote_code=False`, the backbone ships in the same SafeTensors file (`use_pretrained_backbone` false), and the smoke run loaded and predicted with `HF_HUB_OFFLINE=1`. No pickle checkpoint exists upstream at this revision. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused; the import-boundary tests assert that a missing or tampered snapshot is refused before `torch` or `transformers` is imported; another asserts the package's `DEPTH_RANGES_M` match the committed `config.json`.
- **Input integrity:** the public `validate_inputs(images, *, flip_augmentation)` stage applies exactly the checks `predict` applies (the same `validate_image` and flip check) and returns an input manifest recording the schema, the ceilings, each input's observed mode, size and aspect ratio, the flip flag and the verdict; non-PIL inputs, sides outside 32–4096 px, aspect ratios above 4 and non-boolean flags are rejected; `predict` raises when the backend returns a map of the wrong shape or non-finite or non-positive values; `abs_rel`/`delta1` ignore non-positive reference pixels and refuse mismatched shapes.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; a deterministic forward pass; the pinned processor's own post-processing for un-padding and resizing; every result carries `model_id`, `model_revision`, the flip flag, the unit, the image size and the depth statistics.
- **Refusals:** no batching, no download without the explicit flag, no Hub access at inference time, no pickle deserialisation, no exposure of the internal head choice, no confidence invented, and no attempt to guess whether the image is a photograph — that judgement is the operator's.
- **Bounded adaptation:** training rejects more than 128 records, freezes every parameter outside `metric_head.*`, records trainable/frozen counts and weight movement, and exports only SafeTensors plus a closed digest-bound manifest. Dataset balancing and deployment-specific fairness mitigation remain the operator's responsibility.

###### Risks and harms

- **Metres for anything:** the model has no abstention — a blank image received 1.4–2.3 m and noise 1.4–1.8 m in the smoke run — so a non-photographic image, an unfamiliar camera or a scene outside both heads' ranges produces plausible-looking metres with no signal; downstream consumers that trust them (navigation, measurement, reconstruction) inherit the error silently.
- **Scale from a guess:** the metric scale follows the internal indoor/outdoor decision and the assumed camera; a misrouted scene or a wide-angle phone lens can be off by a large factor while the relative layout still looks right.
- **Smooth, coarse maps:** the 384×512 working resolution blurs edges and thin structures; a map used for measurement or collision margins carries that error.
- **Automation bias:** a clean depth render invites trust that an uncertainty-free estimate has not earned.
- **Depth to people:** distances to people are estimated like any surface and can be surfaced or acted on; such uses fall under §Use cases.
- **Bias amplification:** any scene population NYU and KITTI under-represent (other regions, cameras, weather, interiors) is reproduced as uneven accuracy, undetected because no per-scene evaluation exists.
- **Resource use:** a 1.38 GB model and ~1.5 s per image on the reference CPU (2.7 s with flip); an image-heavy workload scales linearly, and the CUDA path was not measured.

###### Use cases

Prohibited even where the model would work: using the metres for collision avoidance, navigation, safety distances or any physical actuation without a real range sensor; dimensional measurement for construction, medical, insurance, forensic or legal purposes presented as measured; range-finding or tracking of people for surveillance; processing images the operator has no right to process, including intimate imagery and licence-restricted material; presenting estimated depth as measured depth or as evidence; and any use that violates the upstream MIT licence terms, the DIMER deployment terms, or the consent and data-protection obligations attached to the images processed. Autonomous high-consequence actions triggered by unreviewed depth estimates are prohibited by the intended-use contract above.

## Immutable provenance

- Model: `Intel/zoedepth-nyu-kitti`
- Revision: `f364d4c7936e91f465abba182208dd68142bf0ca`
- Snapshot manifest: `weights/zoedepth-nyu-kitti/dimer-base-manifest.json`, 4 files (no `pytorch_model.bin` upstream at this revision), `totalBytes` 1380379860
- `model.safetensors` SHA-256: `c5494fa0938f18d71e215e245472470c3aefebd7b434abd89750e5ae4008e2dc` (1,380,374,404 bytes, float32)
- `config.json` SHA-256: `58494c160c520023c4d5bdeebb3b2d035e37e48cc81225580f49fcb06e175913` (2,225 bytes; `ZoeDepthForDepthEstimation`, BEiT-large backbone, `bin_configurations` nyu 0.001–10 m / kitti 0.001–80 m)
- `preprocessor_config.json` SHA-256: `0b64d8edc980d7b7abb819085650c43da8b8183acbd4336ab5a9e0e9caf4648e` (723 bytes; `ZoeDepthImageProcessor`, 384×512 keep-aspect, multiples of 32, pad, mean/std 0.5)
- Weight format: SafeTensors; loader `ZoeDepthForDepthEstimation.from_pretrained(<dir>, local_files_only=True, trust_remote_code=False, dtype=float32)` with `ZoeDepthImageProcessor` from the same directory and its `post_process_depth_estimation(outputs, source_sizes=[(H, W)], outputs_flipped=…)`.

## Input/output contract

- `ZoeDepthMetricPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — stages missing manifest files (only with `allow_download=True`), verifies digests, loads; `device` defaults to `cuda:0` when visible, else `cpu`; float32 on both.
- `predict(image, *, flip_augmentation=False) -> dict` with keys `depth` (float32 `(H, W)` in metres, larger = farther), `depth_kind` (`"metric"`), `depth_unit` (`"metres"`), `depth_min`, `depth_median`, `depth_max`, `flip_augmentation`, `height`, `width`, `model_id`, `model_revision`.
- `abs_rel(pred, ref_depth) -> float`; `delta1(pred, ref_depth, *, threshold=1.25) -> float`; `validate_image(image) -> Image`.
- Ceilings and constants: `MIN_IMAGE_SIDE = 32`, `MAX_IMAGE_SIDE = 4096`, `MAX_ASPECT_RATIO = 4.0`, `DEPTH_KIND`, `DEPTH_UNIT`, `DEPTH_RANGES_M`, `FLIP_AUGMENTATION = False`, `DELTA_THRESHOLD = 1.25`, `INPUT_SCHEMA`.
- `validate_inputs(images, *, flip_augmentation, names) -> dict`; `evaluation_report(result, reference_depth=None, *, sample_kind) -> dict` where `reference_depth` is a metric depth map in metres at the image's resolution; `verify_snapshot(path=None) -> dict`; `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`; Python 3.12.
- Precision: float32; preprocessing resizes the image to fit 384×512 keeping the aspect ratio (sides rounded to multiples of 32), pads and normalises with mean/std 0.5 (`ZoeDepthImageProcessor`, snapshot defaults); the prediction is un-padded and interpolated back to the input size by the processor's post-processing; optional flip augmentation averages a mirrored pass.
- Measured 2026-09-14 in the Windows venv (`torch 2.14.0+cu130`) with `CUDA_VISIBLE_DEVICES=-1` and `HF_HUB_OFFLINE=1`, device `cpu`: `verify_snapshot` 0.72 s (4 files, 1.38 GB); load 5.33 s; `predict` on a synthetic 640×480 cartoon room (wall, floor, far cabinet, near box, window; drawn with Pillow) 1.47 s → 1.59–1.90 m, median 1.82 m; probes: near box 1.69 m, far cabinet 1.87 m, floor nearest the camera 1.68 m, wall top 1.86 m (near < far, the drawn ordering); with `flip_augmentation=True` 2.73 s, median 1.85 m, mean absolute difference 0.024 m; `evaluation_report` without a reference → `not-measurable`; blank 640×480 white image → 1.43–2.33 m (median 1.93 m); uniform noise → 1.39–1.82 m (median 1.68 m); 4096×4096 blank image 2.53 s at a 512×512 working resolution; a 1600×400 image works at 384×1408 and a 2000×400 one is refused by the aspect-ratio ceiling.
- Tutorial execution: `tutorials/zoedepth_metric_depth_colab.ipynb` ran top-to-bottom in a fresh local kernel (all 8 code cells, 133.2 s including the 1.38 GB staging, same depth statistics and probe ordering as the smoke run, report `not-measurable` by design); recorded in `docs/release-verification.md` as pre-flight, not supported-runtime evidence.
- Tests: `pytest -q -o addopts= tests` — offline, no weights required; `ruff check src tests tools` clean.
- Not executed: CUDA path, photographs (only a drawn room, blank images and noise), any `abs_rel`/`delta1` measurement against a real metric reference, outdoor scenes (the KITTI head), images of people.

## References

- Bhat, Birkl, Wofk, Wonka, Müller. ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth. 2023. https://arxiv.org/abs/2302.12288
- Ranftl, Bochkovskiy, Koltun. Vision Transformers for Dense Prediction (DPT). ICCV 2021. https://arxiv.org/abs/2103.13413
- Silberman, Hoiem, Kohli, Fergus. Indoor Segmentation and Support Inference from RGBD Images (NYU Depth v2). ECCV 2012. https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
- Geiger, Lenz, Urtasun. Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. CVPR 2012. https://www.cvlibs.net/datasets/kitti/
- Upstream code: https://github.com/isl-org/ZoeDepth
- Upstream card: https://huggingface.co/Intel/zoedepth-nyu-kitti
- Transformers `ZoeDepth` documentation: https://huggingface.co/docs/transformers/model_doc/zoedepth
