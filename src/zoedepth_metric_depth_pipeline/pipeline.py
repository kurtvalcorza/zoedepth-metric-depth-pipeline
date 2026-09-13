"""Monocular metric depth estimation with the pinned ``Intel/zoedepth-nyu-kitti`` checkpoint (ZoeDepth).

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the ZoeDepth architecture (a BEiT-large DPT backbone with metric bin heads
for the NYU and KITTI ranges) comes from the pinned ``transformers`` release, the weights are
SafeTensors, and no model-repository code is executed. The output is depth in metres — a metric
estimate with no confidence, not a measurement — at the caller's resolution.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_ID = "Intel/zoedepth-nyu-kitti"
MODEL_REVISION = "f364d4c7936e91f465abba182208dd68142bf0ca"
MODEL_LICENSE = "mit"
MODEL_KEY = "zoedepth-nyu-kitti"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Input ceilings. The ZoeDepth processor resizes the image to fit 384x512 while keeping the aspect
# ratio (sides rounded to multiples of 32) and pads, so the backbone cost grows with the aspect ratio,
# not the pixel count; the prediction is interpolated back to the caller's resolution.
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 32
MAX_ASPECT_RATIO = 4.0
DEPTH_KIND = "metric"
DEPTH_UNIT = "metres"
# The checkpoint's two metric heads (config.json bin_configurations): NYU indoor 0.001-10 m and KITTI
# outdoor 0.001-80 m; the model routes each image to one head by its own domain classifier.
DEPTH_RANGES_M = {"nyu": (0.001, 10.0), "kitti": (0.001, 80.0)}
# Optional horizontal-flip test-time augmentation (the upstream evaluation setting): two forward
# passes, predictions averaged. Off by default; a caller-owned request parameter.
FLIP_AUGMENTATION = False
# delta1 accuracy threshold (the depth-estimation convention): max(pred/ref, ref/pred) < 1.25.
DELTA_THRESHOLD = 1.25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _valid_mask(pred: np.ndarray, ref_depth: np.ndarray) -> np.ndarray:
    if pred.shape != ref_depth.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs ref {ref_depth.shape}")
    valid = np.isfinite(ref_depth) & (ref_depth > 0) & np.isfinite(pred) & (pred > 0)
    if valid.sum() < 2:
        raise ValueError("need at least 2 valid reference pixels (ref_depth > 0 and pred > 0)")
    return valid


def abs_rel(pred: np.ndarray, ref_depth: np.ndarray) -> float:
    """Absolute relative error ``mean(|pred - ref| / ref)`` of metric depth against metric reference depth.

    Both arrays are in metres at the same H x W; no scale or shift alignment is applied, because the
    model claims metric output — a scale error therefore shows up in the number, as it should.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref_depth = np.asarray(ref_depth, dtype=np.float64)
    valid = _valid_mask(pred, ref_depth)
    return float(np.mean(np.abs(pred[valid] - ref_depth[valid]) / ref_depth[valid]))


def delta1(pred: np.ndarray, ref_depth: np.ndarray, *, threshold: float = DELTA_THRESHOLD) -> float:
    """Fraction of valid pixels whose ratio ``max(pred/ref, ref/pred)`` is below ``threshold`` (1.25)."""
    pred = np.asarray(pred, dtype=np.float64)
    ref_depth = np.asarray(ref_depth, dtype=np.float64)
    valid = _valid_mask(pred, ref_depth)
    ratio = np.maximum(pred[valid] / ref_depth[valid], ref_depth[valid] / pred[valid])
    return float(np.mean(ratio < threshold))


def validate_image(image: Any) -> Image.Image:
    """Type- and size-check a caller image and return it as RGB."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    short, long = min(width, height), max(width, height)
    if short < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {short} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if long > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {long} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    if long / short > MAX_ASPECT_RATIO:
        raise ValueError(f"aspect ratio {long / short:.2f} > MAX_ASPECT_RATIO {MAX_ASPECT_RATIO}")
    return image.convert("RGB")


def _check_flip(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError("flip_augmentation must be a bool")
    return value


INPUT_SCHEMA: dict[str, Any] = {
    "input": "PIL.Image.Image, or a sequence of them for the validation stage; any mode, converted to RGB",
    "short_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "long_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "aspect_ratio": [1.0, MAX_ASPECT_RATIO],
    "flip_augmentation": "bool; two forward passes (image and its mirror) averaged when true",
    "output": (
        f"{DEPTH_KIND} depth in {DEPTH_UNIT}, float32 H x W at the input resolution (larger = farther); "
        "an estimate with no confidence, routed to the NYU (0.001-10 m) or KITTI (0.001-80 m) head by "
        "the model's own domain classifier"
    ),
    "preprocessing": (
        "convert to RGB; the ZoeDepth processor resizes to fit 384x512 keeping the aspect ratio "
        "(sides rounded to multiples of 32), pads, normalises with mean/std 0.5; the prediction is "
        "interpolated back to the input resolution by post_process_depth_estimation"
    ),
}


def validate_inputs(
    images: Any, *, flip_augmentation: bool = FLIP_AUGMENTATION, names: Sequence[str] | None = None
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, request, verdict).

    Each image is routed through the public ``validate_image`` that ``predict`` itself calls, so a
    rejection here raises exactly what ``predict`` would; a caller that wants the finding recorded
    catches the exception and stores ``str(exc)`` under ``findings``.
    """
    batch = [images] if isinstance(images, Image.Image) else images
    if not isinstance(batch, Sequence) or isinstance(batch, str | bytes):
        raise TypeError("images must be a PIL.Image.Image or a sequence of them")
    if len(batch) < 1:
        raise ValueError("at least one image is required")
    if names is not None and len(names) != len(batch):
        raise ValueError("names must have one entry per image")
    flip = _check_flip(flip_augmentation)
    inputs = []
    for index, candidate in enumerate(batch):
        rgb = validate_image(candidate)
        width, height = rgb.size
        long_side, short_side = max(width, height), min(width, height)
        inputs.append(
            {
                "id": names[index] if names else f"image-{index}",
                "mode": getattr(candidate, "mode", rgb.mode),
                "size": [width, height],
                "aspect_ratio": round(long_side / short_side, 3),
            }
        )
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": inputs,
        "n_images": len(inputs),
        "flip_augmentation": flip,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any],
    reference_depth: Any | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``reference_depth`` (metric depth in metres, same H x W as the prediction, non-positive or
    non-finite pixels ignored) the report carries ``abs_rel`` and ``delta1`` computed without any
    alignment — the model claims metres, so scale errors count — as sample-sanity evidence. Without it
    the verdict is ``not-measurable`` and the report says what ground truth would make the task
    measurable: a depth map in metres has no intrinsic score.
    """
    depth = np.asarray(result["depth"])
    base = {
        "task": "monocular metric depth estimation",
        "score_semantics": (
            f"{result.get('depth_kind', DEPTH_KIND)} depth in {DEPTH_UNIT} with no confidence; the value "
            "is an estimate whose scale depends on the model having recognised the scene's domain and "
            "camera, and a blank image still yields a depth map"
        ),
        "sample_kind": sample_kind,
        "flip_augmentation": bool(result.get("flip_augmentation", FLIP_AUGMENTATION)),
        "n_images": 1,
        "n_pixels": int(depth.size),
        "depth_min_m": float(depth.min()),
        "depth_max_m": float(depth.max()),
        "depth_median_m": float(np.median(depth)),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if reference_depth is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no metric reference depth was supplied for the evaluated image",
            "needs": (
                "a metric depth map in metres with the same height and width as the image, from a depth "
                "sensor, LiDAR, stereo or an RGB-D benchmark, scored with abs_rel and delta1 against a "
                "constant-depth prior (the reference's median) as the trivial baseline"
            ),
        }
    ref = np.asarray(reference_depth, dtype=np.float64)
    valid = np.isfinite(ref) & (ref > 0)
    n_valid = int(valid.sum())
    constant = np.full_like(ref, float(np.median(ref[valid])) if n_valid else 1.0)
    return {
        **base,
        "metrics": [
            {
                "id": "abs_rel",
                "value": abs_rel(depth, ref),
                "align": False,
                "n_valid_pixels": n_valid,
                "estimation": "single image, no alignment (metric output as-is), no dispersion estimate",
            },
            {
                "id": "delta1",
                "value": delta1(depth, ref),
                "threshold": DELTA_THRESHOLD,
                "n_valid_pixels": n_valid,
                "estimation": "single image, fraction of valid pixels within the ratio threshold",
            },
        ],
        "baselines": [
            {
                "id": "constant_median_depth",
                "abs_rel": abs_rel(constant, ref),
                "delta1": delta1(constant, ref),
                "note": "every pixel predicted at the reference's median depth",
            }
        ],
        "verdict": "sample-sanity",
        "reason": "one image with caller-supplied metric depth; not a benchmark",
        "needs": "a held-out set of metric depth maps from the deployment domain for any generalisable claim",
    }


@dataclass
class ZoeDepthMetricPipeline:
    """Monocular metric depth estimation over the pinned ZoeDepth NYU+KITTI checkpoint."""

    _runner: Callable[[Image.Image, bool], np.ndarray]
    device: str

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> ZoeDepthMetricPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), {"local_files_only": True}
        elif allow_download:
            source, kwargs = MODEL_ID, {}
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage {MODEL_ID}@{MODEL_REVISION} under weights/{MODEL_KEY}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = ZoeDepthImageProcessor.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = ZoeDepthForDepthEstimation.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, dtype=torch.float32, **kwargs
        )
        model = model.to(resolved_device).eval()

        def runner(image: Image.Image, flip: bool) -> np.ndarray:
            inputs = processor(images=image, return_tensors="pt").to(resolved_device)
            with torch.inference_mode():
                outputs = model(**inputs)
                extra = {}
                if flip:
                    mirrored = torch.flip(inputs["pixel_values"], dims=[3])
                    extra["outputs_flipped"] = model(pixel_values=mirrored)
            # The pinned processor's post-processing un-pads, un-flips (when given) and interpolates
            # the prediction back to the source size; it returns metres.
            post = processor.post_process_depth_estimation(
                outputs, source_sizes=[(image.height, image.width)], **extra
            )
            return post[0]["predicted_depth"].float().cpu().numpy()

        return cls(runner, resolved_device)

    def predict(self, image: Image.Image, *, flip_augmentation: bool = FLIP_AUGMENTATION) -> dict[str, Any]:
        """Return metric depth in metres as a float32 H x W array at the input resolution."""
        rgb = validate_image(image)
        flip = _check_flip(flip_augmentation)
        depth = np.asarray(self._runner(rgb, flip), dtype=np.float32)
        if depth.shape != (rgb.height, rgb.width):
            raise RuntimeError(f"backend returned shape {depth.shape}, expected {(rgb.height, rgb.width)}")
        if not np.all(np.isfinite(depth)) or depth.min() <= 0:
            raise RuntimeError("backend returned non-finite or non-positive depth values")
        return {
            "depth": depth,
            "depth_kind": DEPTH_KIND,
            "depth_unit": DEPTH_UNIT,
            "depth_min": float(depth.min()),
            "depth_max": float(depth.max()),
            "depth_median": float(np.median(depth)),
            "flip_augmentation": flip,
            "height": rgb.height,
            "width": rgb.width,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
