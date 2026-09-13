"""Role-helper contract: validate_inputs (validation stage) and evaluation_report (evaluation stage)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from zoedepth_metric_depth_pipeline import (
    DELTA_THRESHOLD,
    INPUT_SCHEMA,
    MAX_ASPECT_RATIO,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    abs_rel,
    evaluation_report,
    validate_inputs,
)


def _image(width: int = 64, height: int = 48, mode: str = "RGB") -> Image.Image:
    return Image.new(mode, (width, height))


def _result(depth: np.ndarray, flip: bool = False) -> dict:
    return {
        "depth": depth.astype(np.float32),
        "depth_kind": "metric",
        "depth_unit": "metres",
        "flip_augmentation": flip,
        "height": depth.shape[0],
        "width": depth.shape[1],
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs([_image(), _image(96, 32, "L")], names=["room.png", "wide.png"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["short_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["aspect_ratio"] == [1.0, MAX_ASPECT_RATIO]
    assert manifest["inputs"] == [
        {"id": "room.png", "mode": "RGB", "size": [64, 48], "aspect_ratio": 1.333},
        {"id": "wide.png", "mode": "L", "size": [96, 32], "aspect_ratio": 3.0},
    ]
    assert manifest["n_images"] == 2 and manifest["flip_augmentation"] is False
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_single_image_default_id_and_flip() -> None:
    manifest = validate_inputs(_image(), flip_augmentation=True)
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["flip_augmentation"] is True


def test_validate_inputs_rejects_like_predict() -> None:
    with pytest.raises(TypeError, match="sequence"):
        validate_inputs("room.png")
    with pytest.raises(ValueError, match="at least one"):
        validate_inputs([])
    with pytest.raises(ValueError, match="MAX_ASPECT_RATIO"):
        validate_inputs([_image(400, 64)])
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs([_image(8, 8)])
    with pytest.raises(TypeError, match="flip_augmentation"):
        validate_inputs([_image()], flip_augmentation="yes")
    with pytest.raises(ValueError, match="one entry per image"):
        validate_inputs([_image()], names=["a", "b"])


def test_evaluation_report_not_measurable_without_reference() -> None:
    depth = np.linspace(1.0, 3.0, 64 * 48).reshape(48, 64)
    report = evaluation_report(_result(depth), sample_kind="BYOD")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == [] and report["baselines"] == []
    assert report["n_pixels"] == 64 * 48 and report["depth_min_m"] == pytest.approx(1.0)
    assert report["depth_median_m"] == pytest.approx(2.0, abs=0.01) and report["flip_augmentation"] is False
    assert "abs_rel" in report["needs"] and "delta1" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert "no confidence" in report["score_semantics"]


def test_evaluation_report_sample_sanity_with_reference() -> None:
    ref = np.full((48, 64), 2.0)
    pred = ref * 1.2
    pred[:, :32] = 2.0
    report = evaluation_report(_result(pred, flip=True), ref)
    assert report["verdict"] == "sample-sanity" and report["flip_augmentation"] is True
    by_id = {metric["id"]: metric for metric in report["metrics"]}
    assert by_id["abs_rel"]["value"] == pytest.approx(0.1) and by_id["abs_rel"]["align"] is False
    assert by_id["delta1"]["value"] == 1.0 and by_id["delta1"]["threshold"] == DELTA_THRESHOLD
    assert by_id["abs_rel"]["n_valid_pixels"] == 48 * 64
    baseline = report["baselines"][0]
    assert (
        baseline["id"] == "constant_median_depth" and baseline["abs_rel"] == 0.0 and baseline["delta1"] == 1.0
    )
    assert abs_rel(pred, ref) == pytest.approx(by_id["abs_rel"]["value"], rel=1e-5)


def test_evaluation_report_rejects_mismatched_reference() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluation_report(_result(np.ones((48, 64))), np.ones((10, 10)))
