import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from zoedepth_metric_depth_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    DELTA_THRESHOLD,
    DEPTH_KIND,
    DEPTH_RANGES_M,
    DEPTH_UNIT,
    FLIP_AUGMENTATION,
    MAX_ASPECT_RATIO,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    ZoeDepthMetricPipeline,
    abs_rel,
    delta1,
    stage_missing_files,
    validate_image,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "Intel/zoedepth-nyu-kitti"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert DEPTH_KIND == "metric" and DEPTH_UNIT == "metres" and FLIP_AUGMENTATION is False
    assert DEPTH_RANGES_M == {"nyu": (0.001, 10.0), "kitti": (0.001, 80.0)} and DELTA_THRESHOLD == 1.25
    assert MIN_IMAGE_SIDE == 32 and MAX_IMAGE_SIDE == 4096 and MAX_ASPECT_RATIO == 4.0
    manifest = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["modelId"] == MODEL_ID
        assert data["revision"] == MODEL_REVISION
        paths = [entry["path"] for entry in data["files"]]
        assert "model.safetensors" in paths and "pytorch_model.bin" not in paths
        config = json.loads((REPO / "weights" / MODEL_KEY / "config.json").read_text(encoding="utf-8"))
        ranges = {b["name"]: (b["min_depth"], b["max_depth"]) for b in config["bin_configurations"]}
        assert ranges == DEPTH_RANGES_M


def _write_snapshot(root: Path, content: bytes, sha: str | None = None, size: int | None = None) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_manifest(tmp_path):
    _write_snapshot(tmp_path, b'{"model_type": "zoedepth"}')
    info = verify_snapshot(tmp_path)
    assert info["revision"] == MODEL_REVISION and info["files"] == 1


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"model_type": "zoedepth"}'
    good = hashlib.sha256(content).hexdigest()
    flipped = ("0" if good[0] != "0" else "1") + good[1:]
    _write_snapshot(tmp_path, content, sha=flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_size_missing_file_and_revision(tmp_path):
    _write_snapshot(tmp_path, b"abc", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["revision"] = "0" * 40
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_validate_image_ceilings():
    assert validate_image(Image.new("L", (64, 48))).mode == "RGB"
    with pytest.raises(TypeError):
        validate_image(np.zeros((48, 64, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_image(Image.new("RGB", (MIN_IMAGE_SIDE - 1, 64)))
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        validate_image(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 64)))
    with pytest.raises(ValueError, match="MAX_ASPECT_RATIO"):
        validate_image(Image.new("RGB", (int(64 * MAX_ASPECT_RATIO) + 8, 64)))


def _fake_pipeline(calls: list | None = None) -> ZoeDepthMetricPipeline:
    def runner(image: Image.Image, flip: bool) -> np.ndarray:
        if calls is not None:
            calls.append((image.mode, image.size, flip))
        depth = np.linspace(1.0, 3.0, image.width, dtype=np.float32)
        return np.tile(depth, (image.height, 1)) * (1.1 if flip else 1.0)

    return ZoeDepthMetricPipeline(runner, "cpu")


def test_predict_output_fields_and_defaults():
    calls: list = []
    pipe = _fake_pipeline(calls)
    result = pipe.predict(Image.new("L", (64, 48)))
    assert calls == [("RGB", (64, 48), False)]
    assert result["depth"].shape == (48, 64) and result["depth"].dtype == np.float32
    assert (result["depth_kind"], result["depth_unit"]) == ("metric", "metres")
    assert result["depth_min"] == pytest.approx(1.0) and result["depth_max"] == pytest.approx(3.0)
    assert result["depth_median"] == pytest.approx(2.0, abs=0.05)
    assert result["flip_augmentation"] is False
    assert (result["height"], result["width"]) == (48, 64)
    assert (result["model_id"], result["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_predict_flip_is_caller_owned_and_type_checked():
    calls: list = []
    pipe = _fake_pipeline(calls)
    result = pipe.predict(Image.new("RGB", (64, 48)), flip_augmentation=True)
    assert calls == [("RGB", (64, 48), True)] and result["flip_augmentation"] is True
    assert result["depth_max"] == pytest.approx(3.3)
    with pytest.raises(TypeError, match="flip_augmentation"):
        pipe.predict(Image.new("RGB", (64, 48)), flip_augmentation=1)


def test_predict_rejects_malformed_backend_output():
    wrong_shape = ZoeDepthMetricPipeline(lambda image, flip: np.ones((10, 10), np.float32), "cpu")
    with pytest.raises(RuntimeError, match="shape"):
        wrong_shape.predict(Image.new("RGB", (64, 48)))
    non_positive = ZoeDepthMetricPipeline(lambda image, flip: np.zeros((48, 64), np.float32), "cpu")
    with pytest.raises(RuntimeError, match="non-positive"):
        non_positive.predict(Image.new("RGB", (64, 48)))
    non_finite = ZoeDepthMetricPipeline(lambda image, flip: np.full((48, 64), np.nan, np.float32), "cpu")
    with pytest.raises(RuntimeError, match="non-finite"):
        non_finite.predict(Image.new("RGB", (64, 48)))


def test_abs_rel_and_delta1_without_alignment():
    ref = np.full((4, 4), 2.0)
    assert abs_rel(ref, ref) == 0.0 and delta1(ref, ref) == 1.0
    assert abs_rel(ref * 1.5, ref) == pytest.approx(0.5)  # a scale error counts: metric output is not aligned
    assert delta1(ref * 1.5, ref) == 0.0 and delta1(ref * 1.2, ref) == 1.0
    ref_holes = ref.copy()
    ref_holes[0, :] = 0.0  # invalid reference pixels are ignored
    assert abs_rel(ref * 1.5, ref_holes) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="shape mismatch"):
        abs_rel(ref, np.ones((2, 2)))
    with pytest.raises(ValueError, match="valid reference"):
        delta1(ref, np.zeros((4, 4)))
