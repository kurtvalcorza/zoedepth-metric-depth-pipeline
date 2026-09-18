from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import zoedepth_metric_depth_pipeline.pipeline as pipeline_module
from zoedepth_metric_depth_pipeline.pipeline import (
    ARTIFACT_MANIFEST_NAME,
    ARTIFACT_WEIGHTS_NAME,
    ZoeDepthMetricPipeline,
    _prepare_depth_target,
    validate_depth_dataset,
)


class _Batch(dict):
    def to(self, device):
        return _Batch({key: value.to(device) for key, value in self.items()})


class _Processor:
    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        array = np.asarray(images, dtype=np.float32) / 255.0
        values = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return _Batch(pixel_values=values)


class _FakeZoeDepth(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Conv2d(3, 3, 1)
        self.metric_head = torch.nn.Conv2d(3, 1, 1)

    def forward(self, *, pixel_values):
        depth = torch.nn.functional.softplus(self.metric_head(pixel_values))[:, 0] + 0.1
        return SimpleNamespace(predicted_depth=depth)


def _records() -> list[dict]:
    records = []
    for index in range(4):
        height, width = 32, 40
        yy, xx = np.mgrid[:height, :width]
        red = ((xx + index * 3) / (width + 12) * 255).astype(np.uint8)
        green = (yy / (height - 1) * 255).astype(np.uint8)
        blue = np.full_like(red, 80 + index * 10)
        image = Image.fromarray(np.stack([red, green, blue], axis=-1))
        depth = (0.8 + 2.0 * yy / (height - 1) + 0.5 * xx / (width - 1)).astype(np.float32)
        records.append({"id": f"depth-{index}", "image": image, "depth_m": depth})
    return records


def _pipeline(model=None) -> ZoeDepthMetricPipeline:
    model = model or _FakeZoeDepth()

    def runner(image, flip):
        return np.ones((image.height, image.width), dtype=np.float32)

    return ZoeDepthMetricPipeline(
        runner,
        "cpu",
        model=model,
        processor=_Processor(),
    )


def test_dataset_validation_rejects_duplicate_ids_and_invalid_depth():
    records = _records()
    assert validate_depth_dataset(records)["records"] == 4
    records[1]["id"] = records[0]["id"]
    with pytest.raises(ValueError, match="duplicate record id"):
        validate_depth_dataset(records)
    records = _records()
    records[0]["depth_m"][0, 0] = 0.0
    with pytest.raises(ValueError, match="finite positive"):
        validate_depth_dataset(records)


def test_finetune_updates_only_metric_head():
    pipeline = _pipeline()
    counts = pipeline.freeze_for_adaptation()
    assert counts["trainable_parameters"] > 0
    assert pipeline.model.backbone.weight.requires_grad is False
    before = pipeline.model.metric_head.weight.detach().clone()
    history = pipeline.finetune(_records()[:2], _records()[2:], epochs=1, learning_rate=1e-2)
    after = pipeline.model.metric_head.weight.detach()
    assert history[0]["optimizer_steps"] == 2
    assert not torch.equal(before, after)
    assert pipeline.adaptation_config["weight_delta_l2"] > 0
    assert pipeline.adaptation_config["optimizer"] == "AdamW"
    assert pipeline.adaptation_config["optimizer_betas"] == [0.9, 0.999]
    assert pipeline.adaptation_config["optimizer_epsilon"] == 1e-8
    assert pipeline.adaptation_config["optimizer_weight_decay"] == 0.01


def test_finetune_rejects_train_validation_overlap_by_id_or_content():
    records = _records()
    with pytest.raises(ValueError, match="overlap by id"):
        _pipeline().finetune(records[:2], records[1:3], epochs=1)

    renamed = dict(records[0], id="renamed-depth")
    with pytest.raises(ValueError, match="overlap by RGB content"):
        _pipeline().finetune(records[:2], [renamed, records[2]], epochs=1)

    recalibrated = dict(
        records[0],
        id="recalibrated-depth",
        depth_m=records[0]["depth_m"] + 0.1,
    )
    with pytest.raises(ValueError, match="overlap by RGB content"):
        _pipeline().finetune(records[:2], [recalibrated, records[2]], epochs=1)


def test_finetune_converts_validated_images_to_rgb():
    records = [dict(record, image=record["image"].convert("L")) for record in _records()[:2]]
    pipeline = _pipeline()
    pipeline.freeze_for_adaptation()
    history = pipeline.finetune(records, epochs=1, learning_rate=1e-2)
    assert history[0]["optimizer_steps"] == 2


def test_failed_retraining_restores_weights_and_metadata(monkeypatch):
    pipeline = _pipeline()
    pipeline.freeze_for_adaptation()
    pipeline.adaptation_config.update(
        {
            "weight_delta_l2": 1.0,
            "training_median_depth_m": 2.0,
            "history": [{"epoch": 1}],
        }
    )
    previous_config = dict(pipeline.adaptation_config)
    before = {
        name: parameter.detach().clone()
        for name, parameter in pipeline.model.named_parameters()
        if parameter.requires_grad
    }
    original_forward = pipeline.model.forward

    def non_finite_forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        return SimpleNamespace(predicted_depth=outputs.predicted_depth * float("nan"))

    monkeypatch.setattr(pipeline.model, "forward", non_finite_forward)
    with pytest.raises(RuntimeError, match="non-finite loss"):
        pipeline.finetune(_records()[:2], epochs=1, learning_rate=1e-2)
    assert pipeline.adaptation_config == previous_config
    for name, parameter in pipeline.model.named_parameters():
        if name in before:
            assert torch.equal(parameter, before[name])


def test_dataset_validation_bounds_total_pixels(monkeypatch):
    monkeypatch.setattr(pipeline_module, "MAX_ADAPTATION_PIXELS", 2_000)
    with pytest.raises(ValueError, match="MAX_ADAPTATION_PIXELS"):
        validate_depth_dataset(_records()[:2])


def test_evaluation_weights_metrics_by_valid_pixels():
    small = {
        "id": "small",
        "image": Image.new("RGB", (32, 32), "white"),
        "depth_m": np.ones((32, 32), dtype=np.float32),
    }
    large = {
        "id": "large",
        "image": Image.new("RGB", (64, 64), "white"),
        "depth_m": np.full((64, 64), 2.0, dtype=np.float32),
    }
    pipeline = _pipeline()
    pipeline.adaptation_config["training_median_depth_m"] = 1.0
    report = pipeline.evaluate_adaptation([small, large])
    assert report["valid_pixels"] == 5_120
    assert report["abs_rel"] == pytest.approx(0.4)
    assert report["delta1"] == pytest.approx(0.2)


def test_depth_target_uses_processor_padding_before_resize():
    class _PaddingProcessor:
        do_pad = True

        def pad_image(self, image, *, input_data_format, data_format):
            assert input_data_format == "channels_last"
            assert data_format == "channels_last"
            return np.pad(image, ((1, 1), (2, 2), (0, 0)), mode="reflect")

    depth = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0
    target = _prepare_depth_target(
        depth,
        _PaddingProcessor(),
        output_size=(5, 8),
        device=torch.device("cpu"),
    )
    padded = np.pad(depth, ((1, 1), (2, 2)), mode="reflect")
    expected = torch.nn.functional.interpolate(
        torch.from_numpy(padded)[None, None],
        size=(5, 8),
        mode="bilinear",
        align_corners=True,
    )[:, 0]
    assert torch.equal(target, expected)


def test_artifact_round_trip_and_integrity_rejection(tmp_path):
    source = _pipeline()
    source.freeze_for_adaptation()
    source.adaptation_config.update(
        {
            "weight_delta_l2": 1.0,
            "training_median_depth_m": 2.0,
            "history": [{"epoch": 1}],
        }
    )
    artifact = source.save_artifact(tmp_path / "artifact", producer_revision="b" * 40)

    fresh = _pipeline()
    fresh.load_artifact(artifact)
    assert torch.equal(source.model.metric_head.weight, fresh.model.metric_head.weight)

    unexpected = artifact / "unexpected.txt"
    unexpected.write_text("not declared", encoding="utf-8")
    with pytest.raises(ValueError, match="contain exactly"):
        _pipeline().load_artifact(artifact)
    unexpected.unlink()

    unexpected_dir = artifact / "retained-data"
    unexpected_dir.mkdir()
    with pytest.raises(ValueError, match="contain exactly"):
        _pipeline().load_artifact(artifact)
    unexpected_dir.rmdir()

    manifest_path = artifact / ARTIFACT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["adaptation"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rejected = _pipeline()
    before = rejected.model.metric_head.weight.detach().clone()
    with pytest.raises(ValueError, match="adaptation metadata"):
        rejected.load_artifact(artifact)
    assert torch.equal(before, rejected.model.metric_head.weight)

    manifest["adaptation"] = source.adaptation_config
    manifest["files"][0]["bytes"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="size or SHA-256"):
        _pipeline().load_artifact(artifact)
    assert (artifact / ARTIFACT_WEIGHTS_NAME).is_file()


def test_artifact_export_requires_completed_update(tmp_path):
    pipeline = _pipeline()
    pipeline.freeze_for_adaptation()
    with pytest.raises(RuntimeError, match="completed finite fine-tuning"):
        pipeline.save_artifact(tmp_path / "artifact", producer_revision="a" * 40)
