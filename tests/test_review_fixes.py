"""Regression tests for the 2026-10-05 notebook review findings (ZOE-M1..M3, ZOE-m1..m3).

They need only CI's lightweight dependencies: the generated notebook's own learner cells are executed with a stand-in
depth model (no weights, no torch call), on the real generated scenes and on real ZIP archives. None of this is model
evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import zoedepth_metric_depth_pipeline.pipeline as pipeline_module

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"zoe_fix_{name}", ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(notebook: dict) -> list[str]:
    return [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


def _cell(notebook: dict, needle: str) -> str:
    (source,) = [s for s in _code(notebook) if needle in s]
    return source


class _Tensor:
    def __init__(self, value):
        self.value = np.array(value, dtype=np.float64)

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return _Tensor(self.value.copy())


class _Head:
    def __init__(self):
        self.weight = _Tensor([1.0, 2.0])

    def state_dict(self):
        return {"weight": self.weight}

    def load_state_dict(self, state):
        self.weight = state["weight"].clone()


class _StubPipe:
    """Predicts depth from the head's weight so a restored head is observable; upside down like the real model."""

    def __init__(self):
        self.model = types.SimpleNamespace(metric_head=_Head())
        self.adaptation_config = {}
        self.device = "cpu"

    def predict(self, image):
        height = image.height
        rows = np.linspace(3.0, 1.0, height)[:, None] * np.ones((1, image.width))
        return {"depth": rows * float(self.model.metric_head.weight.value[0])}

    def freeze_for_adaptation(self):
        self.adaptation_config = {"trainable_parameters": 2}
        return {"trainable_parameters": 2, "frozen_parameters": 10}

    def finetune(self, train_records, val_records, *, epochs, learning_rate, seed):
        self.model.metric_head.weight.value = self.model.metric_head.weight.value + epochs * 0.1
        self.adaptation_config["weight_delta_l2"] = 0.1 * epochs
        return [{"epoch": e + 1, "optimizer_steps": len(train_records)} for e in range(epochs)]

    def evaluate_adaptation(self, records):
        return {"abs_rel": 0.466, "delta1": 0.212}


def _namespace(tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.chdir(tmp_path)
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    ns = {k: v for k, v in vars(pipeline_module).items() if not k.startswith("__")}
    ns.update(json=json, torch=fake_torch, pipe=_StubPipe())
    return ns


def _section4(notebook: dict, ns: dict, **changes: str) -> None:
    source = _cell(notebook, "def generated_depth_records(")
    for old, new in changes.items():
        source = source.replace(old, new)
    exec(source, ns)  # noqa: S102 - the notebook's own cell


def _zip(path: Path, n: int, *, prefix: str = "", sparse: bool = False) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(n):
            buffer = io.BytesIO()
            Image.fromarray(np.full((96, 128, 3), 40 + i, dtype=np.uint8)).save(buffer, format="PNG")
            archive.writestr(f"{prefix}images/frame{i:02d}.png", buffer.getvalue())
            depth = np.full((96, 128), 2.0 + 0.01 * i, dtype=np.float32)
            if sparse:
                depth[0, 0] = 0.0
            depth_buffer = io.BytesIO()
            np.save(depth_buffer, depth)
            archive.writestr(f"{prefix}depth/frame{i:02d}.npy", depth_buffer.getvalue())
    return path


# ---- ZOE-M1: isolated runtime, no restart --------------------------------------------------------------------------


def test_zoe_M1_no_kernel_install_no_restart_and_a_hash_lock(notebook: dict) -> None:
    assert "Restart the runtime" not in NOTEBOOK.read_text(encoding="utf-8")
    sources = _code(notebook)
    assert not any("[sys.executable, '-m', 'pip'" in s for s in sources)
    kernel = [s for s in sources if "# dimer: kernel cell" in s]
    assert len(kernel) == 2
    install = next(s for s in kernel if "LOCK_TEXT = r'''" in s)
    for needed in ('"--managed-python"', '"--require-hashes"', '"--only-binary"', "UV_SHA256", "LOCK_SHA256"):
        assert needed in install
    build.check_lock(build._pins(ROOT, TEMPLATE), (ROOT / TEMPLATE["lock"]).read_text(encoding="utf-8"))


# ---- ZOE-M2: the inverted layout is disclosed, shown and interpreted ----------------------------------------------


def test_zoe_M2_generated_scenes_are_nearest_at_the_top_and_the_notebook_says_so(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    ns = _namespace(tmp_path, monkeypatch)
    _section4(notebook, ns)
    rows = np.asarray(ns["dataset_records"][0]["depth_m"]).mean(axis=1)
    assert rows[0] < 1.6 and rows[-1] > 3.3 and np.corrcoef(rows, np.arange(len(rows)))[0, 1] > 0.9
    md = _markdown(notebook)
    assert "**Read this sample for what it is: its depth layout is upside down.**" in md
    assert "0.431" in md and "0.453" in md and "0.466" in md and "0.258" in md


def test_zoe_M2_section8_draws_the_scene_and_interprets_against_the_constant(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    ns = _namespace(tmp_path, monkeypatch)
    _section4(notebook, ns)
    exec(_cell(notebook, "dataset_manifest = {'kind': dataset_kind"), ns)  # noqa: S102 - Section 5
    exec(_cell(notebook, "base_eval = score_records(pipe, val_records, training_median)"), ns)  # noqa: S102 - Section 6
    exec(_cell(notebook, "history = pipe.finetune("), ns)  # noqa: S102 - Section 7
    exec(_cell(notebook, "adapted_eval = pipe.evaluate_adaptation(val_records)"), ns)  # noqa: S102 - Section 8
    interpretation = json.loads((tmp_path / "outputs" / f"{TEMPLATE['stem']}_evaluation_report.json").read_text(encoding="utf-8"))["interpretation"]
    assert interpretation["row_correlation_pretrained"] < 0  # the stand-in, like the model, slopes the wrong way
    assert {"pretrained_beats_constant_abs_rel", "pretrained_beats_constant_delta1", "abs_rel_change_after_training"} <= set(interpretation)
    assert Image.open(tmp_path / "outputs" / f"{TEMPLATE['stem']}_heldout_example.png").size == (528, 100)


# ---- ZOE-M3: the guided layer --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marker",
    ["## How to use this notebook", "**Who this notebook is for.**", "## The task: Input → Model/System → Output", "## Roadmap", "<strong>Glossary</strong>", "**Predict before running:**", "**What to notice", "<summary>Check your reasoning</summary>", "## 12. Activity: change one thing — train longer", "## Troubleshooting", "## Conclusion (your notes)"],
)
def test_zoe_M3_guided_layer_marker_is_present(notebook: dict, marker: str) -> None:
    assert marker in _markdown(notebook)


def test_zoe_M3_infrastructure_cells_are_collapsed(notebook: dict) -> None:
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    first_learner = next(i for i, c in enumerate(code) if "USE_BYOD = False" in c["source"])
    assert first_learner >= 5
    for cell in code[:first_learner]:
        assert cell["metadata"].get("cellView") == "form", cell["source"][:60]


# ---- ZOE-m1: BYOD by path, folder ZIPs, stated minimum, named sparse-depth refusal ---------------------------------


def test_zoe_m1_folder_zipped_byod_by_path_is_accepted(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    ns = _namespace(tmp_path, monkeypatch)
    archive = _zip(tmp_path / "my_rgbd.zip", 8, prefix="my_rgbd/")
    _section4(notebook, ns, **{"USE_BYOD = False": "USE_BYOD = True", "BYOD_ZIP_PATH = ''": f"BYOD_ZIP_PATH = {str(archive)!r}"})
    assert ns["dataset_kind"] == "BYOD" and (len(ns["train_records"]), len(ns["val_records"])) == (6, 2)


@pytest.mark.parametrize(
    ("n", "kwargs", "message"),
    [(4, {}, r"has 4 image/depth pairs; the notebook needs 8\.\."), (8, {"sparse": True}, r"frame00: depth has zero, negative or non-finite values"), (0, {}, r"has no images/<id>\.png\|jpg and depth/<id>\.npy pairs")],
)
def test_zoe_m1_refusals_name_the_rule(notebook: dict, tmp_path: Path, monkeypatch, n: int, kwargs: dict, message: str) -> None:
    ns = _namespace(tmp_path, monkeypatch)
    archive = _zip(tmp_path / "bad.zip", n, **kwargs) if n else tmp_path / "empty.zip"
    if not n:
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("notes.txt", "nothing here")
    with pytest.raises(ValueError, match=message):
        _section4(notebook, ns, **{"USE_BYOD = False": "USE_BYOD = True", "BYOD_ZIP_PATH = ''": f"BYOD_ZIP_PATH = {str(archive)!r}"})


@pytest.mark.parametrize("uploads", [{}, {"a.zip": b"x", "b.zip": b"y"}])
def test_zoe_m1_cancelled_or_multi_file_upload_stops_with_the_rule(notebook: dict, tmp_path: Path, monkeypatch, uploads: dict) -> None:
    ns = _namespace(tmp_path, monkeypatch)
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: uploads
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(RuntimeError, match=rf"Upload exactly one ZIP \(received {len(uploads)}\)"):
        _section4(notebook, ns, **{"USE_BYOD = False": "USE_BYOD = True"})


# ---- ZOE-m2: re-runs start from the pretrained head ------------------------------------------------------------------


def test_zoe_m2_rerunning_sections_6_and_7_does_not_accumulate_training(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    ns = _namespace(tmp_path, monkeypatch)
    _section4(notebook, ns)
    exec(_cell(notebook, "dataset_manifest = {'kind': dataset_kind"), ns)  # noqa: S102
    section6 = _cell(notebook, "base_eval = score_records(pipe, val_records, training_median)")
    section7 = _cell(notebook, "history = pipe.finetune(")
    exec(section6, ns)  # noqa: S102
    first_baseline = dict(ns["base_eval"])
    exec(section7, ns)  # noqa: S102
    after_one = float(ns["pipe"].model.metric_head.weight.value[0])
    exec(section7, ns)  # noqa: S102 - re-run with the same settings
    assert float(ns["pipe"].model.metric_head.weight.value[0]) == after_one
    exec(section6, ns)  # noqa: S102 - re-running the baseline after training still scores the pretrained head
    assert ns["base_eval"] == first_baseline


# ---- ZOE-m3: CUDA is checked before the download; the records say what ran ------------------------------------------


def test_zoe_m3_cuda_is_checked_in_section_1_and_the_readme_claims_no_byod_run(notebook: dict) -> None:
    runtime_cell = next(s for s in _code(notebook) if "NOTEBOOK_SOURCE = {" in s)
    model_cell = next(s for s in _code(notebook) if "stage_missing_files(WEIGHTS_DIR, allow_download=True)" in s)
    code = _code(notebook)
    assert "raise RuntimeError('This notebook needs an NVIDIA GPU exposed as CUDA" in runtime_cell
    assert code.index(runtime_cell) < code.index(model_cell)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "and BYOD path" not in readme and "no BYOD run is recorded" in readme
