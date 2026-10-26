"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import types
import zipfile
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
import pytest

from test_review_fixes import _namespace, _section4, _zip

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/zoedepth_metric_depth_colab.ipynb"
BYOD_CELL = "zoedepth_metric_depth-16"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


STEM = BYOD_CELL.rsplit("-", 1)[0]
NB_DATA = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
EARLIER = {"dataset_records": "old", "train_records": "old", "base_eval": "old", "adapted_eval": "old", "reload_summary": "old", "payload": "old"}


def _seed(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    (out / "zoedepth-metric-head-adapter-v1").mkdir(parents=True)
    for name in ("dataset_manifest.json", "evaluation_report.json", "result.json"):
        (out / f"{STEM}_{name}").write_text("{}", encoding="utf-8")
    (out / "unrelated.txt").write_text("kept", encoding="utf-8")


def _byod(tmp_path, monkeypatch, path: str, ns: dict | None = None) -> dict:
    ns = ns if ns is not None else {**_namespace(tmp_path, monkeypatch), **EARLIER}
    _seed(tmp_path)
    _section4(NB_DATA, ns, **{"USE_BYOD = False": "USE_BYOD = True", "BYOD_ZIP_PATH = ''": f"BYOD_ZIP_PATH = {path!r}"})
    return ns


def _cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime\n", encoding="utf-8")
    _zip(tmp_path / "sparse.zip", 8, sparse=True)
    _zip(tmp_path / "four.zip", 4)
    archive = _zip(tmp_path / "shape.zip", 8)
    with zipfile.ZipFile(archive) as source:
        entries = {i.filename: source.read(i.filename) for i in source.infolist()}
    depth_name = next(n for n in entries if n.endswith(".npy"))
    buffer = io.BytesIO()
    np.save(buffer, np.ones((3, 3), dtype=np.float32))
    entries[depth_name] = buffer.getvalue()
    with zipfile.ZipFile(tmp_path / "shape.zip", "w") as target:
        for name, data in entries.items():
            target.writestr(name, data)
    (tmp_path / "folder").mkdir()
    return {
        "text file, not a ZIP (the scripted /etc/hostname run)": ("hostname", "File is not a zip file"),
        "sparse depth": ("sparse.zip", "depth has zero, negative or non-finite values"),
        "too few pairs": ("four.zip", "has 4 image/depth pairs"),
        "depth shape differs from the image (validate_depth_dataset)": ("shape.zip", "record"),
        "a folder (OSError)": ("folder", "folder"),
        "missing path (OSError)": ("absent.zip", "absent.zip"),
    }


CASES = ["text file, not a ZIP (the scripted /etc/hostname run)", "sparse depth", "too few pairs",
         "depth shape differs from the image (validate_depth_dataset)", "a folder (OSError)", "missing path (OSError)"]


@pytest.mark.parametrize("case", CASES)
def test_byod_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    file_name, rule = _cases(tmp_path)[case]
    ns = {**_namespace(tmp_path, monkeypatch), **EARLIER}
    with pytest.raises(ValueError) as caught:
        _byod(tmp_path, monkeypatch, str(tmp_path / file_name), ns)
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    assert " again" in line and "earlier results of this session and their outputs/ files were removed" in line
    assert not set(EARLIER) & set(ns), "an earlier run's results survived the refusal"
    assert "pipe" in ns
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == ["unrelated.txt"]


def test_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _byod(tmp_path, monkeypatch, "")
    assert _refusal_line(caught.value).startswith("Refused: (no file): there is no upload dialog outside Google Colab")


@pytest.mark.parametrize(("uploads", "file_name"), [({}, "(no file)"), ({"a.zip": b"x", "b.zip": b"y"}, "a.zip, b.zip")])
def test_cancelled_or_multiple_upload_is_one_refused_line(monkeypatch, tmp_path, uploads, file_name):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploads)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError) as caught:
        _byod(tmp_path, monkeypatch, "")
    assert _refusal_line(caught.value).startswith(f"Refused: {file_name}: Upload exactly one ZIP")


def test_uploaded_zip_is_named_in_its_refusal(monkeypatch, tmp_path):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: {"mine.zip": b"not a zip"})
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError) as caught:
        _byod(tmp_path, monkeypatch, "")
    assert _refusal_line(caught.value).startswith("Refused: mine.zip: File is not a zip file")


def test_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    archive = _zip(tmp_path / "good.zip", 8)
    ns = {**_namespace(tmp_path, monkeypatch), **EARLIER}

    def broken(_records):
        raise RuntimeError("internal bug")

    ns["validate_depth_dataset"] = broken
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _byod(tmp_path, monkeypatch, str(archive), ns)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert set(EARLIER) <= set(ns) and (tmp_path / "outputs" / f"{STEM}_result.json").exists()  # nothing removed


def test_good_zip_and_generated_default_still_load(monkeypatch, tmp_path):
    ns = _byod(tmp_path, monkeypatch, str(_zip(tmp_path / "good.zip", 8)))
    assert ns["dataset_kind"] == "BYOD" and len(ns["train_records"]) == 6
    ns = _namespace(tmp_path, monkeypatch)
    _section4(NB_DATA, ns)
    assert ns["dataset_kind"] == "generated" and len(ns["dataset_records"]) == 24


@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs a POSIX /etc/hostname, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_hostname(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD=True;BYOD_ZIP_PATH='/etc/hostname': a text file present in every fresh runtime."""
    with pytest.raises(ValueError) as caught:
        _byod(tmp_path, monkeypatch, "/etc/hostname")
    assert _refusal_line(caught.value).startswith("Refused: hostname: File is not a zip file")
