"""Tier C review findings T118-1..T118-4 and T118-S (2026-10-25)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "zoedepth_metric_depth_colab.ipynb"


def _nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def _markdown():
    md = "\n".join("".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "markdown")
    return " ".join(md.split())


def test_t118_1_no_stale_run_claims():
    md = _markdown()
    for stale in ("no run of this revision is recorded yet", "this version has not been timed yet",
                  "The one recorded hosted run"):
        assert stale not in md, stale
    assert "`441111a9`" in md


def test_t118_2_only_printed_numbers():
    md = _markdown()
    for unprinted in ("1.25 m (top)", "3.85 m", "2.29 m to 2.01 m", "fell from 0.433", "0.72–4.38 m"):
        assert unprinted not in md, unprinted
    assert "0.4287" in md and "−0.841" in md


def test_t118_3_holdout_called_near_copies():
    assert "near-copies" in _markdown()


def test_t118_4_dat9_overlap_statement():
    md = _markdown()
    assert "**Overlap with the training data:**" in md and "verified by construction" in md


def test_t118_s_spec_items():
    nb = _nb()
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    head = "\n".join("".join(c["source"]) for c in nb["cells"][:first_code])
    assert "Notice.** For research, training and evaluation only." in head
    md = _markdown()
    assert "[1] S. F. Bhat" in md and "`DIMER_ISOLATED_ENV`" in md
