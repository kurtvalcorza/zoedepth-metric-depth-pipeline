import builtins
import importlib.util

import pytest

# test_adaptation.py imports torch at module level; without torch (a lightweight local run) it is
# skipped at collection
# instead of erroring. CI installs torch, so it always runs there.
collect_ignore = [] if importlib.util.find_spec("torch") else ["test_adaptation.py"]


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing or initializing model libraries."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in {"torch", "transformers", "timm", "gliner"}:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
