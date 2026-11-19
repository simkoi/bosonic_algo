"""Regression checks for running each notebook directly from a repository checkout."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "01_fourier_workflow.ipynb",
    "02_dynamics.ipynb",
    "03_results_analysis.ipynb",
)


@pytest.mark.parametrize("notebook_name", NOTEBOOKS)
@pytest.mark.parametrize("working_dir", (".", "notebooks"))
def test_notebook_imports_package_from_checkout(monkeypatch, notebook_name, working_dir):
    notebook_path = ROOT / "notebooks" / notebook_name
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    startup = "".join(notebook["cells"][1]["source"])

    monkeypatch.chdir(ROOT / working_dir)
    old_sys_path = sys.path[:]
    try:
        namespace = {}
        exec(compile(startup, str(notebook_path), "exec"), namespace)
        assert namespace["ROOT"] == ROOT
        assert str(ROOT / "src") in sys.path
    finally:
        sys.path[:] = old_sys_path
