"""Deterministic tests for progress reporter (no quantum dependencies)."""
from bosonic_algo.progress import ProgressReporter
import pytest


def test_progress_reports_elapsed_eta_and_checkpoints(capsys):
    now = [100.]
    def clock():
        return now[0]

    p = ProgressReporter("tgifs", interval_seconds=5.0, clock=clock)
    p.log("Starting")
    p.start_steps(100, 0.0, 1400.0)
    now[0] += 1
    p.checkpoint(10, 100, 140)
    assert "10/100" not in capsys.readouterr().out
    now[0] += 5
    p.checkpoint(20, 100, 280)
    text = capsys.readouterr().out
    assert "[TGIFS]" in text
    assert "20/100" in text
    assert "ETA=" in text
    assert "t=280" in text
    now[0] += 1
    p.checkpoint(100, 100, 1400)
    assert "100/100" in capsys.readouterr().out


def test_progress_disabled_and_invalid_period(capsys):
    with pytest.raises(ValueError):
        ProgressReporter("tgifs", interval_seconds=0)
    p = ProgressReporter("classical", enabled=False)
    p.log("Silent")
    p.start_steps(1, 0, 1)
    p.checkpoint(1, 1, 1)
    assert capsys.readouterr().out == ""
