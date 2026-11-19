"""Dependency-free, time-based progress messages for long simulations.

This module deliberately stays outside numerical source fingerprints.
"""
from __future__ import annotations

from time import perf_counter


class ProgressReporter:
    """Report stages, elapsed wall time, and an ETA for propagation."""

    def __init__(self, label: str, *, interval_seconds: float = 15.0,
                 enabled: bool = True, clock=perf_counter):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.label = label.upper()
        self.interval_seconds = float(interval_seconds)
        self.enabled = bool(enabled)
        self._clock = clock
        self._start = clock()
        self._propagation_start = None
        self._last_report = None

    def now(self) -> float:
        return self._clock()

    def log(self, message: str) -> None:
        if self.enabled:
            print(f"[{self.label}] {message}", flush=True)

    def stage(self, name: str, started: float) -> None:
        now = self.now()
        self.log(f"{name}: {now - started:.2f}s (total elapsed {now - self._start:.2f}s)")

    def start_steps(self, total: int, first_time: float, last_time: float) -> None:
        self._propagation_start = self.now()
        self._last_report = self._propagation_start
        self.log(f"Starting propagation: {total} substeps, physical time "
                 f"{first_time:.6g} to {last_time:.6g}")

    def checkpoint(self, complete: int, total: int, physical_time: float,
                   *, force: bool = False) -> None:
        if not self.enabled or self._propagation_start is None:
            return
        now = self.now()
        if (not force and complete != total and
                now - self._last_report < self.interval_seconds):
            return
        elapsed = now - self._propagation_start
        remaining = (elapsed / complete * max(0, total - complete)
                     if complete else float("inf"))
        self.log(f"{complete}/{total} substeps ({100 * complete / total:.1f}%)"
                 f" | t={physical_time:.6g} | elapsed={elapsed:.1f}s"
                 f" | ETA={remaining:.1f}s")
        self._last_report = now
