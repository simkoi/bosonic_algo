"""Public workflow API: load a saved fit, then run/load classical or TGIFS."""
from __future__ import annotations
from pathlib import Path
from .config import dynamics_config, normalized_for_method
from .physics import initial_fock_coefficients, initial_hash
from .fourier import load_fit, FourierFit
from .classical import run_classical
from .tgifs import run_tgifs
from .storage import ResultStore,source_fingerprint
from .progress import ProgressReporter
from time import perf_counter


def _load_verified_fit(fit_id,fit_root,cfg):
    if fit_id is None:
        raise ValueError("A saved fit_id is required for this calculation")
    fit=load_fit(fit_id,fit_root)
    if fit.metadata.get("model")!=cfg["model"]:
        raise ValueError("Saved Fourier fit was made using different physical model parameters")
    return fit


def _progress(cfg: dict, method: str) -> ProgressReporter:
    runtime = cfg["runtime"]
    return ProgressReporter(
        method,
        enabled=runtime["show_progress"],
        interval_seconds=float(runtime["progress_interval_seconds"]),
    )


def run_or_load_classical(overrides: dict | None = None, *, potential="exact",
                          fit_id=None, fit_root="data/fits",
                          result_root="results", force=False):
    """Load or propagate classical dynamics using immutable saved fits."""
    cfg = dynamics_config(overrides)
    if potential not in ("exact", "fitted"):
        raise ValueError("Unknown classical potential")
    progress = _progress(cfg, "CLASSICAL")
    progress.log(f"Preparing {potential} molecular dynamics...")
    started = perf_counter()
    fit = _load_verified_fit(fit_id, fit_root, cfg) if potential == "fitted" else None
    cr, cR, captured = initial_fock_coefficients(cfg)
    progress.stage("Initial state and fit loading", started)
    identity = normalized_for_method(
        cfg, "classical", potential, fit.checksum if fit else None,
        initial_hash(cr, cR),
    )
    identity.update({
        "fit_id": fit.fit_id if fit else None,
        "code_fingerprint": source_fingerprint("classical"),
    })
    store = ResultStore(result_root)
    return store.run_or_load(
        "classical", identity,
        lambda: run_classical(
            cfg, cr, cR, potential=potential, fit=fit, progress=progress
        ), force=force, progress=progress,
    )


def run_or_load_tgifs(fit_id: str, overrides: dict | None = None, *,
                      fit_root="data/fits", result_root="results", force=False):
    """Load or propagate unified 2D/3D TGIFS molecular dynamics."""
    cfg = dynamics_config(overrides)
    progress = _progress(cfg, "TGIFS")
    progress.log(f"Preparing Fourier fit {fit_id} and initial state...")
    started = perf_counter()
    fit = _load_verified_fit(fit_id, fit_root, cfg)
    cr, cR, captured = initial_fock_coefficients(cfg)
    progress.stage("Initial state and fit loading", started)
    identity = normalized_for_method(
        cfg, "tgifs", "fitted", fit.checksum, initial_hash(cr, cR)
    )
    identity.update({
        "fit_id": fit.fit_id,
        "code_fingerprint": source_fingerprint("tgifs"),
    })
    store = ResultStore(result_root)
    return store.run_or_load(
        "tgifs", identity,
        lambda: run_tgifs(cfg, fit, cr, cR, progress=progress),
        force=force, progress=progress,
    )


def selected_fit_classical_validation(fit_ids:list[str],overrides:dict|None=None, *,fit_root="data/fits",result_root="results"):
    """Only explicitly selected fit IDs are dynamically validated."""
    if not fit_ids:
        return []
    exact=run_or_load_classical(overrides,potential="exact",fit_root=fit_root,result_root=result_root)
    runs=[]
    for fit_id in fit_ids:
        fitted=run_or_load_classical(overrides,potential="fitted",fit_id=fit_id,
                                     fit_root=fit_root,result_root=result_root)
        runs.append({"fit_id":fit_id,"exact":exact,"fitted":fitted})
    return runs
