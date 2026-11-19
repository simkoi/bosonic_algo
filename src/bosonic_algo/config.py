"""Central defaults and validation for reproducible Shin--Metiu calculations."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import numpy as np

MODEL_DEFAULTS = {
    "hbar": 1.0, "me": 1.0, "M": 1836.0,
    "L_SM": 19.0, "Z_nuc": 1.0, "Rl": 3.1, "Rr": 4.0, "Rf": 5.0,
    "r0": 0.0, "R0": 0.0, "omega": 0.025, "Omega": 0.0025,
}
INITIAL_DEFAULTS = {
    "bo_R": -4.0, "bo_state_index": 1, "n_dvr": 150,
    "dvr_half_width": 18.5, "nuclear_center": -4.0,
    "nuclear_beta": 2.85, "nuclear_grid_min": -15.0,
    "nuclear_grid_max": 15.0, "nuclear_grid_points": 4000,
}
DYNAMICS_DEFAULTS = {
    "model": MODEL_DEFAULTS, "initial": INITIAL_DEFAULTS,
    "basis": {"Nr": 16, "NR": 128, "Nquad_r": 200, "Nquad_R": 200},
    "time": {"start": 0.0, "stop": 1400.0, "points": 71},
    "cavity": {"enabled": False, "include_dse": True,
               "omega_alpha": 0.1, "lambda_alpha": 0.005, "cutoff_ph": 8},
    "tgifs": {"substeps": 20, "kinetic": "gaussian", "coeff_tol": 1e-12},
    "output": {"snapshot_every": 10, "save_last": True},
    "runtime": {"run_classical_exact": False, "run_classical_fitted": False,
                "run_tgifs": True, "show_progress": True,
                "progress_interval_seconds": 15.0},
}
FOURIER_DEFAULTS = {
    "model": MODEL_DEFAULTS,
    "basis": "half_lattice", "orders": [[14, 14]],
    "boxes": [[20.0, 80.0], [30.0, 80.0], [40.0, 80.0]],
    "ridge_lambdas": [1e-4], "order_strengths": [10.0],
    "order_powers": [1.0], "top_n": [50],
    "selection_methods": ["force_scale"], "refit_ridge_multiplier": 1.0,
    "raw_v_rmse_max": 0.01,
    "grid": {"r_min": -30.0, "r_max": 30.0, "r_step": 1.0,
             "R_min": -6.0, "R_max": 6.0, "R_step": 1.0},
    "diagnostic_dt": 10.0, "coeff_tol": 1e-12,
    # Explicit selection, never inferred from best-fit sorting.
    "selected_for_dynamics": [],
}


def merge_defaults(defaults: dict, overrides: dict | None) -> dict:
    out = deepcopy(defaults)
    for key, value in (overrides or {}).items():
        if key not in out:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = merge_defaults(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def dynamics_config(overrides: dict | None = None) -> dict:
    cfg = merge_defaults(DYNAMICS_DEFAULTS, overrides)
    b, t, cav, g = cfg["basis"], cfg["time"], cfg["cavity"], cfg["tgifs"]
    if min(int(b[k]) for k in ("Nr", "NR", "Nquad_r", "Nquad_R")) < 1:
        raise ValueError("All Fock and quadrature sizes must be positive")
    if int(t["points"]) < 1 or float(t["stop"]) < float(t["start"]):
        raise ValueError("Invalid time grid")
    if int(g["substeps"]) < 1 or g["kinetic"] not in ("gaussian", "matrix"):
        raise ValueError("Invalid TGIFS evolution settings")
    if cav["enabled"] and (float(cav["omega_alpha"]) <= 0 or int(cav["cutoff_ph"]) < 2):
        raise ValueError("A cavity requires positive frequency and cutoff >= 2")
    if cav["enabled"] and int(cav["cutoff_ph"]).bit_count() != 1:
        raise ValueError("Bosonic-Qiskit photon cutoff must be a power of two")
    if cfg["output"]["snapshot_every"] is not None and int(cfg["output"]["snapshot_every"]) < 1:
        raise ValueError("snapshot_every must be positive or None")
    runtime = cfg["runtime"]
    if float(runtime["progress_interval_seconds"]) <= 0:
        raise ValueError("progress_interval_seconds must be positive")
    if any(not isinstance(runtime[k], bool) for k in
           ("run_classical_exact", "run_classical_fitted", "run_tgifs", "show_progress")):
        raise ValueError("Runtime switches must be booleans")
    if any(float(cfg["model"][k]) <= 0 for k in ("hbar", "me", "M", "omega", "Omega", "Rl", "Rr", "Rf")):
        raise ValueError("Invalid physical parameters")
    return cfg


def times_from_config(cfg: dict) -> np.ndarray:
    t = cfg["time"]
    return np.linspace(t["start"], t["stop"], int(t["points"]), dtype=float)


def read_yaml(path: str | Path) -> dict:
    import yaml
    with Path(path).open(encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError("Configuration YAML must contain a mapping")
    return value


def normalized_for_method(cfg: dict, method: str, potential: str, fit_checksum: str | None, initial_hash: str) -> dict:
    """Only meaningful simulation inputs contribute to a cache key."""
    c = deepcopy(cfg)
    # Runtime switches and reporting settings must not invalidate cached trajectories.
    c.pop("runtime", None)
    # Snapshot/output policy is part of the cache key to avoid returning wrong snapshots.
    if not c["cavity"]["enabled"]:
        c["cavity"] = {"enabled": False}
    elif not c["cavity"]["include_dse"]:
        c["cavity"].pop("include_dse", None)
    if method == "classical":
        c.pop("tgifs", None)
    else:
        if potential != "fitted":
            raise ValueError("TGIFS requires a saved Fourier fit")
    if potential == "exact":
        fit_checksum = None
    return {"method": method, "potential": potential, "config": c,
            "fit_checksum": fit_checksum, "initial_hash": initial_hash}
