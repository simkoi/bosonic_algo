"""Single authoritative Fourier fitting workflow and immutable fit files."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from itertools import product
from typing import Any
import hashlib
import json
import os
import shutil
import tempfile
import numpy as np
from .config import FOURIER_DEFAULTS, merge_defaults
from .physics import shin_metiu_potential, shin_metiu_forces, coordinate_lengths

SCHEMA_VERSION = 1


def canonical_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _array_checksum(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode("utf-8"))
        h.update(str(a.dtype).encode("ascii"))
        h.update(str(a.shape).encode("ascii"))
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass
class FourierFit:
    """Coefficients are ordered as [R,r]; independent of any propagation engine."""
    mvec: np.ndarray
    mu_phys: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C0: float
    metadata: dict[str, Any]

    def __post_init__(self):
        self.mvec = np.asarray(self.mvec, dtype=np.int64)
        self.mu_phys = np.asarray(self.mu_phys, dtype=float)
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        self.C0 = float(self.C0)
        n = len(self.A)
        if self.mvec.shape != (n,2) or self.mu_phys.shape != (n,2) or self.B.shape != (n,):
            raise ValueError("Fourier coefficient arrays have inconsistent dimensions")
        for a in (self.A,self.B,self.mu_phys):
            if not np.all(np.isfinite(a)):
                raise ValueError("Nonfinite Fourier coefficient")

    @property
    def arrays(self) -> dict[str,np.ndarray]:
        return {"mvec":self.mvec,"mu_phys":self.mu_phys,"A":self.A,
                "B":self.B,"C0":np.asarray(self.C0,dtype=float)}

    @property
    def checksum(self) -> str:
        return _array_checksum(self.arrays)

    @property
    def fit_id(self) -> str:
        return "fit_" + self.checksum[:16]

    def potential(self, r, R):
        """Broadcast correctly, including scalar and already-broadcast 2D coordinates."""
        rr,RR = np.broadcast_arrays(np.asarray(r,dtype=float),np.asarray(R,dtype=float))
        p = RR.ravel()[:,None]*self.mu_phys[None,:,0] + rr.ravel()[:,None]*self.mu_phys[None,:,1]
        return (self.C0+np.cos(p)@self.A+np.sin(p)@self.B).reshape(rr.shape)

    def forces(self,r,R):
        rr,RR = np.broadcast_arrays(np.asarray(r,dtype=float),np.asarray(R,dtype=float))
        p = RR.ravel()[:,None]*self.mu_phys[None,:,0] + rr.ravel()[:,None]*self.mu_phys[None,:,1]
        derivative = np.sin(p)*self.A[None,:] - np.cos(p)*self.B[None,:]
        return ((derivative@self.mu_phys[:,0]).reshape(rr.shape),
                (derivative@self.mu_phys[:,1]).reshape(rr.shape))


def build_mvec(order, basis="half_lattice") -> np.ndarray:
    nR,nr = map(int,order)
    if min(nR,nr)<2:
        raise ValueError("Fourier order must be at least 2 on each axis")
    if basis == "half_lattice":
        modes = [(0,j) for j in range(1,nr)] + [(i,j) for i in range(1,nR) for j in range(-nr+1,nr)]
    elif basis == "positive_only":
        modes = [(i,j) for i in range(nR) for j in range(nr) if i>0 or j>0]
    else:
        raise ValueError(f"Unsupported Fourier basis: {basis}")
    return np.asarray(modes,dtype=int)


def _fit_on_support(mvec, order, box, grid, model, lam, strength, power, metadata) -> FourierFit:
    r,R = grid
    rr,RR = np.meshgrid(r,R,indexing="xy")
    mu = np.asarray(mvec)* (2*np.pi/np.asarray(box,dtype=float))[None,:]
    p = RR.ravel()[:,None]*mu[None,:,0]+rr.ravel()[:,None]*mu[None,:,1]
    design = np.column_stack((np.ones(p.shape[0]),np.cos(p),np.sin(p)))
    target = shin_metiu_potential(rr,RR,model).ravel()
    nR,nr = order
    order2 = (mvec[:,0]/max(nR-1,1))**2+(mvec[:,1]/max(nr-1,1))**2
    w = 1 + strength*order2**power
    reg = np.sqrt(lam*np.r_[0,w,w])
    # Augmented least squares, avoiding explicitly squaring the condition number.
    aug = np.vstack((design,np.diag(reg)))
    y = np.r_[target,np.zeros(design.shape[1])]
    coeff, *_ = np.linalg.lstsq(aug,y,rcond=None)
    n=len(mvec)
    return FourierFit(mvec,mu,coeff[1:n+1],coeff[n+1:],coeff[0],metadata)


def metrics(fit: FourierFit, grid: tuple[np.ndarray,np.ndarray], model: dict, dt: float, coeff_tol=1e-12) -> dict:
    r,R=grid
    rr,RR=np.meshgrid(r,R,indexing="xy")
    err=fit.potential(rr,RR)-shin_metiu_potential(rr,RR,model)
    fR,fr=fit.forces(rr,RR)
    true_R,true_r=shin_metiu_forces(rr,RR,model)
    eR=fR-true_R
    er=fr-true_r
    amp=np.hypot(fit.A,fit.B)
    mu=np.linalg.norm(fit.mu_phys,axis=1)
    angles = np.r_[np.abs(fit.A),np.abs(fit.B)] * (dt/2)
    return {
        "v_rmse":float(np.sqrt(np.mean(err**2))),
        "v_max_abs_error":float(np.max(np.abs(err))),
        "force_R_rmse":float(np.sqrt(np.mean(eR**2))),
        "force_r_rmse":float(np.sqrt(np.mean(er**2))),
        "force_combined_rmse":float(np.sqrt(0.5*(np.mean(eR**2)+np.mean(er**2)))),
        "n_modes":int(len(fit.A)),
        "max_amplitude":float(np.max(amp)) if len(amp) else 0.0,
        "sum_amplitude":float(np.sum(amp)),
        "max_force_scale":float(np.max(amp*mu)) if len(amp) else 0.0,
        "max_theta":float(np.max(angles)) if len(angles) else 0.0,
        "n_tgifs_terms":int(np.count_nonzero(np.abs(fit.A)>coeff_tol)+np.count_nonzero(np.abs(fit.B)>coeff_tol)),
    }


def save_fit(fit: FourierFit, root: str | Path) -> Path:
    """Write coeffs + complete metadata atomically; never overwrite an existing fit."""
    root=Path(root)
    root.mkdir(parents=True, exist_ok=True)
    folder = root / fit.fit_id
    if folder.exists():
        check=load_fit(fit.fit_id,root)
        if check.checksum!=fit.checksum:
            raise ValueError("Existing fit ID conflicts with coefficients")
        return folder
    temp=Path(tempfile.mkdtemp(prefix=".fit-",dir=root))
    try:
        np.savez_compressed(temp/"coefficients.npz", **fit.arrays)
        meta={"schema_version":SCHEMA_VERSION,"fit_id":fit.fit_id,
              "checksum":fit.checksum,"n_modes":len(fit.A),"parameters":fit.metadata}
        (temp/"metadata.json").write_text(json.dumps(meta,indent=2,sort_keys=True),encoding="utf-8")
        os.rename(temp,folder)
    finally:
        if temp.exists():shutil.rmtree(temp)
    return folder


def load_fit(fit_id: str, root: str | Path) -> FourierFit:
    folder=Path(root)/fit_id
    meta=json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    if meta.get("schema_version")!=SCHEMA_VERSION or meta.get("fit_id")!=fit_id:
        raise ValueError("Incompatible fit metadata")
    with np.load(folder/"coefficients.npz",allow_pickle=False) as z:
        fit=FourierFit(z["mvec"],z["mu_phys"],z["A"],z["B"],float(z["C0"]),meta["parameters"])
    if fit.checksum != meta.get("checksum") or fit.fit_id!=fit_id:
        raise ValueError("Fourier coefficient checksum mismatch")
    return fit


def list_fits(root: str | Path) -> list[dict]:
    items=[]
    for path in sorted(Path(root).glob("fit_*/metadata.json")):
        try:
            metadata=json.loads(path.read_text(encoding="utf-8"))
            items.append({"fit_id":metadata["fit_id"],"n_modes":metadata["n_modes"],
                          **metadata["parameters"].get("metrics",{})})
        except (ValueError,KeyError):
            continue
    return items


def generate_fits(overrides: dict | None=None, root: str | Path="data/fits") -> list[dict]:
    """Fit/screen/reduce/save all survivors; performs NO classical propagation."""
    cfg=merge_defaults(FOURIER_DEFAULTS,overrides)
    g=cfg["grid"]
    grid=(np.arange(g["r_min"],g["r_max"],g["r_step"]),
          np.arange(g["R_min"],g["R_max"],g["R_step"]))
    rows=[]
    for order,box,lam,strength,power in product(cfg["orders"],cfg["boxes"],cfg["ridge_lambdas"],cfg["order_strengths"],cfg["order_powers"]):
        mvec=build_mvec(order,cfg["basis"])
        base={"model":cfg["model"],"grid":g,"basis":cfg["basis"],"fourier_order":order,
              "fourier_box":box,"ridge_lambda":lam,"order_strength":strength,"order_power":power}
        raw=_fit_on_support(mvec,order,box,grid,cfg["model"],lam,strength,power,
                            {**base,"selection_method":"raw"})
        raw_metrics=metrics(raw,grid,cfg["model"],cfg["diagnostic_dt"],cfg["coeff_tol"])
        if raw_metrics["v_rmse"]>cfg["raw_v_rmse_max"]:
            rows.append({"fit_id":None,"raw_rmse":raw_metrics["v_rmse"],"status":"rejected",**base})
            continue
        for method,n in product(cfg["selection_methods"],cfg["top_n"]):
            if method=="amplitude":score=np.hypot(raw.A,raw.B)
            elif method=="force_scale":score=np.hypot(raw.A,raw.B)*np.linalg.norm(raw.mu_phys,axis=1)
            else:raise ValueError(f"Unknown selection method: {method}")
            if int(n)<1:raise ValueError("top_n must be positive")
            idx=np.argsort(score)[::-1][:min(int(n),len(score))]
            details={**base,"selection_method":method,"top_n":int(n),
                     "parent_fit_checksum":raw.checksum,
                     "refit_ridge_multiplier":cfg["refit_ridge_multiplier"]}
            reduced=_fit_on_support(raw.mvec[idx],order,box,grid,cfg["model"],lam*cfg["refit_ridge_multiplier"],strength,power,details)
            quality=metrics(reduced,grid,cfg["model"],cfg["diagnostic_dt"],cfg["coeff_tol"])
            reduced.metadata["metrics"]=quality
            folder=save_fit(reduced,root)
            rows.append({"fit_id":reduced.fit_id,"raw_rmse":raw_metrics["v_rmse"],
                         "status":"saved",**quality,"path":str(folder)})
    # A disposable listing; full metadata remains in each fit directory.
    import csv
    keys=list(dict.fromkeys(k for row in rows for k in row)) if rows else []
    if keys:
        with (Path(root)/"index.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=keys)
            writer.writeheader();writer.writerows(rows)
    return rows
