"""Shared physical operators, Shin--Metiu potential, and initial conditions."""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.special import erf, eval_hermite, factorial
from numpy.polynomial.hermite import hermgauss


def coordinate_lengths(model: dict) -> tuple[float, float]:
    return (np.sqrt(model["hbar"] / (model["me"] * model["omega"])),
            np.sqrt(model["hbar"] / (model["M"] * model["Omega"])))


def erf_coulomb(x, sigma):
    x = np.asarray(x, dtype=float)
    a = np.abs(x)
    out = np.empty_like(a)
    mask = a > 1e-8
    out[mask] = erf(a[mask] / sigma) / a[mask]
    out[~mask] = 2 / (np.sqrt(np.pi) * sigma) * (1 - (a[~mask]/sigma)**2 / 3)
    return out


def erf_coulomb_prime(x, sigma):
    x = np.asarray(x, dtype=float)
    a = np.abs(x)
    out = np.empty_like(a)
    mask = a > 1e-5 * sigma
    z = a[mask] / sigma
    out[mask] = np.sign(x[mask]) * (2 * np.exp(-z*z) / (np.sqrt(np.pi)*sigma*a[mask]) - erf(z) / a[mask]**2)
    out[~mask] = -4 * x[~mask] / (3*np.sqrt(np.pi)*sigma**3)
    return out


def shin_metiu_potential(r, R, model: dict):
    r, R = np.asarray(r), np.asarray(R)
    L, Z = model["L_SM"], model["Z_nuc"]
    return (Z**2 / np.abs(R - L/2) + Z**2 / np.abs(R + L/2)
            - Z * erf_coulomb(r-R, model["Rf"])
            - Z * erf_coulomb(r-L/2, model["Rr"])
            - Z * erf_coulomb(r+L/2, model["Rl"]))


def shin_metiu_forces(r, R, model: dict):
    """Return (-dV/dR, -dV/dr), matching Fourier wavevector order (R,r)."""
    r, R = np.asarray(r), np.asarray(R)
    L, Z = model["L_SM"], model["Z_nuc"]
    dVdR = (-Z**2*np.sign(R-L/2)/np.abs(R-L/2)**2
             - Z**2*np.sign(R+L/2)/np.abs(R+L/2)**2
             + Z*erf_coulomb_prime(r-R, model["Rf"]))
    dVdr = -Z * (erf_coulomb_prime(r-R, model["Rf"])
                 + erf_coulomb_prime(r-L/2, model["Rr"])
                 + erf_coulomb_prime(r+L/2, model["Rl"]))
    return -dVdR, -dVdr


def x_fock(n: int) -> np.ndarray:
    a = np.sqrt(np.arange(1, n) / 2)
    return np.diag(a, 1) + np.diag(a, -1)


def p_fock(n: int) -> np.ndarray:
    a = np.sqrt(np.arange(1, n) / 2)
    return -1j*np.diag(a, 1) + 1j*np.diag(a, -1)


def kinetic_1d(n: int, mass: float, l: float, hbar: float = 1.0) -> np.ndarray:
    P = p_fock(n)
    return hbar**2/(2*mass*l**2) * (P@P)


def _ho_polynomials(n: int, x: np.ndarray) -> np.ndarray:
    return np.stack([np.pi**(-0.25)*eval_hermite(i, x)/np.sqrt(2**i * factorial(i))
                     for i in range(n)])


def potential_matrix(potential_function, cfg: dict) -> np.ndarray:
    """Gauss-Hermite projection into the (r,R)-ordered two-mode Fock basis."""
    b, m = cfg["basis"], cfg["model"]
    Nr, NR = int(b["Nr"]), int(b["NR"])
    lr, lR = coordinate_lengths(m)
    xr, wr = hermgauss(int(b["Nquad_r"]))
    xR, wR = hermgauss(int(b["Nquad_R"]))
    Br = _ho_polynomials(Nr, xr) * np.sqrt(wr)[None, :]
    BR = _ho_polynomials(NR, xR) * np.sqrt(wR)[None, :]
    r = m["r0"] + lr*xr
    R = m["R0"] + lR*xR
    values = np.asarray(potential_function(r[:,None], R[None,:]))
    if values.shape != (len(xr),len(xR)):
        raise ValueError("Potential function has wrong broadcasting shape")
    return np.einsum('ni,mi,aj,bj,ij->namb', Br, Br, BR, BR, values,
                     optimize=True).reshape(Nr*NR, Nr*NR)


def _ho_realspace(n, q, q0, l):
    x = (q-q0)/l
    return _ho_polynomials(n, x) * np.exp(-x*x/2)[None,:] / np.sqrt(l)


def initial_fock_coefficients(cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Second BO state at fixed R and nuclear Gaussian, as in the original notebooks."""
    b,m,init = cfg["basis"], cfg["model"],cfg["initial"]
    n = int(init["n_dvr"])
    dx = float(init["dvr_half_width"])/n
    inds = np.arange(-n,n+1)
    d = inds[:,None] - inds[None,:]
    d2 = np.empty(d.shape)
    np.fill_diagonal(d2, -np.pi**2/(3*dx**2))
    mask = d != 0
    d2[mask] = -2.0 * (-1.0)**d[mask] / (dx*d[mask])**2
    r_grid = inds*dx
    H = -(m["hbar"]**2)/(2*m["me"]) * d2 + np.diag(
        shin_metiu_potential(r_grid, float(init["bo_R"]), m))
    idx = int(init["bo_state_index"])
    _, v = eigh(H, subset_by_index=[idx,idx])
    psi_e = v[:,0].astype(complex)
    psi_e /= np.sqrt(np.trapezoid(np.abs(psi_e)**2,r_grid))
    lr,lR = coordinate_lengths(m)
    cr = np.trapezoid(_ho_realspace(int(b["Nr"]),r_grid,m["r0"],lr).conj()*psi_e[None,:], r_grid, axis=1)

    R_grid = np.linspace(init["nuclear_grid_min"], init["nuclear_grid_max"],int(init["nuclear_grid_points"]))
    psi_R = np.exp(-float(init["nuclear_beta"])*(R_grid-float(init["nuclear_center"]))**2)
    psi_R /= np.sqrt(np.trapezoid(np.abs(psi_R)**2,R_grid))
    cR = np.trapezoid(_ho_realspace(int(b["NR"]),R_grid,m["R0"],lR).conj()*psi_R[None,:], R_grid, axis=1)
    captured = {"electronic":float(np.vdot(cr,cr).real),"nuclear":float(np.vdot(cR,cR).real)}
    if min(captured.values()) <= 0:
        raise ValueError("Initial state not represented by the selected Fock basis")
    return cr/np.linalg.norm(cr),cR/np.linalg.norm(cR),captured


def initial_hash(cr: np.ndarray, cR: np.ndarray) -> str:
    import hashlib
    h = hashlib.sha256()
    for a in (cr,cR):
        a = np.ascontiguousarray(a,dtype="<c16")
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()
