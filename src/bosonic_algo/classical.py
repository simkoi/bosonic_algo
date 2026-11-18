"""Matrix-free classical molecular and optional molecule-cavity evolution."""
from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import LinearOperator, expm_multiply
from .physics import coordinate_lengths, potential_matrix, shin_metiu_potential, x_fock, kinetic_1d
from .fourier import FourierFit
from .config import times_from_config


def run_classical(cfg: dict, c_e: np.ndarray, c_R: np.ndarray,
                  *, potential: str="exact", fit: FourierFit | None=None) -> dict[str,np.ndarray]:
    """Propagate exp(-iHt/hbar) with no TGIFS/Trotter approximation.

    Args:
        potential: 'exact' uses the original Shin-Metiu potential, 'fitted'
            uses a previously loaded FourierFit. Never performs fitting.
    """
    if potential not in ("exact","fitted"):
        raise ValueError("Classical potential must be 'exact' or 'fitted'")
    if potential=="fitted" and fit is None:
        raise ValueError("Fitted classical evolution requires a saved FourierFit")
    b,m,cav,out=cfg["basis"],cfg["model"],cfg["cavity"],cfg["output"]
    Nr,NR=int(b["Nr"]),int(b["NR"])
    if c_e.shape!=(Nr,) or c_R.shape!=(NR,):
        raise ValueError("Initial state length differs from basis")
    ph_enabled=bool(cav["enabled"])
    Nph=int(cav["cutoff_ph"]) if ph_enabled else 1
    lr,lR=coordinate_lengths(m)
    Tr=kinetic_1d(Nr,m["me"],lr,m["hbar"])
    TR=kinetic_1d(NR,m["M"],lR,m["hbar"])
    r_op=m["r0"]*np.eye(Nr)+lr*x_fock(Nr)
    R_op=m["R0"]*np.eye(NR)+lR*x_fock(NR)
    r2=r_op@r_op
    R2=R_op@R_op
    if potential=="exact":
        Vmat=potential_matrix(lambda r,R:shin_metiu_potential(r,R,m),cfg)
    else:
        Vmat=potential_matrix(fit.potential,cfg)
    Xph=x_fock(Nph) if ph_enabled else np.zeros((1,1))
    Eph=(cav["omega_alpha"]*(np.arange(Nph)+0.5)) if ph_enabled else np.zeros(1)
    coupling=float(cav["lambda_alpha"])*np.sqrt(float(cav["omega_alpha"])) if ph_enabled else 0.0
    dse=0.5*float(cav["lambda_alpha"])**2 if ph_enabled and cav["include_dse"] else 0.0

    def dipole(u):
        return u@R_op.T-r_op@u

    def dipole2(u):
        return u@R2.T+r2@u-2*r_op@u@R_op.T

    def H_dot(flat):
        psi=np.asarray(flat,dtype=complex).reshape(Nph,Nr,NR)
        computed=np.empty_like(psi)
        dip=np.empty_like(psi) if ph_enabled else None
        for ph in range(Nph):
            u=psi[ph]
            computed[ph]=Tr@u+u@TR.T+(Vmat@u.reshape(-1)).reshape(Nr,NR)+Eph[ph]*u
            if ph_enabled:
                dip[ph]=dipole(u)
                if dse:
                    computed[ph]+=dse*dipole2(u)
        if ph_enabled:
            computed+=coupling*(Xph@dip.reshape(Nph,-1)).reshape(Nph,Nr,NR)
        return computed.ravel()

    # Analytic trace of the full Hamiltonian for expm_multiply.
    trace_matter=NR*np.trace(Tr)+Nr*np.trace(TR)+np.trace(Vmat)
    trace_d2=Nr*np.trace(R2)+NR*np.trace(r2)-2*np.trace(R_op)*np.trace(r_op)
    trace_H=float(np.real(Nph*(trace_matter+dse*trace_d2)+Nr*NR*np.sum(Eph)))

    psi=np.zeros((Nph,Nr,NR),dtype=complex)
    psi[0]=np.outer(c_e,c_R)
    psi=psi.ravel()
    times=times_from_config(cfg)
    records={k:[] for k in ("mean_r","sigma_r","mean_R","sigma_R","mean_nph",
                            "p_ph_vacuum","photon_edge_pop","norm")}
    snapshots=[]
    indices=[]
    every=out["snapshot_every"]
    for i,t in enumerate(times):
        if i:
            dt=float(t-times[i-1]);scale=-1j*dt/m["hbar"]
            operator=LinearOperator((psi.size,psi.size),dtype=np.complex128,
                                    matvec=lambda v:scale*H_dot(v),
                                    rmatvec=lambda v:np.conj(scale)*H_dot(v))
            psi=expm_multiply(operator,psi,traceA=scale*trace_H)
        state=psi.reshape(Nph,Nr,NR)
        norm=float(np.vdot(psi,psi).real)
        rmean=float(np.einsum('pra,rs,psa->',state.conj(),r_op,state,optimize=True).real)
        Rmean=float(np.einsum('pra,ab,prb->',state.conj(),R_op,state,optimize=True).real)
        rmean2=float(np.einsum('pra,rs,psa->',state.conj(),r2,state,optimize=True).real)
        Rmean2=float(np.einsum('pra,ab,prb->',state.conj(),R2,state,optimize=True).real)
        pop=np.sum(np.abs(state)**2,axis=(1,2))
        vals={"mean_r":rmean,"sigma_r":np.sqrt(max(rmean2-rmean*rmean,0)),
              "mean_R":Rmean,"sigma_R":np.sqrt(max(Rmean2-Rmean*Rmean,0)),
              "mean_nph":float(np.dot(np.arange(Nph),pop)),
              "p_ph_vacuum":float(pop[0]),"photon_edge_pop":float(pop[-1]) if ph_enabled else 0.,
              "norm":norm}
        for key,val in vals.items():records[key].append(val)
        if every is not None and (i%int(every)==0 or (out["save_last"] and i==len(times)-1)):
            snapshots.append(state.copy())
            indices.append(i)
    result={k:np.asarray(v) for k,v in records.items()}
    result.update({"times":times,
                   "psi_t":np.stack(snapshots) if snapshots else np.empty((0,Nph,Nr,NR),complex),
                   "snapshot_indices":np.asarray(indices,int),
                   "snapshot_times":times[indices]})
    return result
