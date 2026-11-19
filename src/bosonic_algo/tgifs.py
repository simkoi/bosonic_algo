"""Unified cavity-free / cavity-coupled TGIFS Bosonic-Qiskit engine.

Fourier fitting is deliberately absent: a verified FourierFit is mandatory.
Bosonic-Qiskit and Qiskit are imported lazily so classical tools work without them.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.linalg import expm
from .config import times_from_config
from .fourier import FourierFit
from .physics import coordinate_lengths, x_fock, p_fock


def run_tgifs(cfg: dict, fit: FourierFit, c_e: np.ndarray, c_R: np.ndarray) -> dict[str,np.ndarray]:
    try:
        import qiskit
        import bosonic_qiskit
        from qiskit.circuit.library import UnitaryGate
        from qiskit.quantum_info import Statevector
    except ImportError as exc:
        raise RuntimeError("TGIFS requires qiskit and bosonic_qiskit; install the quantum extra") from exc
    b,m,cav,alg,out=cfg["basis"],cfg["model"],cfg["cavity"],cfg["tgifs"],cfg["output"]
    Nr,NR=int(b["Nr"]),int(b["NR"])
    if c_e.shape!=(Nr,) or c_R.shape!=(NR,):
        raise ValueError("Initial-state vector does not match Fock basis")
    if (Nr & (Nr-1)) or (NR & (NR-1)):
        raise ValueError("Bosonic-Qiskit matter cutoffs must be powers of two")
    ph_enabled=bool(cav["enabled"])
    Nph=int(cav["cutoff_ph"]) if ph_enabled else 1
    lr,lR=coordinate_lengths(m)
    times=times_from_config(cfg)
    substeps=int(alg["substeps"])
    sub_dt=(times[1]-times[0])/substeps if len(times)>1 else 0.0
    mu=fit.mu_phys
    phi0=mu[:,0]*m["R0"]+mu[:,1]*m["r0"]
    # V=C0 + Aeff cos(k.X) + Beff sin(k.X), k=(mu_R*lR,mu_r*lr).
    Aeff=fit.A*np.cos(phi0)+fit.B*np.sin(phi0)
    Beff=fit.B*np.cos(phi0)-fit.A*np.sin(phi0)
    kappas=0.5*mu*np.array([lR,lr])[None,:]

    qR=bosonic_qiskit.QumodeRegister(1,int(np.log2(NR)),name="Rmode")
    qr=bosonic_qiskit.QumodeRegister(1,int(np.log2(Nr)),name="rmode")
    qph=bosonic_qiskit.QumodeRegister(1,int(np.log2(Nph)),name="phmode") if ph_enabled else None
    anc=qiskit.QuantumRegister(1,name="anc")
    def circuit():
        registers=[qR,qr]+([qph] if ph_enabled else [])+[anc]
        try:return bosonic_qiskit.CVCircuit(*registers,force_parameterized_unitary_gate=False)
        except TypeError:return bosonic_qiskit.CVCircuit(*registers)

    def cached_controlled_d(c, alpha, mode):
        cache=c.__dict__.setdefault("_controlled_displacement_cache",{})
        key=(tuple(mode),complex(alpha))
        if key not in cache:
            c.cv_c_d(alpha,mode,anc[0])
            cache[key]=c.data[-1].operation
        else:
            try:c.append(cache[key],[*mode,anc[0]],copy=False)
            except TypeError:c.append(cache[key],[*mode,anc[0]])

    def wx(c, k):
        c.ry(-np.pi/2,anc[0])
        for kk,mode in zip(k,(qR[0],qr[0])):
            if abs(kk)>1e-14:
                cached_controlled_d(c,1j*kk/np.sqrt(2),mode)
        c.ry(np.pi/2,anc[0])

    def utheta(c,k,theta):
        wx(c,-np.asarray(k))
        c.rz(-2*theta,anc[0])
        wx(c,+np.asarray(k))

    hyz=UnitaryGate(np.array([[1,-1j],[1j,-1]],complex)/np.sqrt(2),label="Hyz")
    def gcos(c,k,theta):
        utheta(c,-k,theta)
        utheta(c,+k,theta)

    def gsin(c,k,theta):
        c.append(hyz,[anc[0]])
        utheta(c,-k,-theta)
        utheta(c,+k,+theta)
        c.append(hyz,[anc[0]])

    def potential(c,dt):
        for k,a,bcoef in zip(kappas,Aeff,Beff):
            if abs(a)>alg["coeff_tol"]:gcos(c,k,-a*dt/2)
            if abs(bcoef)>alg["coeff_tol"]:gsin(c,k,-bcoef*dt/2)

    def free_kinetic(c,dt,mass,l,mode,N):
        if alg["kinetic"]=="matrix":
            P=p_fock(N)
            U=expm(-1j*(m["hbar"]*dt/(2*mass*l*l))*(P@P))
            c.cv_gate_from_matrix(U,qumodes=mode,label="kinetic")
        else:
            s=m["hbar"]*dt/(mass*l*l)
            phi=np.arctan(s/2)
            squeeze=-np.arcsinh(s/2)
            alpha=-phi/2-np.pi/4
            beta=-phi/2+np.pi/4
            c.cv_r(beta,mode)
            if abs(squeeze)>1e-15:c.cv_sq(squeeze,mode)
            c.cv_r(alpha,mode)

    def sum_corrected(c,chi,source,target):
        """Preserve ordering fix used by the source cavity notebook."""
        c.cv_sum(chi,source,target)
        instruction=c.data.pop()
        try:c.append(instruction.operation,[*target,*source],copy=False)
        except TypeError:c.append(instruction.operation,[*target,*source])

    def xx(c,chi,source,target):
        if abs(chi)<1e-15:return
        c.cv_r(np.pi/2,target)
        sum_corrected(c,chi,source,target)
        c.cv_r(-np.pi/2,target)

    def position_quadratic(c,s,mode):
        if abs(s)<1e-15:return
        c.cv_r(np.pi/2,mode)
        free_kinetic(c,s,1.,1.,mode,len(c_e) if mode==qr[0] else NR)
        c.cv_r(-np.pi/2,mode)

    def photon_matter(c,dt):
        pref=dt*cav["lambda_alpha"]*np.sqrt(cav["omega_alpha"])
        c.cv_r(np.pi/2,qph[0])
        sum_corrected(c,pref*lR,qR[0],qph[0])
        sum_corrected(c,-pref*lr,qr[0],qph[0])
        c.cv_r(-np.pi/2,qph[0])
        d0=m["R0"]-m["r0"]
        if abs(d0)>1e-14:c.cv_d(-1j*pref*d0/np.sqrt(2),qph[0])

    def self_energy(c,dt):
        scale=dt*cav["lambda_alpha"]**2
        d0=m["R0"]-m["r0"]
        position_quadratic(c,scale*lR*lR,qR[0])
        position_quadratic(c,scale*lr*lr,qr[0])
        xx(c,-scale*lR*lr,qR[0],qr[0])
        if abs(d0)>1e-14:
            c.cv_d(-1j*scale*d0*lR/np.sqrt(2),qR[0])
            c.cv_d(1j*scale*d0*lr/np.sqrt(2),qr[0])

    def substep(c,dt):
        free_kinetic(c,dt/2,m["M"],lR,qR[0],NR)
        free_kinetic(c,dt/2,m["me"],lr,qr[0],Nr)
        if ph_enabled:c.cv_r(-cav["omega_alpha"]*dt/2,qph[0])
        potential(c,dt)
        if ph_enabled:
            photon_matter(c,dt)
            if cav["include_dse"]:self_energy(c,dt)
            c.cv_r(-cav["omega_alpha"]*dt/2,qph[0])
        free_kinetic(c,dt/2,m["M"],lR,qR[0],NR)
        free_kinetic(c,dt/2,m["me"],lr,qr[0],Nr)

    step=circuit()
    substep(step,sub_dt)
    init=circuit()
    init.initialize([1.0,0.0],anc)
    init.cv_initialize(c_R,qR[0])
    init.cv_initialize(c_e,qr[0])
    if ph_enabled:
        ph=np.zeros(Nph,complex)
        ph[0]=1.
        init.cv_initialize(ph,qph[0])
    state=Statevector.from_instruction(init)
    r_op=m["r0"]*np.eye(Nr)+lr*x_fock(Nr)
    R_op=m["R0"]*np.eye(NR)+lR*x_fock(NR)
    r2=r_op@r_op;R2=R_op@R_op
    names=("p0","mean_r","sigma_r","mean_R","sigma_R","mean_nph",
           "p_ph_vacuum","photon_edge_pop","leak_r","leak_R")
    traces={key:[] for key in names}
    snaps=[];snap_indices=[]
    every=out["snapshot_every"]
    for i in range(len(times)):
        if i%10==0:
            print("step ",i)
        if i:
            for _ in range(substeps):state=state.evolve(step)
        full=np.asarray(state.data).reshape((2,Nph,Nr,NR) if ph_enabled else (2,Nr,NR))
        psi=full[0]
        p0=float(np.vdot(psi,psi).real)
        if p0<=0:raise RuntimeError("TGIFS ancilla-|0> branch vanished")
        psi=psi/np.sqrt(p0)
        psi_ph=psi if ph_enabled else psi[None,:,:]
        mr=float(np.einsum('pra,rs,psa->',psi_ph.conj(),r_op,psi_ph,optimize=True).real)
        mR=float(np.einsum('pra,ab,prb->',psi_ph.conj(),R_op,psi_ph,optimize=True).real)
        r2mean=float(np.einsum('pra,rs,psa->',psi_ph.conj(),r2,psi_ph,optimize=True).real)
        R2mean=float(np.einsum('pra,ab,prb->',psi_ph.conj(),R2,psi_ph,optimize=True).real)
        prob=np.abs(psi_ph)**2
        pops=np.sum(prob,axis=(1,2))
        vals={"p0":p0,"mean_r":mr,"sigma_r":np.sqrt(max(r2mean-mr*mr,0)),
              "mean_R":mR,"sigma_R":np.sqrt(max(R2mean-mR*mR,0)),
              "mean_nph":float(np.dot(np.arange(len(pops)),pops)),
              "p_ph_vacuum":float(pops[0]),"photon_edge_pop":float(pops[-1]) if ph_enabled else 0.,
              "leak_r":float(prob[:,Nr:,:].sum()),"leak_R":float(prob[:,:,NR:].sum())}
        for key in names:traces[key].append(vals[key])
        if every is not None and (i%int(every)==0 or (out["save_last"] and i==len(times)-1)):
            snaps.append(psi.copy());snap_indices.append(i)
    saved_shape=(Nph,Nr,NR) if ph_enabled else (Nr,NR)
    result={key:np.asarray(v) for key,v in traces.items()}
    result.update({"times":times,
                   "psi_t":np.stack(snaps) if snaps else np.empty((0,*saved_shape),complex),
                   "snapshot_indices":np.asarray(snap_indices,int),
                   "snapshot_times":times[snap_indices]})
    return result
