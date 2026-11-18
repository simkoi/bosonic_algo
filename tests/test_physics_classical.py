import numpy as np
from bosonic_algo.physics import (shin_metiu_potential,shin_metiu_forces,
                                  coordinate_lengths,initial_fock_coefficients)
from bosonic_algo.config import MODEL_DEFAULTS,normalized_for_method
from bosonic_algo.classical import run_classical


def test_forces_are_derivatives():
    r,R=0.21,-1.3
    eps=1e-5
    fR,fr=shin_metiu_forces(r,R,MODEL_DEFAULTS)
    assert np.isclose(fR,-(shin_metiu_potential(r,R+eps,MODEL_DEFAULTS)-shin_metiu_potential(r,R-eps,MODEL_DEFAULTS))/(2*eps))
    assert np.isclose(fr,-(shin_metiu_potential(r+eps,R,MODEL_DEFAULTS)-shin_metiu_potential(r-eps,R,MODEL_DEFAULTS))/(2*eps))


def test_small_classical_reference(tiny_config):
    ce,cR,_=initial_fock_coefficients(tiny_config)
    res=run_classical(tiny_config,ce,cR)
    assert res['psi_t'].shape==(2,1,2,2)
    assert np.allclose(res['norm'],1.,atol=1e-9)
    assert np.array_equal(res['snapshot_indices'],[0,2])
    assert res['mean_nph'].sum()==0


def test_small_classical_cavity(tiny_config):
    tiny_config['cavity'].update({"enabled":True,"cutoff_ph":2,"lambda_alpha":0.01})
    ce,cR,_=initial_fock_coefficients(tiny_config)
    res=run_classical(tiny_config,ce,cR)
    assert res['psi_t'].shape==(2,2,2,2)
    assert np.allclose(res['norm'],1.,atol=1e-9)
    assert res['mean_nph'][-1]>=0


def test_cache_identity_keeps_snapshot_policy(tiny_config):
    old=normalized_for_method(tiny_config,"classical","exact",None,"init")
    tiny_config['output']['snapshot_every']=1
    newer=normalized_for_method(tiny_config,"classical","exact",None,"init")
    assert old!=newer
    assert 'tgifs' not in old['config']
