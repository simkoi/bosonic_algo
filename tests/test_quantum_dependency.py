import importlib.util
import numpy as np
import pytest
from bosonic_algo.tgifs import run_tgifs
from bosonic_algo.fourier import FourierFit
from bosonic_algo.config import MODEL_DEFAULTS


def test_tgifs_is_optional_dependency(tiny_config):
    if importlib.util.find_spec('bosonic_qiskit') is not None and importlib.util.find_spec('qiskit') is not None:
        pytest.skip('Installed quantum engine should use integration tests')
    fit=FourierFit(np.array([[1,0]]),np.array([[.1,0]]),np.array([.2]),np.array([0.]),0.,
                   {'model':MODEL_DEFAULTS})
    with pytest.raises(RuntimeError,match='requires qiskit'):
        run_tgifs(tiny_config,fit,np.ones(2)/np.sqrt(2),np.ones(2)/np.sqrt(2))


def test_cutoff_edge_populations_measure_top_fock_levels():
    from bosonic_algo.tgifs import cutoff_edge_populations
    prob = np.zeros((1, 5, 6), dtype=float)
    prob[0, 4, 5] = 0.3
    prob[0, 0, 0] = 0.7
    edge_r, edge_R = cutoff_edge_populations(prob)
    assert edge_r == pytest.approx(0.3)
    assert edge_R == pytest.approx(0.3)
    assert cutoff_edge_populations(prob, edge_levels=1) == pytest.approx((0.3, 0.3))
    with pytest.raises(ValueError):
        cutoff_edge_populations(np.ones((5, 6)))
    with pytest.raises(ValueError):
        cutoff_edge_populations(prob, edge_levels=0)
