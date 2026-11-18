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
