import numpy as np
import pytest
from bosonic_algo.fourier import FourierFit,save_fit
from bosonic_algo.workflow import run_or_load_classical, run_or_load_tgifs, selected_fit_classical_validation
from bosonic_algo.config import MODEL_DEFAULTS


def test_dynamics_refuses_missing_fit(tiny_config,tmp_path):
    with pytest.raises(FileNotFoundError):
        run_or_load_tgifs('fit_missing',tiny_config,fit_root=tmp_path)
    assert selected_fit_classical_validation([],tiny_config,fit_root=tmp_path)==[]


def test_exact_classical_cache_independent_of_fit(tiny_config,tmp_path):
    kwargs={"result_root":tmp_path/'results'}
    a,m,hit=run_or_load_classical(tiny_config,**kwargs)
    assert not hit
    b,meta,hit=run_or_load_classical(tiny_config,**kwargs)
    assert hit and meta['run_id']==m['run_id']
    assert np.allclose(a['mean_R'],b['mean_R'])


def test_fitted_classical_loads_saved_fit_and_reuses(tiny_config,tmp_path):
    fit=FourierFit(np.array([[1,0]]),np.array([[.2,0.]]),np.array([.1]),np.array([0.]),0.2,
                   {'model':MODEL_DEFAULTS})
    save_fit(fit,tmp_path/'fits')
    kwargs={'potential':'fitted','fit_id':fit.fit_id,'fit_root':tmp_path/'fits','result_root':tmp_path/'results'}
    a,m,hit=run_or_load_classical(tiny_config,**kwargs)
    assert not hit
    b,n,hit=run_or_load_classical(tiny_config,**kwargs)
    assert hit and m['run_id']==n['run_id']
    assert np.allclose(a['mean_R'],b['mean_R'])
    other=dict(tiny_config)
    other['model']=dict(MODEL_DEFAULTS, Rf=5.5)
    with pytest.raises(ValueError,match='different physical model'):
        run_or_load_classical(other,**kwargs)
