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


def test_runtime_progress_and_baseline_flags_do_not_invalidate_results(tiny_config, tmp_path):
    from bosonic_algo.config import dynamics_config, normalized_for_method
    from bosonic_algo.storage import result_identity, source_fingerprint
    original = dynamics_config(tiny_config)
    modified = dynamics_config({
        **tiny_config,
        "runtime": {
            "run_classical_exact": True,
            "run_classical_fitted": True,
            "run_tgifs": False,
            "show_progress": False,
            "progress_interval_seconds": 2.5,
        },
    })
    for method in ("classical", "tgifs"):
        a = normalized_for_method(original, method, "fitted", "fit_checksum", "initial_hash")
        b = normalized_for_method(modified, method, "fitted", "fit_checksum", "initial_hash")
        assert a == b
        assert "runtime" not in a["config"]
        assert result_identity(a) == result_identity(b)
        assert source_fingerprint(method)
    assert source_fingerprint("classical") != source_fingerprint("tgifs")
    with pytest.raises(ValueError, match="Unknown propagation method"):
        source_fingerprint("other")


def test_short_classical_run_with_progress_and_cache(tiny_config, tmp_path, capsys):
    settings = dict(tiny_config)
    settings["runtime"] = {"progress_interval_seconds": 0.0001}
    a, meta, hit = run_or_load_classical(settings, result_root=tmp_path)
    assert not hit
    output = capsys.readouterr().out
    assert "Starting propagation" in output
    assert "Potential matrix construction" in output
    assert "Saved run" in output
    assert "substeps" in output
    b, loaded, hit = run_or_load_classical(settings, result_root=tmp_path)
    assert hit and loaded["run_id"] == meta["run_id"]
    assert "Cache hit" in capsys.readouterr().out
    assert np.allclose(a["mean_R"], b["mean_R"])
