import numpy as np
import pytest
from bosonic_algo.fourier import (FourierFit,save_fit,load_fit,build_mvec,
                                  generate_fits,list_fits)
from bosonic_algo.config import MODEL_DEFAULTS


def sample_fit():
    return FourierFit(np.array([[1,0],[1,-1]]), np.array([[.2,0.],[.2,-.1]]),
                      np.array([.4,.2]),np.array([.02,-.1]),0.03,
                      {"model":MODEL_DEFAULTS,"fourier_box":[20,80]})


def test_fit_roundtrip_and_tamper(tmp_path):
    fit=sample_fit()
    folder=save_fit(fit,tmp_path)
    assert folder.name==fit.fit_id
    again=load_fit(fit.fit_id,tmp_path)
    assert again.checksum==fit.checksum
    assert np.allclose(again.potential([0,1],[1,2]),fit.potential([0,1],[1,2]))
    assert save_fit(fit,tmp_path)==folder
    import json
    p=folder/'metadata.json'
    metadata=json.loads(p.read_text())
    metadata['checksum']='invalid'
    p.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='checksum'):load_fit(fit.fit_id,tmp_path)


def test_fourier_force_finite_difference():
    fit=sample_fit()
    r,R=.3,-2.
    eps=1e-6
    force_R,force_r=fit.forces(r,R)
    assert np.isclose(force_R,-(fit.potential(r,R+eps)-fit.potential(r,R-eps))/(2*eps))
    assert np.isclose(force_r,-(fit.potential(r+eps,R)-fit.potential(r-eps,R))/(2*eps))


def test_generate_small_support_only_no_dynamics(tmp_path):
    cfg={"orders":[[2,2]],"boxes":[[20.,30.]],"top_n":[2],
         "raw_v_rmse_max":100.,"grid":{"r_min":-2.,"r_max":2.,"r_step":1.,
                                         "R_min":-2.,"R_max":2.,"R_step":1.}}
    rows=generate_fits(cfg,root=tmp_path)
    saved=[x for x in rows if x['status']=='saved']
    assert len(saved)==1
    fit=load_fit(saved[0]['fit_id'],tmp_path)
    assert fit.A.size==2
    assert list_fits(tmp_path)[0]['n_modes']==2
    assert not list(tmp_path.rglob('psi_t*.npz'))


def test_half_lattice_includes_mixed_signs():
    modes=build_mvec((3,3))
    assert (1,-1) in map(tuple,modes)
    assert (1,1) in map(tuple,modes)
    assert (-1,1) not in map(tuple,modes)


def test_rejected_raw_fits_have_separate_uniform_summary(tmp_path):
    import pandas as pd
    cfg = {
        "orders": [[2, 2]],
        "boxes": [[20.0, 30.0]],
        "top_n": [2],
        # Force rejection independently of the actual positive RMS error.
        "raw_v_rmse_max": -1.0,
        "grid": {
            "r_min": -2.0, "r_max": 2.0, "r_step": 1.0,
            "R_min": -2.0, "R_max": 2.0, "R_step": 1.0,
        },
    }
    rows = generate_fits(cfg, root=tmp_path)
    assert rows == []
    raw_df = pd.read_csv(tmp_path / "raw_fit_summary.csv")
    assert len(raw_df) == 1
    assert raw_df.iloc[0]["status"] == "rejected"
    assert np.isfinite(raw_df.iloc[0]["v_rmse"])
    assert np.isfinite(raw_df.iloc[0]["force_combined_rmse"])
    assert not raw_df.isna().any().any()
    saved_df = pd.read_csv(tmp_path / "index.csv")
    assert saved_df.empty
    assert "fit_id" in saved_df.columns


def test_saved_and_rejected_fit_tables_do_not_mix_columns(tmp_path):
    import pandas as pd
    cfg = {
        "orders": [[2, 2]],
        "boxes": [[20.0, 30.0]],
        "top_n": [2],
        "raw_v_rmse_max": 100.0,
        "grid": {
            "r_min": -2.0, "r_max": 2.0, "r_step": 1.0,
            "R_min": -2.0, "R_max": 2.0, "R_step": 1.0,
        },
    }
    rows = generate_fits(cfg, root=tmp_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "saved"
    assert rows[0]["fit_id"]
    assert all(np.isfinite(row["force_combined_rmse"]) for row in rows)
    saved_df = pd.read_csv(tmp_path / "index.csv")
    raw_df = pd.read_csv(tmp_path / "raw_fit_summary.csv")
    assert len(saved_df) == 1
    assert len(raw_df) == 1
    assert raw_df.iloc[0]["status"] == "passed"
    assert not saved_df.isna().any().any()
    assert not raw_df.isna().any().any()
