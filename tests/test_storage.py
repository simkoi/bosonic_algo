import numpy as np
import json
import pytest
from bosonic_algo.storage import ResultStore


def example():
    return {"times":np.array([0.,1.]),"mean_r":np.array([1.,2.]),
            "psi_t":np.ones((1,2,2),complex),"snapshot_indices":np.array([0]),
            "snapshot_times":np.array([0.])}


def inputs():
    return {"method":"tgifs","potential":"fitted","fit_id":"fit_a",
            "config":{"cavity":{"enabled":False},"tgifs":{"substeps":20}},
            "initial_hash":"123","code_fingerprint":"abcdef"}


def test_reuse_only_exact_identity(tmp_path):
    store=ResultStore(tmp_path)
    counter=[]
    def compute():counter.append(1);return example()
    a,m,hit=store.run_or_load('tgifs',inputs(),compute)
    assert not hit and len(counter)==1
    a,m,hit=store.run_or_load('tgifs',inputs(),compute)
    assert hit and len(counter)==1
    changed=inputs()
    changed['config']['tgifs']['substeps']=40
    a,m,hit=store.run_or_load('tgifs',changed,compute)
    assert not hit and len(counter)==2
    found=store.find_results(method='tgifs',fit_id='fit_a',include_cavity=False)
    assert len(found)==2
    assert len(store.find_results(method='tgifs',**{'config.tgifs.substeps':20}))==1


def test_corrupt_data_not_reused(tmp_path):
    store=ResultStore(tmp_path)
    data,meta,_=store.run_or_load('classical',{**inputs(),"method":"classical"},example)
    path=store.path('classical',meta['run_id'])/'observables.npz'
    np.savez_compressed(path,times=np.array([23.]))
    with pytest.raises(ValueError,match='checksum'):
        store.run_or_load('classical',{**inputs(),"method":"classical"},example)


def test_incomplete_cache_not_reused(tmp_path):
    store=ResultStore(tmp_path)
    _,meta,_=store.run_or_load('tgifs',inputs(),example)
    p=store.path('tgifs',meta['run_id'])/'metadata.json'
    content=json.loads(p.read_text())
    content['status']='running'
    p.write_text(json.dumps(content))
    with pytest.raises(ValueError,match='incomplete'):
        store.run_or_load('tgifs',inputs(),example)
