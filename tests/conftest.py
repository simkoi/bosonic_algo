import pytest
from bosonic_algo.config import dynamics_config, MODEL_DEFAULTS

@pytest.fixture
def tiny_config():
    return dynamics_config({
        "basis":{"Nr":2,"NR":2,"Nquad_r":16,"Nquad_R":16},
        "initial":{"n_dvr":10,"nuclear_grid_points":101},
        "time":{"start":0.,"stop":0.2,"points":3},
        "output":{"snapshot_every":2,"save_last":True},
        "tgifs":{"substeps":2},
    })
