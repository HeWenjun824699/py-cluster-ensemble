import pytest
import numpy as np
from pce.consensus import (
    cspa, mcla, hgpa, ptaal, ptacl, ptasl, ptgp, lwea, lwgp, drec, usenc, celta,
    ecpcs_hc, ecpcs_mc, spce, trce, cdkm, mdecbg, mdechc, mdecsc, eccms, gtlec,
    kcc_uc, kcc_uh, ceam, space, cdec, icsc, dcc
)

ALL_CONSENSUS_METHODS = [
    cspa, mcla, hgpa, ptaal, ptacl, ptasl, ptgp, lwea, lwgp, drec, usenc, celta,
    ecpcs_hc, ecpcs_mc, spce, trce, cdkm, mdecbg, mdechc, mdecsc, eccms, gtlec,
    kcc_uc, kcc_uh, ceam, space, cdec, icsc, dcc
]


@pytest.mark.parametrize("consensus_func", ALL_CONSENSUS_METHODS)
def test_consensus_methods_basic(consensus_func, base_partitions, synthetic_data):
    X, Y = synthetic_data
    n_samples = X.shape[0]
    n_classes = len(np.unique(Y))
    
    # Common parameters
    kwargs = {
        'BPs': base_partitions,
        'Y': Y,
        'nClusters': n_classes,
        'nBase': 2,
        'nRepeat': 2,
        'seed': 2026
    }
    
    # Specific parameter adjustments
    if consensus_func.__name__ == 'lwea' or consensus_func.__name__ == 'lwgp':
        kwargs['theta'] = 10
    
    # Some methods might need explicit nInnerRepeat if they use it (e.g. cdkm)
    if consensus_func.__name__ == 'cdkm':
         kwargs['nInnerRepeat'] = 2

    try:
        labels_list, time_list = consensus_func(**kwargs)
        
        assert len(labels_list) == 2
        assert len(time_list) == 2
        assert labels_list[0].shape == (n_samples,)
        # Check if labels are integers
        assert np.issubdtype(labels_list[0].dtype, np.integer) or np.issubdtype(labels_list[0].dtype, np.floating)
        
    except Exception as e:
        pytest.fail(f"Method {consensus_func.__name__} failed with error: {e}")
