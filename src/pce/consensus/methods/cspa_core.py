import numpy as np
from .CSCSPA_HGPA_MCLA_JMLR_2003.clstoclbs import clstoclbs
from .CSCSPA_HGPA_MCLA_JMLR_2003.checks import checks
from .CSCSPA_HGPA_MCLA_JMLR_2003.metis import metis


def cspa_core(cls, k=None):
    """
    Performs CSPA for CLUSTER ENSEMBLES

    Strict translation of cspa.m
    """

    # if ~exist('k','var')
    #  k = max(max(cls));
    # end
    if k is None:
        k = np.max(cls)

    # clbs = clstoclbs(cls);
    clbs = clstoclbs(cls)

    # s = clbs' * clbs;
    s = np.dot(clbs.T, clbs)

    # s = checks(s./size(cls,1));
    s = checks(s / cls.shape[0])

    # cl = metis(s,k);
    cl = metis(s, k)

    return cl
