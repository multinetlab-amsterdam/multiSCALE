"""
degrees_und.py

Python conversion of degrees_und.m
Olaf Sporns, Indiana University, 2002/2006/2008
"""

import numpy as np


def degrees_und(CIJ):
    """
    Computes the degree for an undirected (binary) matrix.
    Weights are discarded (any nonzero entry is treated as an edge).

    Parameters
    ----------
    CIJ : array_like (N x N)
        Connection/adjacency matrix.

    Returns
    -------
    deg : ndarray (N,)
        Degree for all vertices.
    """
    CIJ = np.asarray(CIJ)

    # ensure CIJ is binary
    CIJ = (CIJ != 0).astype(float)

    deg = CIJ.sum(axis=0)

    return deg
