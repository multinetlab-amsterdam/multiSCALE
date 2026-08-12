"""
global_efficiency_wu.py

Python conversion of global_efficiency_wu.m
Original MATLAB author: Stavros Dimitriadis, 5/2009

Calculates the global efficiency of a weighted graph.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def global_efficiency_wu(w):
    """
    Calculate global efficiency.

    Parameters
    ----------
    w : array_like (N x N)
        Inverse FCG (functional connectivity graph), i.e. a matrix where
        entries represent "distances" (the inverse of connection
        strength) rather than connection strengths themselves.

    Returns
    -------
    gl_node : ndarray (N,)
        Global efficiency per node.
    gl : float
        Total global efficiency of the network.
    """
    w = np.asarray(w, dtype=float)
    n = w.shape[0]

    graph = csr_matrix(w)

    distance = shortest_path(graph, method="D", directed=False)

    inv_distance = np.zeros((n, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(n):
            for j in range(i + 1, n):
                d = distance[i, j]
                val = 1.0 / d  # d == inf (disconnected) -> val = 0.0
                inv_distance[i, j] = val
                inv_distance[j, i] = val

    gl_node = inv_distance.sum(axis=1) / (n - 1)

    gl = gl_node.sum() / n

    return gl_node, gl
