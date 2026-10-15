"""
kruskal.py

Python conversion of kruskal.m

Finds the minimum spanning tree of a graph where each edge has a
specified weight, using Kruskal's algorithm.

Original Author (MATLAB): Georgios Papachristoudis
Copyright 2013 Georgios Papachristoudis
Date: 2013/05/26
"""

import numpy as np


def kruskal(X, w):
    """
    Find the minimum spanning tree of a graph using Kruskal's algorithm.

    Parameters
    ----------
    X : array_like
        Either:
          - N x N binary adjacency matrix. X[i, j] = 1 means there is a
            directed edge from node i to node j. If X is symmetric, the
            graph is treated as undirected.
          - E x 2 array of edges (neighbors' matrix). Each row is an
            edge (source, target).

    w : array_like
        Either:
          - N x N weight matrix in adjacency form (if X is a matrix). If
            X is symmetric (undirected graph), w must also be symmetric.
          - length-E array of weights, one per edge (if X is an edge list).

    Returns
    -------
    w_st : float
        Total weight of the minimum spanning tree.
    ST : ndarray (Nst x 2)
        Neighbors' matrix (edge list) of the minimum spanning tree.
    X_st : ndarray (N x N)
        Adjacency matrix of the minimum spanning tree. If X_st is
        symmetric, the tree is undirected.
    """
    X = np.asarray(X)
    is_undirected = False

    # Determine whether X is a square binary adjacency matrix or an
    # edge list (E x 2).
    if X.ndim == 2 and X.shape[0] == X.shape[1] and _is_binary(X):
        if np.any(X != X.T):
            is_undirected = False
        else:
            is_undirected = True
        ne = _cnvrtX2ne(X, is_undirected)
    else:
        # Edge list form (E x 2)
        ne = np.asarray(X, dtype=int)
        is_undirected = False

    w = np.asarray(w)

    # Convert weight matrix from adjacency form to neighbors' (edge) form
    if w.ndim == 2 and w.shape[0] == w.shape[1]:
        if is_undirected and np.any(w != w.T):
            raise ValueError(
                "If it is an undirected graph, weight matrix has to be symmetric."
            )
        w_edges = _cnvrtw2ne(w, ne)
    else:
        w_edges = np.asarray(w).flatten()

    N = int(ne.max()) + 1 if ne.size > 0 else 0
    Ne = ne.shape[0]

    order = np.argsort(w_edges, kind="stable")
    w_sorted = w_edges[order]
    ne_sorted = ne[order]

    repr_ = np.arange(N)
    rnk = np.zeros(N, dtype=int)

    def find(i):
        root = i
        while repr_[root] != root:
            root = repr_[root]
        while repr_[i] != root:
            repr_[i], i = root, repr_[i]
        return root

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        if rnk[ri] > rnk[rj]:
            repr_[rj] = ri
        else:
            repr_[ri] = rj
            if rnk[ri] == rnk[rj]:
                rnk[rj] += 1

    lidx = np.zeros(Ne, dtype=bool)

    # Kruskal's algorithm
    for k in range(Ne):
        i, j = int(ne_sorted[k, 0]), int(ne_sorted[k, 1])
        if find(i) != find(j):
            lidx[k] = True
            union(i, j)

    # MST
    ST = ne_sorted[lidx]
    w_st_edges = w_sorted[lidx]

    # Generate adjacency matrix of the MST
    X_st = np.zeros((N, N), dtype=int)
    for k in range(ST.shape[0]):
        i, j = int(ST[k, 0]), int(ST[k, 1])
        X_st[i, j] = 1
        if is_undirected:
            X_st[j, i] = 1

    # Evaluate the total weight of the minimum spanning tree
    w_st = float(w_st_edges.sum()) if w_st_edges.size > 0 else 0.0

    return w_st, ST, X_st


def _is_binary(X):
    vals = np.unique(X)
    return np.all(np.isin(vals, [0, 1]))


def _cnvrtX2ne(X, is_undirected):
    """Convert adjacency matrix to a neighbors' (edge list) matrix."""
    n = X.shape[0]
    edges = []
    for i in range(n):
        js = np.where(X[i, :] != 0)[0]
        if is_undirected:
            js = js[js > i]
        for j in js:
            edges.append((i, j))
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=int)
    return np.array(edges, dtype=int)


def _cnvrtw2ne(w, ne):
    """Convert weight matrix (adjacency form) to neighbors' (edge) form."""
    if ne.shape[0] == 0:
        return np.zeros((0,))
    return np.array([w[i, j] for i, j in ne])
