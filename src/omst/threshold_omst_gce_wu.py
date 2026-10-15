"""
Python conversion of threshold_omst_gce_wu.m

Optimizing the formula GE-C (Global Efficiency - Cost) via orthogonal
minimum spanning trees (OMSTs).

This function thresholds the connectivity matrix by selecting edges
through a sequence of orthogonal minimum spanning trees (OMSTs), and
searches for the point (i.e. number of aggregated OMSTs) at which the
formula

    global efficiency - cost

is maximized. This is based on:

    Bassett & Bullmore, "Cognitive fitness of cost-efficient brain
    functional networks", PNAS, 2009.

Original MATLAB author: Dr. Stavros Dimitriadis, 16/7/2013
CUBRIC Neuroimaging Center / Marie-Curie COFUND EU-UK Research Fellow
"""

import numpy as np

from degrees_und import degrees_und
from kruskal import kruskal
from global_efficiency_wu import global_efficiency_wu


def threshold_omst_gce_wu(CIJ, flag=0):
    """
    Threshold a weighted connectivity matrix using orthogonal MSTs (OMSTs)
    by maximizing global efficiency minus cost.

    Parameters
    ----------
    CIJ : array_like (N x N)
        Weighted (or binary) connectivity matrix.
    flag : int, optional
        If 1, plot the GE-Cost curve vs. Cost. Default 0 (no plot).

    Returns
    -------
    nCIJtree : ndarray (no_msts, N, N)
        3D array; nCIJtree[k, :, :] is the aggregated (thresholded)
        connectivity matrix after including the first k+1 orthogonal
        MSTs.
    CIJtree : ndarray (N, N)
        Thresholded connectivity matrix based on the optimal number of
        orthogonal MSTs (i.e. the one maximizing GE - Cost).
    mdeg : float
        Mean degree of the thresholded (optimal) graph.
    globalcosteffmax : float
        Value of the GE-Cost formula at the optimal OMST.
    costmax : float
        Cost (ratio of sum of selected edge weights to total sum of
        weights of the original full-weighted graph) at the optimal OMST.
    E : float
        Global efficiency of the thresholded (optimal) graph.
    """
    CIJ = np.array(CIJ, dtype=float)
    nodes = CIJ.shape[0]

    CIJ = np.triu(CIJ)

    no_edges = int(np.count_nonzero(CIJ > 0))
    no_msts = round(no_edges / (nodes - 1)) + 1
    pos_no_msts = round(no_edges / (nodes - 1))
    if no_msts > pos_no_msts:
        no_msts = pos_no_msts

    CIJnotintree = CIJ.copy()

    # keep the N-1 connections of the no_msts MSTs
    no_msts = 15

    mst_con = np.zeros((no_msts * (nodes - 1), 2), dtype=int)

    count = 0
    CIJtree = np.zeros((nodes, nodes))
    nCIJtree = np.zeros((no_msts, nodes, nodes))
    omst = np.zeros((no_msts, nodes, nodes))

    for no in range(no_msts):
        adj = (CIJnotintree > 0).astype(int)
        with np.errstate(divide="ignore"):
            w = 1.0 / CIJnotintree

        w_st, links, X_st = kruskal(adj, w)

        if links.shape[0] > 0:
            links = np.sort(links, axis=1)

        mst = np.zeros((nodes, nodes))
        for k in range(links.shape[0]):
            i, j = int(links[k, 0]), int(links[k, 1])

            CIJtree[i, j] = CIJ[i, j]
            CIJtree[j, i] = CIJ[i, j]

            mst_con[count, 0] = i
            mst_con[count, 1] = j
            count += 1

            mst[i, j] = CIJ[i, j]
            mst[j, i] = CIJ[i, j]

        CIJnotintree = CIJnotintree * (CIJtree == 0)

        nCIJtree[no, :, :] = CIJtree
        omst[no, :, :] = mst

    with np.errstate(divide="ignore"):
        _, E_ini = global_efficiency_wu(1.0 / CIJ)
    cost_ini = CIJ.sum()

    globalcosteff = np.zeros(no_msts)
    degree = np.zeros(no_msts)
    cost = np.zeros(no_msts)

    for k in range(no_msts):
        graph = nCIJtree[k, :, :]
        deg = degrees_und(graph)
        degree[k] = deg.mean()

        cost[k] = graph.sum() / cost_ini
        with np.errstate(divide="ignore"):
            _, E = global_efficiency_wu(1.0 / graph)
        globalcosteff[k] = E / E_ini - cost[k]

    ind = int(np.argmax(globalcosteff))

    mdeg = degree[ind]
    CIJtree_final = nCIJtree[ind, :, :]
    costmax = cost[ind]

    with np.errstate(divide="ignore"):
        _, E = global_efficiency_wu(1.0 / CIJtree_final)

    globalcosteffmax = globalcosteff[ind]

    if flag == 1:
        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.plot(cost, globalcosteff)
        plt.plot(costmax, globalcosteffmax, "r*")
        plt.xlabel("Cost")
        plt.ylabel("Global Cost Efficiency")
        plt.title("Economical small-world network at max Global Cost Efficiency")
        plt.annotate(
            r"$\leftarrow$ max Global Cost Efficiency",
            xy=(costmax, globalcosteffmax),
            xytext=(costmax, globalcosteffmax),
            horizontalalignment="left",
        )
        plt.show()

    return nCIJtree, CIJtree_final, mdeg, globalcosteffmax, costmax, E
