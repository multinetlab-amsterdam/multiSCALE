import numpy as np
import pandas as pd
from . import RCCA
from sklearn.covariance import graphical_lasso
from gglasso.problem import glasso_problem

import copy

def normalize_0_1_matrix(matrix):
    matrix = np.abs(np.array(matrix, dtype=float))
    min_val = matrix.min()
    max_val = matrix.max()
    
    # Avoid division by zero if all values are the same
    if max_val == min_val:
        return np.zeros_like(matrix)
    
    normalized = (matrix - min_val) / (max_val - min_val)
    return normalized

def compute_consensus(subject_matrices, threshold=0.15, aggreg_strategy="mean"):
    """
    Compute a consensus matrix from a list/array of subjects.
    An edge is kept if it appears in at least `threshold` proportion of subjects, then aggregated across subjects using the specified strategy.

    Parameters
    ----------
    subject_matrices : list of np.ndarray or np.ndarray of shape (n_subjects, n_rois, n_rois)
        Binary or weighted connectivity matrices per subject.
    threshold : float
        Proportion of subjects that must have an edge for it to be retained (e.g. 0.5 = 50%).
    aggreg_strategy : str
        Aggregation strategy for retained edges: "mean" or "median".

    Returns
    -------
    consensus_matrix : np.ndarray of shape (n_rois, n_rois)
        Weighted consensus matrix: aggregated weight where edge is retained, 0 otherwise.
    """
    if aggreg_strategy not in ("mean", "median"):
        raise ValueError(f"aggreg_strategy must be 'mean' or 'median', got '{aggreg_strategy}'")
        
    matrices = np.array(subject_matrices)          # (n_subjects, n_rois, n_rois)
    n_subjects = matrices.shape[0]

    binary_matrices = (matrices != 0).astype(float)

    edge_frequency = binary_matrices.sum(axis=0) / n_subjects
    
    edge_mask = edge_frequency >= threshold                     

    # Aggregate weights only over subjects where the edge is present
    # Zero-out absent edges per subject before aggregating so they don't pull the statistic
    present_weights = np.where(binary_matrices.astype(bool), matrices, np.nan)

    if aggreg_strategy == "mean":
        with np.errstate(all='ignore'):  
            aggregated = np.nanmean(present_weights, axis=0)
    else:  
        with np.errstate(all='ignore'):  
            aggregated = np.nanmedian(present_weights, axis=0)

    consensus_matrix = np.where(edge_mask, aggregated, 0.0)

    return consensus_matrix 
        
def RCCA_multi(X, Y, lambda1=5, lambda2=0):
    P, Q = X.shape[1], Y.shape[1]
    rcca = RCCA.RCCA(X, Y, lambda1=lambda1, lambda2=lambda2)
    
    # Remove last canonical component that is degenerate, which produces a low canonical correlation and a near-zero variance canonical weights. If we keep it, it would impact the subsequent computation of canonical loadings.
    
    n_comp = rcca['x_vars'].shape[1]
    if n_comp == min(X.shape[0], Y.shape[0]):
        keep_n = n_comp - 1
    else:
        keep_n = n_comp
    
    X_loadings = np.corrcoef(X, rcca['x_vars'][:, :keep_n], rowvar=False)[:P, P:]
    Y_loadings = np.corrcoef(Y, rcca['y_vars'][:, :keep_n], rowvar=False)[:Q, Q:]
    
    return X_loadings, Y_loadings

def partial_interlayer(X_loadings, Y_loadings, N, lambda1=0.01): 
    P, Q = X_loadings.shape[0], Y_loadings.shape[0]
    
    full_loadings = np.vstack([X_loadings, Y_loadings])  
    
    S = np.corrcoef(full_loadings)
    
    reg_params = {
        'lambda1': lambda1,
    }
    
    problem = glasso_problem(S, N=N, reg = "GGL", reg_params = reg_params, latent = False)
    problem.solve(solver_params={'max_iter': 10000}, verbose=False)

    sol = problem.solution.precision_
    
    # Compute partial correlations
    D = np.diag(1 / np.sqrt(np.diag(sol)))
    partial_corr = -D @ sol @ D

    interlayer = partial_corr[:P, P:P+Q].T
    
    return interlayer

def split_interlayer_links(interlayer):   
    def normalize_0_1_preserve_sign(arr):
        arr = np.asarray(arr, dtype=float)
        
        sign = np.sign(arr)          
        abs_arr = np.abs(arr)        
        
        # normalize the absolute values to [0, 1]
        min_val, max_val = abs_arr.min(), abs_arr.max()
        if max_val - min_val == 0:
            norm = np.zeros_like(abs_arr)
        else:
            norm = (abs_arr - min_val) / (max_val - min_val)
        
        # reapply original sign
        result = norm * sign
        return result
    
    interlayer_links_split = {}
    
    tmp_pos_1 = copy.deepcopy(normalize_0_1_preserve_sign(interlayer))
    tmp_neg_1  = copy.deepcopy(normalize_0_1_preserve_sign(interlayer))
    tmp_pos_1[tmp_pos_1 < 0] = 0
    tmp_neg_1[tmp_neg_1 > 0] = 0
    interlayer_links_split["pos"] = tmp_pos_1
    interlayer_links_split["neg"] = np.abs(tmp_neg_1)
    
    return interlayer_links_split

def build_supra_adjacency(intra_layers, inter_layers, layer_size):
    L = sum(layer_size)
    supra_adj = np.zeros((L, L))
    
    # Intra-layers
    cum_sum = 0
    for i, layer in enumerate(intra_layers):
        supra_adj[cum_sum:cum_sum + layer_size[i], cum_sum:cum_sum + layer_size[i]] = layer
        cum_sum += layer_size[i]
        
    # Inter-layer links
    N = len(layer_size)
    count = 0
    cum_sum_i = 0 
    for i in range(N):
        cum_sum_j = cum_sum_i + layer_size[i]
        for j in range(i+1, N):
            supra_adj[cum_sum_i:cum_sum_i + layer_size[i], cum_sum_j:cum_sum_j + layer_size[j]] = inter_layers[count]
            supra_adj[cum_sum_j:cum_sum_j + layer_size[j], cum_sum_i:cum_sum_i + layer_size[i]] = inter_layers[count].T
            cum_sum_j += layer_size[j]
            count += 1
        cum_sum_i += layer_size[i]
        
    return supra_adj

def ebic_glasso(data: pd.DataFrame, gamma: float = 0.5, rowvar=False, n_lambdas: int = 100,
                 lambda_min_ratio: float = 0.01, tol: float = 1e-4, max_iter: int = 2000):
    X = data.values
    n, p = X.shape

    corr = np.corrcoef(X, rowvar=rowvar)

    lambda_max = np.max(np.abs(corr - np.diag(np.diag(corr))))
    lambda_min = lambda_min_ratio * lambda_max
    lambdas = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), n_lambdas))

    best_ebic = np.inf
    best_result = None

    for lam in lambdas:
        try:
            cov, precision = graphical_lasso(corr, alpha=lam, max_iter=max_iter, tol=tol)
        except FloatingPointError:
            continue

        nonzero = np.sum(np.abs(precision) > 1e-10)
        edges = (nonzero - p) / 2

        sign, logdet = np.linalg.slogdet(precision)
        log_likelihood = 0.5 * n * (logdet - np.trace(corr @ precision))

        ebic = -2 * log_likelihood + edges * np.log(n) + 4 * edges * gamma * np.log(p)

        if ebic < best_ebic:
            best_ebic = ebic
            best_result = {"lambda": lam, "precision": precision, "cov": cov, "ebic": ebic}

    precision = best_result["precision"]
    d = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 0)
    best_result["network"] = pd.DataFrame(partial_corr, index=data.columns, columns=data.columns)

    return best_result
