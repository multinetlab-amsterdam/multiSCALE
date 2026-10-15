import numpy as np
from scipy import linalg
from .utils import svd_sign_flip

def rcca_tr(X: np.ndarray, lambda_: float, name: str = "X"):
    """
    Applies kernel trick + regularization transform to matrix X.
    Returns transformed matrix, optional right singular vectors V, and regularized covariance.
    """
    n, p = X.shape

    if lambda_ < 0:
        raise ValueError(f"Please make lambda for '{name}' >= 0")

    V = None
    # Kernel trick
    if p > n:
        if lambda_ == 0:
            raise ValueError(
                f"Singularity issue: please impose a penalty on '{name}' side "
                f"(p={p} > n={n} requires lambda > 0)"
            )
        U, d, Vt = linalg.svd(X, full_matrices=False)
        X = U * d          
        V = Vt.T           

    
    C = np.cov(X, rowvar=False)          
    np.fill_diagonal(C, np.diag(C) + lambda_)

    return {"mat": X, "tr": V, "cov": C}


def rcca_inv_tr(X: np.ndarray, alpha: np.ndarray, V):
    """
    Inverse-transforms canonical coefficients back to original feature space
    and computes canonical variates.
    """
    n_comp = alpha.shape[1]
    comp_names = [f"can_comp{i+1}" for i in range(n_comp)]

    # Canonical variates: u = X @ alpha
    u = X @ alpha

    # Inverse transform canonical coefficients
    if V is not None:
        alpha = V @ alpha                

    return {"coefs": alpha, "vars": u, "comp_names": comp_names}

def geigen(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    """
    Solves the generalized CCA eigenvalue problem.
    Maximizes α'Aβ subject to α'Bα = 1 and β'Cβ = 1.

    Decomposition:
        M = B^{-1/2} A C^{-1/2}  via Cholesky + SVD

    Returns (rho, alpha, beta) sorted by descending singular value.
    """
    Lb = linalg.cholesky(B, lower=True)   # B = Lb @ Lb.T  (Cxx, X-side)
    Lc = linalg.cholesky(C, lower=True)   # C = Lc @ Lc.T  (Cyy, Y-side)

    Lb_inv = linalg.solve_triangular(Lb, np.eye(len(Lb)), lower=True)
    Lc_inv = linalg.solve_triangular(Lc, np.eye(len(Lc)), lower=True)

    M = Lb_inv @ A @ Lc_inv.T            # (p x q) matrix to decompose

    # U, rho, Vt = linalg.svd(M, full_matrices=False)
    U, rho, Vt = svd_sign_flip(M)

    alpha = Lb_inv.T @ U     # canonical coefficients for X 
    beta  = Lc_inv.T @ Vt.T  # canonical coefficients for Y 

    return rho, alpha, beta


def RCCA(
    X: np.ndarray,
    Y: np.ndarray,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    x_feature_names=None,
    y_feature_names=None,
    obs_names=None,
):
    """
    Regularized Canonical Correlation Analysis (RCCA) with L2 penalty.

    Parameters
    ----------
    X         : (n x p) array  n observations of random vector x
    Y         : (n x q) array  n observations of random vector y
    lambda1   : L2 penalty for X side (default 0 = no regularization)
    lambda2   : L2 penalty for Y side (default 0 = no regularization)
    x_feature_names : optional list of length p for X column names
    y_feature_names : optional list of length q for Y column names
    obs_names       : optional list of length n for row names

    Returns
    -------
    dict with keys:
        n_comp    : number of canonical components  k = min(p, q)
        cors      : (k,) array of canonical correlations  cor(u, v)
        mod_cors  : (k,) array of modified canonical correlations
        x_coefs   : (p x k) canonical coefficient matrix  [α¹ … αᵏ]
        x_vars    : (n x k) canonical variates matrix     [u¹ … uᵏ]
        y_coefs   : (q x k) canonical coefficient matrix  [β¹ … βᵏ]
        y_vars    : (n x k) canonical variates matrix     [v¹ … vᵏ]
        comp_names: list of component labels
    """
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    n_comp = min(X.shape[1], Y.shape[1], X.shape[0])
    comp_names = [f"can_comp{i+1}" for i in range(n_comp)]

    # Transform
    X_tr = rcca_tr(X, lambda1, name="X")
    Y_tr = rcca_tr(Y, lambda2, name="Y")

    # Solve optimization problem 
    Cxx = X_tr["cov"]
    Cyy = Y_tr["cov"]
    Cxy = np.cov(X_tr["mat"], Y_tr["mat"], rowvar=False)[
        : X_tr["mat"].shape[1], X_tr["mat"].shape[1] :
    ]                                                    

    rho_mod_all, alpha_all, beta_all = geigen(Cxy, Cxx, Cyy)

    # Modified canonical correlations (top k)
    rho_mod = rho_mod_all[:n_comp]
    
    # Inverse transform
    X_inv = rcca_inv_tr(X_tr["mat"], alpha_all[:, :n_comp], X_tr["tr"])
    Y_inv = rcca_inv_tr(Y_tr["mat"], beta_all[:, :n_comp],  Y_tr["tr"])

    x_coefs = X_inv["coefs"]   # (p x n_comp)
    x_vars  = X_inv["vars"]    # (n x n_comp)
    y_coefs = Y_inv["coefs"]   # (q x n_comp)
    y_vars  = Y_inv["vars"]    # (n x n_comp)

    # Apply feature / obs names if provided
    if x_feature_names is not None:
        x_coefs = {name: x_coefs[i] for i, name in enumerate(x_feature_names)}
    if y_feature_names is not None:
        y_coefs = {name: y_coefs[i] for i, name in enumerate(y_feature_names)}

    # True canonical correlations  cor(u, v)
    cors = np.array([
        np.corrcoef(x_vars[:, i], y_vars[:, i])[0, 1]
        for i in range(n_comp)
    ])

    return {
        "n_comp":    n_comp,
        "comp_names": comp_names,
        "cors":      cors,       # true   cor(Xα, Yβ)
        "mod_cors":  rho_mod,    # modified (regularized) correlations
        "x_coefs":   x_coefs,
        "x_vars":    x_vars,
        "y_coefs":   y_coefs,
        "y_vars":    y_vars,
    }