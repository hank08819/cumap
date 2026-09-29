"""Biologically-motivated pairwise distance metrics for scRNA-seq data.

Three distance metrics, each addressing a hierarchical biological factor in
single-cell RNA-seq data (Han et al. 2022, c-TSNE paper):

  - Yule         -> Q1: Which genes are expressed? (binary on/off)
  - L-Chebyshev  -> Q2: Levels of top-expressed genes?  (SVD k=15 + Chebyshev)
  - Fractional   -> Q3: Levels of all genes?            (L^(1/4) norm)

All functions take a raw count matrix (n_cells, n_genes) and return a symmetric
(n_cells, n_cells) distance matrix with zero diagonal. NO normalization, NO QC
filtering, NO imputation is applied — every cell, every gene, every zero entry
is treated as informative signal.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist, squareform


def yule_distance(X: np.ndarray) -> np.ndarray:
    """Pairwise Yule distance: binary on/off gene expression similarity.

    Treats non-zero entries as "gene expressed" and zero entries as
    "gene not expressed". This explicitly models scRNA-seq dropout as
    biological signal rather than missing data to be imputed.

    Parameters
    ----------
    X : ndarray of shape (n_cells, n_genes)
        Raw count matrix.

    Returns
    -------
    D : ndarray of shape (n_cells, n_cells)
        Symmetric pairwise Yule distance matrix.
    """
    X = np.asarray(X)
    D = pdist(X, metric="yule")
    return squareform(D)


def fractional_distance(X: np.ndarray, f: float = 0.25) -> np.ndarray:
    """Pairwise fractional L^f distance: whole-gene-level expression.

    Defined as ``d(u, v) = (sum_i |u_i - v_i|^f)^(1/f)``. With f<1 the metric
    down-weights large per-gene differences, making it more robust than the
    Euclidean distance in high-dimensional sparse data.

    Parameters
    ----------
    X : ndarray of shape (n_cells, n_genes)
        Raw count matrix.
    f : float, default=0.25
        Fractional power. f=2 recovers Euclidean; f=1 recovers Manhattan.

    Returns
    -------
    D : ndarray of shape (n_cells, n_cells)
        Symmetric pairwise fractional distance matrix.
    """
    if f <= 0:
        raise ValueError(f"Fractional power f must be positive, got {f}")
    X = np.asarray(X)
    # scipy minkowski with p=f is mathematically identical to (sum |u-v|^f)^(1/f)
    D = pdist(X, metric="minkowski", p=f)
    return squareform(D)


def l_chebyshev_distance(X: np.ndarray, k: int = 15) -> np.ndarray:
    """Pairwise L-Chebyshev distance: top-expression-mode similarity.

    Two steps:
      1. Truncate X to rank-k via SVD, retaining the top-k expression modes
         (analogous to top-k principal components but on raw counts).
      2. Compute pairwise Chebyshev (L-infinity) distance on the truncated
         representation: ``d(u, v) = max_i |u_i - v_i|``.

    Parameters
    ----------
    X : ndarray of shape (n_cells, n_genes)
        Raw count matrix.
    k : int, default=15
        SVD truncation rank. k=0 disables truncation (uses raw X). If ``k``
        exceeds ``min(n_cells, n_genes)``, it is capped to that value and a
        ``UserWarning`` is emitted.

    Returns
    -------
    D : ndarray of shape (n_cells, n_cells)
        Symmetric pairwise L-Chebyshev distance matrix.
    """
    X = np.asarray(X)
    max_k = min(X.shape)
    if k > max_k:
        warnings.warn(
            f"l_chebyshev k={k} exceeds min(X.shape)={max_k}; capping to {max_k}.",
            UserWarning,
            stacklevel=2,
        )
        k = max_k
    X_low = low_rank_approx(X, k=k)
    D = pdist(X_low, metric="chebyshev")
    return squareform(D)


def low_rank_approx(X: np.ndarray, k: int) -> np.ndarray:
    """Truncated SVD rank-k approximation of X.

    Parameters
    ----------
    X : ndarray of shape (m, n)
    k : int
        Rank to retain. k=0 returns X unchanged.

    Returns
    -------
    X_low : ndarray of shape (m, n)
        Rank-k approximation U @ diag(S[:k]) @ V.
    """
    if k == 0:
        return X
    m, n = X.shape
    L = min(m, n)
    if k > L:
        raise ValueError(f"Requested rank k={k} exceeds min(m, n)={L}")
    U, S, V = linalg.svd(X, full_matrices=False)
    sigma = np.zeros((L, L))
    for i in range(k):
        sigma[i, i] = S[i]
    return U @ sigma @ V


_DISTANCE_FUNCTIONS = {
    "yule": yule_distance,
    "fractional": fractional_distance,
    "l_chebyshev": l_chebyshev_distance,
    "l-chebyshev": l_chebyshev_distance,
}


def compute_distance(X: np.ndarray, name_or_callable, **kwargs) -> np.ndarray:
    """Dispatch to a named distance function or call a user-supplied callable.

    Parameters
    ----------
    X : ndarray of shape (n_cells, n_genes)
        Raw count matrix.
    name_or_callable : str or callable
        - str: one of "yule", "fractional", "l_chebyshev".
        - callable: a function ``f(X, **kwargs) -> ndarray (n, n)`` returning a
          symmetric pairwise distance matrix with zero diagonal. Useful for
          plugging in custom metrics (e.g. Jaccard, Wasserstein) without
          modifying cumap.
    **kwargs : passed to the underlying distance function (e.g. ``f`` for
        fractional, ``k`` for l_chebyshev). For user callables, these are
        forwarded as-is; pass nothing if your callable takes no kwargs.

    Returns
    -------
    D : ndarray of shape (n_cells, n_cells)
    """
    if callable(name_or_callable):
        D = np.asarray(name_or_callable(X, **kwargs))
        n = X.shape[0]
        if D.ndim != 2 or D.shape != (n, n):
            raise ValueError(
                f"Callable distance {name_or_callable!r} returned shape {D.shape}; "
                f"expected ({n}, {n})."
            )
        if not np.all(np.isfinite(D)):
            raise ValueError(
                f"Callable distance {name_or_callable!r} returned NaN/Inf values."
            )
        return D
    key = name_or_callable.lower()
    if key not in _DISTANCE_FUNCTIONS:
        valid = sorted({"yule", "fractional", "l_chebyshev"})
        raise ValueError(f"Unknown distance '{name_or_callable}'. Valid: {valid} or a callable.")
    return _DISTANCE_FUNCTIONS[key](X, **kwargs)
