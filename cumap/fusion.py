"""Distance matrix normalization and fusion.

Two fusion modes from Han et al. 2022 (c-TSNE paper):

  - "sum"  : weighted linear combination of normalized distances.
             Best for low-sparsity datasets (sparsity < ~50%).
  - "max"  : element-wise max of normalized distances.
             Best for high-sparsity datasets (sparsity > ~50%).

Normalization divides each distance matrix by its largest-magnitude eigenvalue,
bringing every matrix to comparable scale before fusion.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from scipy.sparse.linalg import eigsh


def normalize_distance(D: np.ndarray) -> np.ndarray:
    """Scale a distance matrix so its spectral radius equals 1.

    Computes the largest-magnitude eigenvalue lambda_max via ARPACK (eigsh) and
    returns ``D / lambda_max``. Faster than full eigendecomposition for
    moderately large N because we only need one eigenvalue.

    Parameters
    ----------
    D : ndarray of shape (n, n)
        Symmetric pairwise distance matrix.

    Returns
    -------
    D_norm : ndarray of shape (n, n)
        Scaled matrix with spectral radius 1.
    """
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square, got shape {D.shape}")
    # 'LM' = largest magnitude; k=1 returns just lambda_max
    eigmax = eigsh(D, k=1, which="LM", return_eigenvectors=False)[0]
    if eigmax == 0:
        raise ValueError("Distance matrix has zero spectral radius; cannot normalize")
    return D / eigmax


def fuse_distances(
    distances: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
    mode: str = "sum",
    normalize: bool = True,
    validate_weights: bool = True,
) -> np.ndarray:
    """Combine multiple distance matrices into a single fused matrix.

    Parameters
    ----------
    distances : sequence of ndarray, each shape (n, n)
        Pairwise distance matrices to fuse. Must have identical shapes.
    weights : sequence of float, optional
        Per-matrix weights for ``mode='sum'``. If None, equal weights are used.
        For ``mode='max'`` the weights are ignored (max is unweighted).
    mode : {"sum", "max"}, default="sum"
        - "sum": weighted linear combination (after optional normalization).
        - "max": element-wise maximum across the normalized matrices.
    normalize : bool, default=True
        If True, each input distance is divided by its spectral radius before
        fusion. Set False only if your distances are already on comparable
        scales.
    validate_weights : bool, default=True
        If True and ``mode='sum'``, negative weights raise ``ValueError`` and
        weights not summing to 1.0 emit ``UserWarning``. Set False to permit
        arbitrary linear combinations.

    Returns
    -------
    D_fused : ndarray of shape (n, n)
        Fused distance matrix.
    """
    if mode not in ("sum", "max"):
        raise ValueError(f"mode must be 'sum' or 'max', got {mode!r}")

    if len(distances) == 0:
        raise ValueError("Need at least one distance matrix to fuse")

    shapes = {D.shape for D in distances}
    if len(shapes) != 1:
        raise ValueError(f"All distance matrices must share shape; got {shapes}")

    # Normalize each distance to unit spectral radius (lambda_max)
    if normalize:
        norm_mats = [normalize_distance(D) for D in distances]
    else:
        norm_mats = [np.asarray(D, dtype=np.float64) for D in distances]

    if mode == "sum":
        if weights is None:
            weights = [1.0 / len(norm_mats)] * len(norm_mats)
        if len(weights) != len(norm_mats):
            raise ValueError(
                f"weights length {len(weights)} != distances length {len(norm_mats)}"
            )
        if validate_weights:
            w_arr = np.asarray(weights, dtype=np.float64)
            if (w_arr < 0).any():
                raise ValueError(
                    f"Negative weights not allowed when validate_weights=True; got {list(weights)}"
                )
            if not np.isclose(w_arr.sum(), 1.0, atol=1e-6):
                warnings.warn(
                    f"Weights sum to {w_arr.sum():.4f}, not 1.0; fused distance "
                    f"will not have unit spectral radius. Pass validate_weights=False to silence.",
                    UserWarning,
                    stacklevel=2,
                )
        D_fused = np.zeros_like(norm_mats[0])
        for w, D in zip(weights, norm_mats):
            D_fused = D_fused + w * D
        return D_fused

    # mode == "max"
    return np.maximum.reduce(norm_mats)
