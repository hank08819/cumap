"""Input adapter: extract a dense 2D ndarray from various input types.

Supports plain ndarray / sparse matrix pass-through and AnnData objects
(scanpy's standard data container). AnnData support uses duck typing —
``anndata`` is not a hard dependency; install via
``pip install cumap[anndata]`` if you want to pass an AnnData object
directly.

The cumap package itself does no preprocessing. Whether the user passes
raw counts (matching the paper's identical-input benchmark) or
scanpy-normalized expression (e.g. ``log1p`` of ``normalize_total``) is
the user's choice; only the shape and orientation must match
``(n_cells, n_genes)``.
"""

from __future__ import annotations

import warnings

import numpy as np

# Warn when densifying sparse input above ~8 GB float64 (1e5 cells x 1e4 genes).
_LARGE_DENSE_WARN_ENTRIES = 1_000_000_000


def _looks_like_anndata(X) -> bool:
    """Duck-type check for AnnData without importing anndata."""
    return hasattr(X, "X") and hasattr(X, "obs") and hasattr(X, "var")


def _extract_matrix(X, layer=None) -> np.ndarray:
    """Return a 2D dense ndarray ``(n_cells, n_genes)`` from ``X``.

    Parameters
    ----------
    X : ndarray-like, scipy sparse matrix, or AnnData
        Input matrix or AnnData object.
    layer : None | str
        Only used when ``X`` is AnnData.
        - ``None``: use ``adata.X`` (scanpy default main matrix).
        - ``"raw"``: use ``adata.raw.X`` (scanpy convention for stashing
          raw counts before normalize+log1p).
        - any other string: use ``adata.layers[layer]``.

    Returns
    -------
    ndarray of shape ``(n_cells, n_genes)``.

    Notes
    -----
    Sparse inputs are densified via ``.toarray()`` because cumap's distance
    metrics require dense input. A warning is emitted when the dense
    result would exceed ~8 GB.
    """
    if _looks_like_anndata(X):
        adata = X
        if layer is None:
            M = adata.X
        elif layer == "raw":
            if getattr(adata, "raw", None) is None:
                raise ValueError(
                    "layer='raw' but adata.raw is None. Set "
                    "`adata.raw = adata` before normalize_total."
                )
            M = adata.raw.X
        else:
            if layer not in adata.layers:
                available = list(adata.layers.keys())
                raise ValueError(
                    f"layer={layer!r} not in adata.layers; "
                    f"available layers: {available}"
                )
            M = adata.layers[layer]
    else:
        if layer is not None:
            raise ValueError(
                f"layer={layer!r} given but X is not an AnnData object "
                f"(type {type(X).__name__}). The layer argument only applies "
                "to AnnData inputs."
            )
        M = X

    if hasattr(M, "toarray"):
        n_entries = int(np.prod(M.shape))
        if n_entries > _LARGE_DENSE_WARN_ENTRIES:
            warnings.warn(
                f"Densifying sparse input of shape {tuple(M.shape)} "
                f"(~{n_entries * 8 / 1e9:.1f} GB float64). cumap distance "
                "metrics require dense input.",
                stacklevel=3,
            )
        M = M.toarray()

    M = np.asarray(M)
    if M.ndim != 2:
        raise ValueError(
            f"Expected 2D matrix (n_cells, n_genes), got shape {M.shape}"
        )
    return M
