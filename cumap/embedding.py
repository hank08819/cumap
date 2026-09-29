"""2D embedding of a precomputed distance matrix via t-SNE or UMAP.

Both backends accept a pairwise distance matrix directly (no recomputation
from feature space) and return a 2D layout suitable for visualization and
downstream clustering.

The UMAP backend can optionally return the fitted ``umap.UMAP`` model object so
callers can access ``model.graph_`` (the internal fuzzy kNN graph) for graph-
based clustering such as Leiden.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Union

import numpy as np


def embed_tsne(
    D: np.ndarray,
    n_components: int = 2,
    perplexity: float = 50.0,
    random_state: int = 42,
    **tsne_kwargs: Any,
) -> np.ndarray:
    """Embed a precomputed distance matrix with t-SNE.

    Parameters
    ----------
    D : ndarray of shape (n, n)
        Pairwise distance matrix (already fused & normalized).
    n_components : int, default=2
        Output dimensionality.
    perplexity : float, default=50.0
        t-SNE perplexity. The c-TSNE paper uses 50; sklearn default is 30.
    random_state : int, default=42
        Seed for reproducibility.
    **tsne_kwargs :
        Additional arguments forwarded to ``sklearn.manifold.TSNE``.

    Returns
    -------
    embedding : ndarray of shape (n, n_components)
    """
    from sklearn.manifold import TSNE

    # init='random' is required when metric='precomputed' (sklearn refuses 'pca')
    tsne = TSNE(
        n_components=n_components,
        metric="precomputed",
        init="random",
        perplexity=perplexity,
        random_state=random_state,
        **tsne_kwargs,
    )
    return tsne.fit_transform(D)


def embed_umap(
    D: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    random_state: int = 42,
    return_model: bool = False,
    **umap_kwargs: Any,
) -> np.ndarray | tuple[np.ndarray, Any]:
    """Embed a precomputed distance matrix with UMAP.

    Parameters
    ----------
    D : ndarray of shape (n, n)
        Pairwise distance matrix.
    n_components : int, default=2
    n_neighbors : int, default=15
        UMAP local neighborhood size.
    random_state : int, default=42
    return_model : bool, default=False
        If True, return ``(embedding, umap_model)`` so callers can access
        ``umap_model.graph_`` for downstream graph clustering.
    **umap_kwargs :
        Additional arguments forwarded to ``umap.UMAP``.

    Returns
    -------
    embedding : ndarray of shape (n, n_components)
    model : umap.UMAP, only if return_model=True
    """
    import umap  # lazy import — umap-learn is heavy and imports numba

    model = umap.UMAP(
        n_components=n_components,
        metric="precomputed",
        n_neighbors=n_neighbors,
        random_state=random_state,
        **umap_kwargs,
    )
    # Suppress two known-harmless UMAP warnings that fire on every fit:
    #   - "using precomputed metric; inverse_transform will be unavailable" — we never use inverse_transform
    #   - "n_jobs value 1 overridden ... by setting random_state" — single-threaded is intentional for reproducibility
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="using precomputed metric")
        warnings.filterwarnings("ignore", message="n_jobs value .* overridden")
        embedding = model.fit_transform(D)
    if return_model:
        return embedding, model
    return embedding


def embed(
    D: np.ndarray,
    backend: Union[str, Callable] = "tsne",
    random_state: int = 42,
    **kwargs: Any,
) -> np.ndarray:
    """Embed a precomputed distance matrix; dispatch by backend name or callable.

    Parameters
    ----------
    D : ndarray of shape (n, n)
    backend : str or callable, default="tsne"
        - str: "tsne" or "umap".
        - callable: a function ``f(D, random_state=int, **kwargs) -> ndarray
          (n, n_components)``. Use this to plug in alternative dimension
          reduction methods (e.g. PHATE, diffusion maps) without modifying
          cumap. The callable should accept a precomputed distance matrix.
    random_state : int, default=42
    **kwargs : forwarded to the chosen backend

    Returns
    -------
    embedding : ndarray of shape (n, n_components)
    """
    if callable(backend):
        return backend(D, random_state=random_state, **kwargs)
    key = backend.lower()
    if key == "tsne":
        return embed_tsne(D, random_state=random_state, **kwargs)
    if key == "umap":
        # Disallow return_model here since the unified API only returns the embedding
        kwargs.pop("return_model", None)
        return embed_umap(D, random_state=random_state, **kwargs)
    raise ValueError(f"Unknown backend '{backend}'. Valid: 'tsne', 'umap', or a callable")
