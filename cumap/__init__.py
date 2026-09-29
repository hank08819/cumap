"""cumap: Explainable dimension reduction for scRNA-seq via distance fusion.

Three biologically-motivated distance metrics (Yule, Fractional, L-Chebyshev)
are computed on raw counts (no preprocessing), fused into a single pairwise
distance matrix, and embedded via t-SNE or UMAP.

Top-level API
-------------
>>> from cumap import CTSNE
>>> model = CTSNE(distances=['yule', 'l_chebyshev'], weights='auto', backend='umap')
>>> emb = model.fit_transform(X, y=y_true)
>>> labels = model.fit_predict(X, y=y_true)
"""

__version__ = "0.0.5"

from cumap.accel import build_distance_graphs, fuse_graphs
from cumap.core import CTSNE
from cumap.distances import (
    compute_distance,
    fractional_distance,
    l_chebyshev_distance,
    low_rank_approx,
    yule_distance,
)
from cumap.datasets import load_pancreas
from cumap.embedding import embed, embed_tsne, embed_umap
from cumap.fusion import fuse_distances, normalize_distance
from cumap.search import search_weights

__all__ = [
    "__version__",
    "CTSNE",
    "compute_distance",
    "yule_distance",
    "fractional_distance",
    "l_chebyshev_distance",
    "low_rank_approx",
    "normalize_distance",
    "fuse_distances",
    "embed",
    "embed_tsne",
    "embed_umap",
    "search_weights",
    "build_distance_graphs",
    "fuse_graphs",
    "load_pancreas",
]
