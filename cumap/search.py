"""Grid search for optimal fusion weights (and Leiden resolution).

Two clustering modes, both supervised by NMI against ground-truth labels:

  - cluster='kmeans' : single-layer sweep over fusion weights.
                       For each weight point: UMAP -> KMeans(n_clusters) -> NMI.
  - cluster='leiden' : two-layer sweep over (weights, leiden_resolution).
                       For each weight: UMAP -> graph_ -> Leiden CPM scanned
                       over ``leiden_resolutions``; the resolution giving the
                       highest NMI is kept. Matches the paper experimental
                       protocol on the H1-H4 pancreas benchmark
                       (Han et al. 2022).

For N distances and ``weights_grid=None``, an (N-1)-simplex grid is generated
with step ``1/(n_grid-1)`` (default ``n_grid=11`` → 11/66/286 points for
N=2/3/4). Pass ``weights_grid`` explicitly to use a sparser grid.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Union

import numpy as np

from .embedding import embed, embed_umap
from .fusion import fuse_distances


# 16-point Leiden CPM resolution grid. Denser in the low-resolution region
# (0.0005..0.005) where typical scRNA-seq best NMI lives — the H1 best is at
# res=0.0025 with cumap's triu-fixed graph (== res=0.005 under the H1-H4
# notebook's double-counted-edge graph).
DEFAULT_LEIDEN_RESOLUTIONS: tuple[float, ...] = (
    0.0005, 0.001, 0.0015, 0.002, 0.0025,
    0.003, 0.004, 0.005, 0.006, 0.008,
    0.010, 0.015, 0.02, 0.03, 0.05, 0.1,
)


def _default_simplex_grid(
    n_dist: int, n_grid: int = 11
) -> list[tuple[float, ...]]:
    """Generate the (n_dist-1)-simplex weight grid with step ``1/(n_grid-1)``.

    Each returned tuple has ``len == n_dist``, all entries are non-negative,
    and entries sum to exactly 1 (integer lattice on the standard simplex
    with denominator ``n_grid - 1``). For ``n_dist=2`` this reproduces the
    historical 11-point 1-simplex used since v0.0.1.

    Point counts equal ``C(n_grid + n_dist - 2, n_dist - 1)``. At
    ``n_grid=11``: 11 (N=2), 66 (N=3), 286 (N=4), 1001 (N=5).
    """
    if n_dist < 2:
        raise ValueError(f"n_dist must be >= 2, got {n_dist}")
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")
    step = 1.0 / (n_grid - 1)
    total = n_grid - 1  # 整数分量之和 = total, 实数权重之和 = total*step = 1
    pts: list[tuple[float, ...]] = []

    def _recurse(remaining: int, depth: int, prefix: list[int]) -> None:
        # 最后一维必须吃掉剩余 quota, 保证 sum == total (i.e. weights sum == 1.0)
        if depth == 1:
            pts.append(tuple((p * step) for p in (prefix + [remaining])))
            return
        # 倒序遍历: 第 1 维从 remaining → 0, 跟 v0.0.1-v0.0.3 旧版 _default_2dist_grid
        # 输出顺序 (1.0, 0.0) → ... → (0.0, 1.0) 完全一致, 保 N=2 结果可复现.
        for i in range(remaining, -1, -1):
            _recurse(remaining - i, depth - 1, prefix + [i])

    _recurse(total, n_dist, [])
    return pts


def _kmeans_nmi(
    embedding: np.ndarray,
    n_clusters: int,
    y_true: np.ndarray,
    random_state: int,
) -> tuple[float, np.ndarray]:
    """Cluster with KMeans, return (NMI(max), labels)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics.cluster import normalized_mutual_info_score

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(embedding)
    score = normalized_mutual_info_score(y_true, labels, average_method="max")
    return float(score), labels


def _leiden_nmi_sweep(
    umap_model: Any,
    y_true: np.ndarray,
    resolutions: Sequence[float],
    random_state: int,
) -> tuple[float, float, np.ndarray]:
    """Sweep Leiden CPM resolutions on a fitted UMAP graph; return best (NMI, res, labels).

    Uses ``scipy.sparse.triu(k=1)`` so each undirected edge appears once
    (the H1-H4 notebooks used ``.nonzero()`` which double-counts every edge).
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError as e:
        raise ImportError(
            "Leiden clustering requires igraph + leidenalg. "
            "Install with: pip install cumap[cluster]"
        ) from e
    from scipy import sparse
    from sklearn.metrics.cluster import normalized_mutual_info_score

    graph_matrix = umap_model.graph_
    coo = sparse.triu(graph_matrix, k=1).tocoo()
    edges = list(zip(coo.row.tolist(), coo.col.tolist()))
    weights = coo.data.tolist()

    g = ig.Graph(n=graph_matrix.shape[0], edges=edges, directed=False)
    g.es["weight"] = weights

    best_score = -np.inf
    best_res: float | None = None
    best_labels: np.ndarray | None = None
    for res in resolutions:
        partition = leidenalg.find_partition(
            g, leidenalg.CPMVertexPartition,
            resolution_parameter=res, weights="weight", seed=random_state,
        )
        labels = np.asarray(partition.membership)
        score = normalized_mutual_info_score(y_true, labels, average_method="max")
        if score > best_score:
            best_score = float(score)
            best_res = float(res)
            best_labels = labels
    return best_score, best_res, best_labels


def search_weights(
    distances: Sequence[np.ndarray],
    y_true: np.ndarray,
    cluster: str = "kmeans",
    weights_grid: Sequence[Sequence[float]] | None = None,
    leiden_resolutions: Sequence[float] | None = None,
    fusion: str = "sum",
    backend: Union[str, Callable] = "umap",
    n_clusters: int | None = None,
    random_state: int = 42,
    verbose: bool = False,
    **embed_kwargs: Any,
) -> dict:
    """Grid-search fusion weights (and optionally Leiden resolution).

    Supervised by NMI against ``y_true``.

    Parameters
    ----------
    distances : sequence of ndarray, each shape (n, n)
        Pairwise distance matrices to fuse.
    y_true : ndarray of shape (n,)
        Ground-truth integer labels. Required.
    cluster : {"kmeans", "leiden"}, default="kmeans"
        Clustering algorithm scored during search.
        - "kmeans": single-layer sweep over weights. UMAP -> KMeans -> NMI.
        - "leiden": two-layer sweep over weights x leiden_resolutions.
                    Requires ``cumap[cluster]`` extras (igraph + leidenalg).
    weights_grid : sequence of weight tuples, optional
        Candidate weight tuples (each must have ``len == len(distances)``).
        If None, an (N-1)-simplex grid with step ``0.1`` is generated for
        ``N = len(distances)`` distances (11 / 66 / 286 / 1001 points for
        N = 2 / 3 / 4 / 5). For ``N >= 5`` a warning is emitted; consider
        passing a sparser ``weights_grid`` explicitly to control cost.
    leiden_resolutions : sequence of float, optional
        CPM resolutions to sweep when ``cluster='leiden'``. If None, uses the
        16-point grid from the H1-H4 notebook.
    fusion : {"sum", "max"}, default="sum"
    backend : str or callable, default="umap"
        UMAP is required when ``cluster='leiden'`` (needs ``graph_``).
    n_clusters : int, optional
        KMeans cluster count. If None, uses ``len(unique(y_true))``.
    random_state : int, default=42
    verbose : bool, default=False
        If True, print per-weight progress.
    **embed_kwargs : forwarded to the embedding backend.

    Returns
    -------
    result : dict with keys
        - 'best_weights' : tuple of weights
        - 'best_score'   : float NMI
        - 'best_labels'  : ndarray of shape (n,)
        - 'best_embedding' : ndarray of shape (n, 2)
        - 'best_fused'   : ndarray of shape (n, n)
        - 'best_umap_model' : ``umap.UMAP`` on best fused distance, or None
        - 'best_resolution' : float (cluster='leiden') or None (cluster='kmeans')
        - 'all_results'  : list of dicts, one per grid point
    """
    if y_true is None:
        raise ValueError("y_true is required (search is supervised by NMI)")

    if cluster not in ("kmeans", "leiden"):
        raise ValueError(f"cluster must be 'kmeans' or 'leiden', got {cluster!r}")

    if weights_grid is None:
        n_dist = len(distances)
        if n_dist >= 5:
            import math
            import warnings
            n_pts = math.comb(11 + n_dist - 2, n_dist - 1)
            warnings.warn(
                f"Default simplex grid for {n_dist} distances at n_grid=11 has "
                f"{n_pts} points (grows combinatorially). "
                "Consider passing weights_grid explicitly.",
                stacklevel=2,
            )
        weights_grid = _default_simplex_grid(n_dist, n_grid=11)

    for w in weights_grid:
        if len(w) != len(distances):
            raise ValueError(
                f"Each weight tuple must have length {len(distances)}, got {len(w)}"
            )

    if n_clusters is None:
        n_clusters = int(len(np.unique(y_true)))

    use_umap = isinstance(backend, str) and backend.lower() == "umap"
    if cluster == "leiden" and not use_umap:
        raise ValueError(
            "cluster='leiden' requires backend='umap' (needs UMAP graph_)"
        )

    if leiden_resolutions is None:
        leiden_resolutions = DEFAULT_LEIDEN_RESOLUTIONS

    all_results: list[dict] = []
    best: dict = {
        "score": -np.inf,
        "weights": None,
        "labels": None,
        "embedding": None,
        "fused": None,
        "umap_model": None,
        "resolution": None,
    }

    for w in weights_grid:
        D_fused = fuse_distances(
            list(distances), weights=list(w), mode=fusion, normalize=True
        )
        umap_model = None
        if use_umap:
            embedding, umap_model = embed_umap(
                D_fused, random_state=random_state, return_model=True, **embed_kwargs
            )
        else:
            embedding = embed(
                D_fused, backend=backend, random_state=random_state, **embed_kwargs
            )

        if cluster == "kmeans":
            score, labels = _kmeans_nmi(embedding, n_clusters, y_true, random_state)
            this_res: float | None = None
        else:  # leiden
            score, this_res, labels = _leiden_nmi_sweep(
                umap_model, y_true, leiden_resolutions, random_state
            )

        all_results.append({
            "weights": tuple(w),
            "score": score,
            "resolution": this_res,
            "n_clusters": int(len(np.unique(labels))),
        })
        if verbose:
            extra = f" res={this_res:.4f}" if this_res is not None else ""
            print(f"  weights={tuple(round(x, 3) for x in w)}  nmi={score:.4f}{extra}")
        if score > best["score"]:
            best = {
                "score": score,
                "weights": tuple(w),
                "labels": labels,
                "embedding": embedding,
                "fused": D_fused,
                "umap_model": umap_model,
                "resolution": this_res,
            }

    return {
        "best_weights": best["weights"],
        "best_score": best["score"],
        "best_labels": best["labels"],
        "best_embedding": best["embedding"],
        "best_fused": best["fused"],
        "best_umap_model": best["umap_model"],
        "best_resolution": best["resolution"],
        "all_results": all_results,
    }
