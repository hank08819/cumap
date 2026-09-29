"""Near-linear accelerated pipeline for large datasets.

The default CTSNE path fuses *dense* ``n x n`` distance matrices, which is
``O(n^2)`` in time and memory and becomes impractical past a few tens of
thousands of cells. This module replaces it with a sparse approximate-nearest-
neighbour (ANN) graph pipeline that is empirically near-linear (measured
end-to-end exponent ~1.0-1.07 over 5k-71k cells):

    X  --(ANN over-fetch top-M, pynndescent)-->  candidate neighbours
       --(exact re-rank to true top-k)-->        sparse k-NN
       --(UMAP fuzzy_simplicial_set)-->          per-distance fuzzy graph
    weighted graph union (fusion)  -->           fused graph
       --(Leiden CPU/GPU  or  UMAP+KMeans)-->     labels

``pynndescent`` (NN-descent) is the only ANN backend used: it is graph-based
and does *not* rely on the triangle inequality, so it stays correct for the
non-metric distances here (binary Yule, fractional Minkowski p<1). KD-tree /
Ball-tree / Annoy / HNSW silently return wrong neighbours for p<1.

Accuracy vs the exact path (validated on 9 scRNA-seq datasets): a small loss
on tiny datasets (which do not need acceleration anyway) and a gain on large
ones. ``CTSNE(method='auto')`` therefore routes ``n < accel_threshold`` to the
exact path and ``n >= accel_threshold`` here.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import scipy.sparse as sp

# Validated defaults (poc benchmark): over-fetch 60 candidates, re-rank to 30.
DEFAULT_OVERFETCH = 60
DEFAULT_K = 30
# CPM resolution grid for the CPU Leiden sweep (matches the validated harness).
DEFAULT_LEIDEN_RESOLUTIONS = (
    0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.0015, 0.002, 0.0025,
    0.003, 0.004, 0.005, 0.0075, 0.01, 0.02, 0.05, 0.1,
)
# Modularity resolution grid for the GPU (cugraph) Leiden sweep.
DEFAULT_GPU_RESOLUTIONS = tuple(round(x, 4) for x in np.geomspace(0.05, 4.0, 24))


def simplex_grid(ndist, step=11):
    """Integer-lattice simplex of fusion weights: ndist=2 -> 11 pts, 3 -> 66."""
    import itertools

    div = step - 1
    return [tuple(c / div for c in combo)
            for combo in itertools.product(range(step), repeat=ndist)
            if sum(combo) == div]


def _distance_spec(name, X, Xbin, Xlow, frac_f):
    """Map a distance name to (data, pynndescent metric, metric_kwds)."""
    if name in ("l_chebyshev", "l-chebyshev"):
        return Xlow, "chebyshev", {}
    if name == "fractional":
        return X, "minkowski", {"p": float(frac_f)}
    if name == "yule":
        return Xbin, "yule", {}
    raise ValueError(f"accel path supports yule/fractional/l_chebyshev, got {name!r}")


def _ann_overfetch(data, metric, kwds, M, random_state):
    """pynndescent: fetch top-M approximate neighbours (excludes self)."""
    from pynndescent import NNDescent

    idx, _ = NNDescent(
        data, metric=metric, metric_kwds=kwds, n_neighbors=M + 1,
        random_state=random_state, n_jobs=-1,
    ).neighbor_graph
    return idx[:, 1:M + 1]


def _rerank(data, metric, kwds, aidx, k):
    """Exact-distance re-rank of each point's M candidates down to true top-k.

    NN-descent over-fetches; recomputing exact distances on the small candidate
    set and keeping the closest k pushes recall to ~1.0 while staying O(n*M*d).
    """
    from scipy.spatial.distance import cdist

    n = data.shape[0]
    cdkw = {"p": kwds["p"]} if metric == "minkowski" else {}
    ni = np.zeros((n, k), int)
    nd = np.zeros((n, k), np.float32)
    for i in range(n):
        cand = aidx[i]
        d = cdist(data[i:i + 1], data[cand], metric, **cdkw)[0]
        o = np.argsort(d)[:k]
        ni[i] = cand[o]
        nd[i] = d[o]
    return ni, nd


def _fuzzy_graph(knn_idx, knn_dist, n, random_state):
    """UMAP fuzzy_simplicial_set from a precomputed sparse k-NN -> CSR graph.

    Local per-point normalization inside fuzzy_simplicial_set replaces the
    global lambda_max distance normalization of the dense path.
    """
    import umap.umap_ as umap_

    si = np.arange(n)[:, None]
    ki = np.hstack([si, knn_idx]).astype(np.int64)
    kd = np.hstack([np.zeros((n, 1), np.float32), knn_dist.astype(np.float32)])
    g, _, _ = umap_.fuzzy_simplicial_set(
        X=np.zeros((n, 1)), n_neighbors=ki.shape[1], random_state=random_state,
        metric="precomputed", knn_indices=ki, knn_dists=kd,
    )
    return g.tocsr()


def build_distance_graphs(
    X, distances, *, fractional_f=0.25, l_chebyshev_k=15,
    k=DEFAULT_K, overfetch=DEFAULT_OVERFETCH, random_state=42,
):
    """Build one fuzzy k-NN graph per distance (the expensive ANN step).

    Returns a list of CSR graphs aligned with ``distances``. Cache and reuse
    these across a weight search — fusing them is cheap.
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    Xbin = (X > 0).astype(np.float32) if "yule" in distances else None
    Xlow = None
    if "l_chebyshev" in distances or "l-chebyshev" in distances:
        from scipy.sparse.linalg import svds

        kk = min(l_chebyshev_k, min(X.shape) - 1)
        U, s, Vt = svds(X.astype(np.float64), k=kk)
        Xlow = ((U * s) @ Vt).astype(np.float32)
    graphs = []
    for name in distances:
        data, metric, kwds = _distance_spec(name, X, Xbin, Xlow, fractional_f)
        aidx = _ann_overfetch(data, metric, kwds, overfetch, random_state)
        idx, dist = _rerank(data, metric, kwds, aidx, k)
        graphs.append(_fuzzy_graph(idx, dist, n, random_state))
    return graphs


def fuse_graphs(graphs, weights):
    """Weighted union of fuzzy graphs (skips zero-weight distances) -> CSR."""
    g = None
    for wi, gi in zip(weights, graphs):
        if wi == 0:
            continue
        g = wi * gi if g is None else g + wi * gi
    if g is None:
        raise ValueError("all weights are zero")
    return g.tocsr()


def _graph_to_ig(graph_csr):
    import igraph as ig

    coo = sp.triu(graph_csr, k=1).tocoo()
    g = ig.Graph(n=graph_csr.shape[0],
                 edges=list(zip(coo.row.tolist(), coo.col.tolist())),
                 directed=False)
    g.es["weight"] = coo.data.tolist()
    return g


def leiden_labels_cpu(graph_csr, resolution, random_state=42):
    """Single CPU Leiden CPM partition on a fused graph -> labels."""
    try:
        import leidenalg as la
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Leiden clustering requires igraph + leidenalg. "
            "Install with: pip install cumap[cluster]"
        ) from e
    g = _graph_to_ig(graph_csr)
    part = la.find_partition(g, la.CPMVertexPartition, weights="weight",
                             resolution_parameter=float(resolution),
                             seed=random_state)
    return np.asarray(part.membership)


def leiden_sweep_cpu(graph_csr, y, resolutions, random_state=42):
    """Sweep CPM resolutions on CPU, return (best_nmi, best_labels)."""
    from sklearn.metrics import normalized_mutual_info_score as nmi

    g = _graph_to_ig(graph_csr)
    import leidenalg as la

    best = (-1.0, None)
    for res in resolutions:
        lab = np.asarray(
            la.find_partition(g, la.CPMVertexPartition, weights="weight",
                              resolution_parameter=float(res),
                              seed=random_state).membership
        )
        v = float(nmi(y, lab, average_method="max"))
        if v > best[0]:
            best = (v, lab)
    return best


def leiden_sweep_gpu(graph_csr, y, resolutions, random_state=42):
    """Sweep modularity resolutions with cugraph Leiden on GPU.

    Much faster than CPU on large graphs but lands ~0.02-0.06 NMI below CPU
    leidenalg (the cugraph modularity objective is not identical to CPM/RB).
    Requires ``pip install cumap[gpu]`` (cudf + cugraph). Returns
    (best_nmi, best_labels).
    """
    try:
        import cudf
        import cugraph
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "GPU Leiden requires cudf + cugraph (RAPIDS), hosted on "
            "pypi.nvidia.com. Install with: "
            "pip install --extra-index-url=https://pypi.nvidia.com cumap[rapids]"
        ) from e
    from sklearn.metrics import normalized_mutual_info_score as nmi

    n = graph_csr.shape[0]
    coo = sp.triu(graph_csr, k=1).tocoo()
    gdf = cudf.DataFrame({
        "src": coo.row.astype(np.int32),
        "dst": coo.col.astype(np.int32),
        "weight": coo.data.astype(np.float32),
    })
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(gdf, source="src", destination="dst",
                         edge_attr="weight", renumber=True)
    best = (-1.0, None)
    for res in resolutions:
        parts, _ = cugraph.leiden(G, max_iter=1000, resolution=float(res))
        ps = parts.sort_values("vertex")
        lab = np.full(n, -1, int)
        lab[ps["vertex"].to_numpy()] = ps["partition"].to_numpy()
        v = float(nmi(y, lab, average_method="max"))
        if v > best[0]:
            best = (v, lab)
    return best


def graph_to_knn(graph_csr, k, n):
    """Recover a top-k sparse k-NN (index, 1-weight as distance) from a graph.

    Used to feed the kmeans path's UMAP embedding from a fused fuzzy graph.
    """
    g = graph_csr.tocsr()
    idx = np.zeros((n, k), np.int32)
    dist = np.zeros((n, k), np.float32)
    for i in range(n):
        s, e = g.indptr[i], g.indptr[i + 1]
        cols, w = g.indices[s:e], g.data[s:e]
        top = np.argpartition(w, -k)[-k:] if len(cols) >= k else np.arange(len(cols))
        c, ww = cols[top], w[top]
        idx[i, :len(c)] = c
        dist[i, :len(c)] = 1.0 - ww
        if len(c) < k:
            idx[i, len(c):] = i
    return idx, dist


def graph_embedding(graph_csr, n, *, n_components=2, n_neighbors=15, random_state=42):
    """2D UMAP embedding from a fused fuzzy graph (for viz / kmeans)."""
    import umap

    fidx, fdist = graph_to_knn(graph_csr, n_neighbors, n)
    return umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors,
        random_state=random_state, precomputed_knn=(fidx, fdist, None),
    ).fit_transform(np.zeros((n, 1), np.float32))


def kmeans_sweep(graph_csr, y, K, n, *, n_neighbors=15, random_state=42):
    """KMeans on the graph's UMAP embedding -> (nmi, labels, embedding)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score as nmi

    emb = graph_embedding(graph_csr, n, n_neighbors=n_neighbors,
                          random_state=random_state)
    lab = KMeans(n_clusters=K, random_state=random_state, n_init=10).fit_predict(emb)
    return float(nmi(y, lab, average_method="max")), lab, emb
