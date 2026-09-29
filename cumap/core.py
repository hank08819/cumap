"""Main user-facing class: ``CTSNE``.

A sklearn-style estimator that wires up the full CUMAP pipeline:

  X  (raw counts, n_cells x n_genes, NO preprocessing)
       │
       ▼  compute distances (Yule / Fractional / L-Chebyshev, configurable)
  D_1, D_2, ...
       │
       ▼  optional weight grid search, scored by NMI (cluster='kmeans' or 'leiden')
  weights = (w_1, w_2, ...)
       │
       ▼  normalize + fuse
  D_fused
       │
       ▼  t-SNE or UMAP with metric='precomputed'
  embedding (n_cells x 2)
       │
       ▼  clustering (KMeans default, Leiden via cumap[cluster])
  labels (n_cells,)

When ``weights='auto'`` and ``cluster='leiden'``, the search is a two-layer
sweep over (weights x leiden_resolutions) — matching the paper experimental
protocol on the H1-H4 pancreas benchmark. With ``cluster='kmeans'`` the
search is a single layer over weights only.

Usage
-----
>>> from cumap import CTSNE
>>> model = CTSNE(distances=['yule', 'l_chebyshev'], weights='auto',
...               cluster='leiden')
>>> labels = model.fit_predict(X, y=y_true)
>>> print(model.weights_, model.best_resolution_, model.best_score_)
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Union

import numpy as np

from ._adapter import _extract_matrix
from .distances import compute_distance
from .embedding import embed, embed_umap
from .fusion import fuse_distances
from .search import search_weights


class CTSNE:
    """Explainable dimension reduction for scRNA-seq via distance fusion.

    Parameters
    ----------
    distances : sequence of str or callable, default=('yule', 'l_chebyshev')
        Names of distance metrics to fuse, or user-supplied callables
        ``f(X, **kwargs) -> (n, n) ndarray``. Names must be valid for
        ``cumap.distances.compute_distance``. The 2-distance default matches
        the best configuration on the paper's H1-H4 pancreas benchmark.
    weights : sequence of float or "auto", default="auto"
        Fusion weights. If "auto", grid search via ``search_weights``;
        ``y`` is required at fit time (search is supervised by NMI).
    cluster : {"kmeans", "leiden"}, default="kmeans"
        Clustering algorithm. Drives both the search-scoring criterion (when
        ``weights='auto'``) and the final ``labels_`` produced by
        ``fit_predict``. ``'leiden'`` requires ``pip install cumap[cluster]``.
    fusion : {"sum", "max"}, default="sum"
    backend : str or callable, default="umap"
        "tsne", "umap", or a callable ``f(D, random_state, **kwargs) -> emb``.
        Leiden clustering requires UMAP.
    weights_grid : sequence of weight tuples, optional
        Custom grid for weight search. For 2 distances, defaults to 11-point
        linear grid. Required for 3+ distances.
    leiden_resolutions : sequence of float, optional
        CPM resolutions swept when ``cluster='leiden'`` and ``weights='auto'``.
        If None, uses a 16-point grid matching the H1-H4 notebook.
    leiden_resolution : float, default=0.005
        CPM resolution used when ``cluster='leiden'`` and ``weights`` is
        fixed (no search). Ignored when ``weights='auto'``.
    n_components : int, default=2
    random_state : int, default=42
    fractional_f : float, default=0.25
    l_chebyshev_k : int, default=15
    perplexity : float, default=50.0
        t-SNE perplexity (used only when backend='tsne').
    n_neighbors : int, default=15
        UMAP n_neighbors (used only when backend='umap').
    layer : None | str, default=None
        Only used when ``fit`` / ``fit_transform`` / ``fit_predict`` is
        called with an AnnData object. ``None`` reads ``adata.X``; the
        string ``"raw"`` reads ``adata.raw.X``; any other string is
        interpreted as a key into ``adata.layers``.

    Attributes (set after ``fit``)
    ------------------------------
    distance_matrices_ : list of ndarray
    weights_ : tuple of float
        Final fusion weights (user-supplied or chosen by search).
    fused_distance_ : ndarray of shape (n, n)
    embedding_ : ndarray of shape (n, n_components)
    labels_ : ndarray of shape (n,)
        Set after ``fit_predict`` / ``fit_predict_precomputed``.
    best_score_ : float, only when weights="auto"
    best_resolution_ : float or None, only when weights="auto"
        Best Leiden CPM resolution found (None when cluster='kmeans').
    search_results_ : list of dict, only when weights="auto"
    """

    def __init__(
        self,
        distances: Sequence[Union[str, Callable]] = ("yule", "l_chebyshev"),
        weights: Union[Sequence[float], str] = "auto",
        cluster: str = "kmeans",
        fusion: str = "sum",
        backend: Union[str, Callable] = "umap",
        method: str = "auto",
        accel_threshold: int = 10000,
        leiden_backend: str = "cpu",
        accel_k: int = 30,
        accel_overfetch: int = 60,
        weights_grid: Sequence[Sequence[float]] | None = None,
        leiden_resolutions: Sequence[float] | None = None,
        leiden_resolution: float = 0.005,
        n_components: int = 2,
        random_state: int = 42,
        fractional_f: float = 0.25,
        l_chebyshev_k: int = 15,
        perplexity: float = 50.0,
        n_neighbors: int = 15,
        layer: str | None = None,
    ):
        if cluster not in ("kmeans", "leiden"):
            raise ValueError(f"cluster must be 'kmeans' or 'leiden', got {cluster!r}")
        if method not in ("auto", "exact", "accel"):
            raise ValueError(f"method must be 'auto', 'exact', or 'accel', got {method!r}")
        if leiden_backend not in ("cpu", "gpu"):
            raise ValueError(f"leiden_backend must be 'cpu' or 'gpu', got {leiden_backend!r}")
        self.distances = tuple(distances)
        self.weights = weights
        self.cluster = cluster
        self.fusion = fusion
        self.backend = backend
        self.method = method
        self.accel_threshold = accel_threshold
        self.leiden_backend = leiden_backend
        self.accel_k = accel_k
        self.accel_overfetch = accel_overfetch
        self.weights_grid = weights_grid
        self.leiden_resolutions = leiden_resolutions
        self.leiden_resolution = leiden_resolution
        self.n_components = n_components
        self.random_state = random_state
        self.fractional_f = fractional_f
        self.l_chebyshev_k = l_chebyshev_k
        self.perplexity = perplexity
        self.n_neighbors = n_neighbors
        self.layer = layer

    def _distance_kwargs(self, name_or_callable) -> dict:
        if callable(name_or_callable):
            return {}
        if name_or_callable == "fractional":
            return {"f": self.fractional_f}
        if name_or_callable in ("l_chebyshev", "l-chebyshev"):
            return {"k": self.l_chebyshev_k}
        return {}

    def _embed_kwargs(self) -> dict:
        if callable(self.backend):
            return {"n_components": self.n_components}
        if self.backend == "tsne":
            return {"n_components": self.n_components, "perplexity": self.perplexity}
        return {"n_components": self.n_components, "n_neighbors": self.n_neighbors}

    def _resolve_method(self, n: int) -> str:
        """Resolve method='auto' to 'exact' (small n) or 'accel' (large n)."""
        if self.method != "auto":
            return self.method
        return "accel" if n >= self.accel_threshold else "exact"

    def fit(self, X, y: np.ndarray | None = None) -> "CTSNE":
        """Compute distances from ``X`` then run the pipeline.

        ``X`` is ndarray, scipy sparse matrix, or AnnData (selected via
        ``self.layer``). ``y`` is required when ``weights='auto'``.

        ``method='auto'`` (default) routes ``n >= accel_threshold`` cells to the
        near-linear accelerated path and smaller datasets to the exact path.
        """
        X = _extract_matrix(X, layer=self.layer)
        self.method_ = self._resolve_method(X.shape[0])
        if self.method_ == "accel":
            return self._fit_accel(X, y)
        distance_matrices = [
            compute_distance(X, name, **self._distance_kwargs(name))
            for name in self.distances
        ]
        return self.fit_precomputed(distance_matrices, y=y)

    def _fit_accel(self, X, y: np.ndarray | None = None) -> "CTSNE":
        """Accelerated sparse-ANN-graph path (see ``cumap.accel``).

        Builds one fuzzy k-NN graph per distance, then either searches the
        fusion-weight grid (``weights='auto'``) or fuses fixed weights, and
        clusters the fused graph (Leiden CPU/GPU or UMAP+KMeans).
        """
        from . import accel

        if any(callable(d) for d in self.distances):
            raise ValueError("accel path supports only named distances "
                             "(yule/fractional/l_chebyshev), not callables")
        n = X.shape[0]
        graphs = accel.build_distance_graphs(
            X, self.distances, fractional_f=self.fractional_f,
            l_chebyshev_k=self.l_chebyshev_k, k=self.accel_k,
            overfetch=self.accel_overfetch, random_state=self.random_state,
        )
        self._accel_graphs_ = graphs
        self._is_accel_ = True
        self._umap_model_ = None
        self.best_resolution_ = None

        gpu = self.leiden_backend == "gpu"
        if self.leiden_resolutions is not None:
            resolutions = tuple(self.leiden_resolutions)
        elif gpu:
            resolutions = accel.DEFAULT_GPU_RESOLUTIONS
        else:
            resolutions = accel.DEFAULT_LEIDEN_RESOLUTIONS

        def cluster_graph(graph):
            """Cluster a fused graph -> (nmi, labels, embedding_or_None). Needs y."""
            if self.cluster == "leiden":
                sweep = accel.leiden_sweep_gpu if gpu else accel.leiden_sweep_cpu
                v, lab = sweep(graph, y, resolutions, self.random_state)
                return v, lab, None
            K = int(len(np.unique(y)))
            return accel.kmeans_sweep(graph, y, K, n,
                                      n_neighbors=self.n_neighbors,
                                      random_state=self.random_state)

        if isinstance(self.weights, str):
            if self.weights != "auto":
                raise ValueError(f"weights must be a sequence or 'auto', got {self.weights!r}")
            if y is None:
                raise ValueError("weights='auto' requires y (search is supervised by NMI)")
            grid = self.weights_grid or accel.simplex_grid(len(self.distances), 11)
            best = (-1.0, None, None, None, None)  # nmi, w, labels, emb, graph
            for w in grid:
                graph = accel.fuse_graphs(graphs, w)
                v, lab, emb = cluster_graph(graph)
                if v > best[0]:
                    best = (v, tuple(w), lab, emb, graph)
            self.best_score_, self.weights_ = best[0], best[1]
            self._predicted_labels_, fused = best[2], best[4]
            emb = best[3]
            self.search_results_ = None
        else:
            self.weights_ = tuple(self.weights)
            fused = accel.fuse_graphs(graphs, self.weights_)
            if y is not None:
                _, self._predicted_labels_, emb = cluster_graph(fused)
            else:
                self._predicted_labels_, emb = None, None

        self._accel_fused_ = fused
        self.embedding_ = emb if emb is not None else accel.graph_embedding(
            fused, n, n_components=self.n_components,
            n_neighbors=self.n_neighbors, random_state=self.random_state)
        return self

    def _validate_precomputed(
        self, mats: Sequence[np.ndarray]
    ) -> list[np.ndarray]:
        """Validate shape / symmetry / diagonal; auto-symmetrize triu-only storage.

        Some large-dataset pipelines store only the upper triangle to halve
        compute and disk (the lower triangle is left as zeros or uninitialized
        garbage). This routine auto-detects that case on a 1000x1000 corner
        and rebuilds the full symmetric matrix in memory, emitting a warning.
        Genuinely asymmetric input (e.g. an actual bug) still raises.

        Returns the list of (possibly symmetrized) matrices.
        """
        import warnings
        fixed = []
        for i, D in enumerate(mats):
            if D.ndim != 2 or D.shape[0] != D.shape[1]:
                raise ValueError(
                    f"distance_matrices[{i}] must be square 2D, got shape {D.shape}"
                )
            n = D.shape[0]
            # Corner sample: avoids materializing an n*n temporary for big mats
            s = min(1000, n)
            sub = D[:s, :s]
            if np.allclose(sub, sub.T, atol=1e-6):
                D_out = D  # already symmetric; fast path
            else:
                # Decide: triu-only storage vs genuine asymmetry.
                upper = sub[np.triu_indices(s, k=1)]
                lower = sub[np.tril_indices(s, k=-1)]
                u_max = float(np.max(np.abs(upper))) if upper.size else 0.0
                l_max = float(np.max(np.abs(lower))) if lower.size else 0.0
                bad_lower = bool(np.isnan(lower).any() or np.isinf(lower).any())
                is_triu = bad_lower or l_max <= u_max * 1e-3 or l_max >= u_max * 1e3
                if not is_triu:
                    raise ValueError(
                        f"distance_matrices[{i}] is not symmetric "
                        f"(corner max |D-D.T| = {np.max(np.abs(sub - sub.T)):.2e}). "
                        "Lower-triangular values are within the same magnitude as "
                        "the upper triangle, so auto-symmetrize was not triggered."
                    )
                warnings.warn(
                    f"distance_matrices[{i}]: detected upper-triangular storage; "
                    "auto-symmetrizing via D = triu(D) + triu(D).T - diag.",
                    UserWarning, stacklevel=3,
                )
                Dt = np.triu(D)
                D_out = Dt + Dt.T - np.diag(np.diag(Dt))
            if not np.allclose(np.diag(D_out), 0.0, atol=1e-6):
                raise ValueError(
                    f"distance_matrices[{i}] has non-zero diagonal "
                    f"(max |diag| = {np.max(np.abs(np.diag(D_out))):.2e})"
                )
            fixed.append(D_out)
        return fixed

    def fit_precomputed(
        self,
        distance_matrices: Sequence[np.ndarray],
        y: np.ndarray | None = None,
    ) -> "CTSNE":
        """Fit when distance matrices are already computed.

        ``distance_matrices`` must be in the same order as ``self.distances``.
        """
        if len(distance_matrices) != len(self.distances):
            raise ValueError(
                f"Expected {len(self.distances)} distance matrices (one per name "
                f"in self.distances={self.distances!r}), got {len(distance_matrices)}"
            )
        shapes = {D.shape for D in distance_matrices}
        if len(shapes) != 1:
            raise ValueError(
                f"All distance matrices must share shape; got {shapes}"
            )
        distance_matrices = self._validate_precomputed(distance_matrices)

        self.distance_matrices_ = [np.asarray(D, dtype=np.float64) for D in distance_matrices]
        self._umap_model_ = None
        self._predicted_labels_ = None
        self.best_resolution_ = None
        self._is_accel_ = False
        self.method_ = "exact"

        if isinstance(self.weights, str):
            if self.weights != "auto":
                raise ValueError(
                    f"weights must be a sequence or 'auto', got {self.weights!r}"
                )
            if y is None:
                raise ValueError(
                    "weights='auto' requires y (search is supervised by NMI)"
                )
            result = search_weights(
                distances=self.distance_matrices_,
                y_true=y,
                cluster=self.cluster,
                weights_grid=self.weights_grid,
                leiden_resolutions=self.leiden_resolutions,
                fusion=self.fusion,
                backend=self.backend,
                random_state=self.random_state,
                **self._embed_kwargs(),
            )
            self.weights_ = result["best_weights"]
            self.best_score_ = result["best_score"]
            self.best_resolution_ = result["best_resolution"]
            self.search_results_ = result["all_results"]
            self.embedding_ = result["best_embedding"]
            self._predicted_labels_ = result["best_labels"]
            self.fused_distance_ = result["best_fused"]
            self._umap_model_ = result["best_umap_model"]
        else:
            self.weights_ = tuple(self.weights)
            self.fused_distance_ = fuse_distances(
                self.distance_matrices_,
                weights=list(self.weights_),
                mode=self.fusion,
                normalize=True,
            )
            use_umap = isinstance(self.backend, str) and self.backend.lower() == "umap"
            if use_umap:
                self.embedding_, self._umap_model_ = embed_umap(
                    self.fused_distance_,
                    random_state=self.random_state,
                    return_model=True,
                    **self._embed_kwargs(),
                )
            else:
                self.embedding_ = embed(
                    self.fused_distance_,
                    backend=self.backend,
                    random_state=self.random_state,
                    **self._embed_kwargs(),
                )

        return self

    def fit_transform(self, X, y: np.ndarray | None = None) -> np.ndarray:
        """Fit and return the 2D embedding."""
        self.fit(X, y)
        return self.embedding_

    def fit_transform_precomputed(
        self,
        distance_matrices: Sequence[np.ndarray],
        y: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fit from precomputed distances and return the 2D embedding."""
        self.fit_precomputed(distance_matrices, y)
        return self.embedding_

    def _do_cluster(self, n_clusters: int | None, y: np.ndarray | None) -> np.ndarray:
        """Run the clustering chosen at init on ``self.embedding_``."""
        if getattr(self, "_is_accel_", False):
            # Accel path clusters the fused graph directly (Leiden) or its
            # embedding (KMeans). When y was available at fit time the labels
            # are already chosen; otherwise cluster now.
            if self._predicted_labels_ is not None:
                self.labels_ = self._predicted_labels_
                return self.labels_
            from . import accel

            if self.cluster == "leiden":
                self.labels_ = accel.leiden_labels_cpu(
                    self._accel_fused_, self.leiden_resolution, self.random_state)
            else:
                if n_clusters is None:
                    if y is None:
                        raise ValueError("n_clusters required when y is None")
                    n_clusters = int(len(np.unique(y)))
                from sklearn.cluster import KMeans

                self.labels_ = KMeans(n_clusters=n_clusters,
                                      random_state=self.random_state,
                                      n_init=10).fit_predict(self.embedding_)
            return self.labels_

        if self.cluster == "kmeans":
            if n_clusters is None:
                if y is None:
                    raise ValueError("n_clusters required when y is None")
                n_clusters = int(len(np.unique(y)))
            # If search_weights already ran KMeans with matching cluster count,
            # reuse those labels.
            if (
                self._predicted_labels_ is not None
                and isinstance(self.weights, str)
                and self.weights == "auto"
                and y is not None
                and int(len(np.unique(y))) == n_clusters
            ):
                self.labels_ = self._predicted_labels_
                return self.labels_

            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            self.labels_ = km.fit_predict(self.embedding_)
            return self.labels_

        # cluster == 'leiden'
        if self._predicted_labels_ is not None and isinstance(self.weights, str) and self.weights == "auto":
            # search_weights already ran the full (weights x resolution) sweep;
            # the best Leiden labels are stored.
            self.labels_ = self._predicted_labels_
            return self.labels_
        self.labels_ = self._leiden_single(self.leiden_resolution)
        return self.labels_

    def fit_predict(
        self,
        X,
        y: np.ndarray | None = None,
        n_clusters: int | None = None,
    ) -> np.ndarray:
        """Fit, cluster the embedding using ``self.cluster``, return labels."""
        self.fit(X, y)
        return self._do_cluster(n_clusters, y)

    def fit_predict_precomputed(
        self,
        distance_matrices: Sequence[np.ndarray],
        y: np.ndarray | None = None,
        n_clusters: int | None = None,
    ) -> np.ndarray:
        """Fit from precomputed distances, cluster, return labels."""
        self.fit_precomputed(distance_matrices, y)
        return self._do_cluster(n_clusters, y)

    def _leiden_single(self, resolution: float) -> np.ndarray:
        """Run a single Leiden CPM partition on the UMAP fuzzy graph.

        Requires ``cumap[cluster]`` extras (igraph + leidenalg).
        """
        if not (isinstance(self.backend, str) and self.backend == "umap"):
            raise ValueError("Leiden clustering requires backend='umap'")
        try:
            import igraph as ig
            import leidenalg
        except ImportError as e:
            raise ImportError(
                "Leiden clustering requires igraph + leidenalg. "
                "Install with: pip install cumap[cluster]"
            ) from e
        from scipy import sparse

        if getattr(self, "_umap_model_", None) is not None:
            umap_model = self._umap_model_
        else:
            _, umap_model = embed_umap(
                self.fused_distance_,
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
                return_model=True,
            )
        graph_matrix = umap_model.graph_

        # Take only upper triangle to avoid double-counting each undirected edge.
        coo = sparse.triu(graph_matrix, k=1).tocoo()
        edges = list(zip(coo.row.tolist(), coo.col.tolist()))
        weights = coo.data.tolist()

        g = ig.Graph(n=graph_matrix.shape[0], edges=edges, directed=False)
        g.es["weight"] = weights

        partition = leidenalg.find_partition(
            g,
            leidenalg.CPMVertexPartition,
            resolution_parameter=resolution,
            weights="weight",
            seed=self.random_state,
        )
        return np.array(partition.membership)

    def score(self, y_true: np.ndarray) -> float:
        """Compute NMI(y_true, self.labels_) using ``average_method='max'``."""
        from sklearn.metrics.cluster import normalized_mutual_info_score

        if not hasattr(self, "labels_"):
            raise RuntimeError("Call fit_predict before score")
        return float(
            normalized_mutual_info_score(y_true, self.labels_, average_method="max")
        )
