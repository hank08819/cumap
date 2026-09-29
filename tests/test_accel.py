"""Tests for the accelerated sparse-ANN-graph path (``cumap.accel`` + method routing).

Small synthetic data (60 cells) needs tiny over-fetch/k since pynndescent
asks for n_neighbors+1 <= n_cells.
"""

import numpy as np
import pytest

from cumap import CTSNE, accel

# Tiny ANN params so 60-cell synthetic data does not exceed n_cells.
ACCEL_KW = dict(accel_k=8, accel_overfetch=16, n_neighbors=8, l_chebyshev_k=8)


class TestSimplexGrid:
    def test_two_distances(self):
        g = accel.simplex_grid(2, 11)
        assert len(g) == 11
        assert (1.0, 0.0) in g and (0.0, 1.0) in g
        assert all(abs(sum(w) - 1.0) < 1e-9 for w in g)

    def test_three_distances(self):
        g = accel.simplex_grid(3, 11)
        assert len(g) == 66
        assert all(abs(sum(w) - 1.0) < 1e-9 for w in g)


class TestGraphBuilders:
    def test_build_distance_graphs(self, small_data):
        X, _ = small_data
        graphs = accel.build_distance_graphs(
            X, ["yule", "fractional", "l_chebyshev"],
            k=8, overfetch=16, l_chebyshev_k=8, random_state=42)
        assert len(graphs) == 3
        for g in graphs:
            assert g.shape == (X.shape[0], X.shape[0])
            assert g.nnz > 0

    def test_fuse_graphs(self, small_data):
        X, _ = small_data
        graphs = accel.build_distance_graphs(
            X, ["yule", "l_chebyshev"], k=8, overfetch=16,
            l_chebyshev_k=8, random_state=42)
        fused = accel.fuse_graphs(graphs, [0.7, 0.3])
        assert fused.shape == (X.shape[0], X.shape[0])

    def test_fuse_all_zero_raises(self, small_data):
        X, _ = small_data
        graphs = accel.build_distance_graphs(
            X, ["yule", "l_chebyshev"], k=8, overfetch=16,
            l_chebyshev_k=8, random_state=42)
        with pytest.raises(ValueError):
            accel.fuse_graphs(graphs, [0.0, 0.0])


class TestMethodRouting:
    def test_resolve_auto_small_is_exact(self):
        m = CTSNE(method="auto", accel_threshold=10000)
        assert m._resolve_method(1925) == "exact"

    def test_resolve_auto_large_is_accel(self):
        m = CTSNE(method="auto", accel_threshold=10000)
        assert m._resolve_method(46346) == "accel"

    def test_resolve_explicit(self):
        assert CTSNE(method="exact")._resolve_method(10 ** 9) == "exact"
        assert CTSNE(method="accel")._resolve_method(1) == "accel"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            CTSNE(method="bogus")

    def test_invalid_leiden_backend_raises(self):
        with pytest.raises(ValueError):
            CTSNE(leiden_backend="tpu")


class TestAccelPipeline:
    def test_accel_kmeans_fixed(self, small_data):
        X, y = small_data
        m = CTSNE(distances=["yule", "fractional", "l_chebyshev"],
                  weights=[0.8, 0.1, 0.1], cluster="kmeans", method="accel",
                  **ACCEL_KW)
        labels = m.fit_predict(X, y)
        assert labels.shape == (X.shape[0],)
        assert m.method_ == "accel"
        assert m.embedding_.shape == (X.shape[0], 2)
        # 3 well-separated clusters: accel should recover structure.
        assert m.score(y) > 0.5

    def test_accel_auto_routes_exact_on_small(self, small_data):
        X, y = small_data
        m = CTSNE(distances=["yule", "l_chebyshev"], weights=[0.9, 0.1],
                  cluster="kmeans", method="auto", l_chebyshev_k=8, n_neighbors=8)
        m.fit_predict(X, y)
        assert m.method_ == "exact"
        assert hasattr(m, "distance_matrices_")

    def test_accel_rejects_callable_distance(self, small_data):
        X, y = small_data
        bad = CTSNE(distances=[lambda Z: np.zeros((Z.shape[0], Z.shape[0]))],
                    weights=[1.0], method="accel")
        with pytest.raises(ValueError):
            bad.fit(X, y)

    def test_accel_leiden_fixed(self, small_data):
        pytest.importorskip("leidenalg")
        pytest.importorskip("igraph")
        X, y = small_data
        m = CTSNE(distances=["yule", "l_chebyshev"], weights=[0.7, 0.3],
                  cluster="leiden", method="accel", leiden_resolution=0.01,
                  **ACCEL_KW)
        labels = m.fit_predict(X, y)
        assert labels.shape == (X.shape[0],)
        assert m.method_ == "accel"

    def test_accel_auto_weight_search_kmeans(self, small_data):
        X, y = small_data
        m = CTSNE(distances=["yule", "l_chebyshev"], weights="auto",
                  cluster="kmeans", method="accel", **ACCEL_KW)
        m.fit_predict(X, y)
        assert m.method_ == "accel"
        assert hasattr(m, "best_score_")
        assert len(m.weights_) == 2
        assert abs(sum(m.weights_) - 1.0) < 1e-9
