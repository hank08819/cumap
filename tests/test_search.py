"""Tests for cumap.search."""

import numpy as np
import pytest

from cumap.distances import fractional_distance, l_chebyshev_distance, yule_distance
from cumap.search import _default_simplex_grid, search_weights


@pytest.fixture
def two_distances(small_data):
    X, _ = small_data
    D_y = yule_distance(X)
    D_c = l_chebyshev_distance(X, k=5)
    return [D_y, D_c]


class TestSearchWeightsKMeans:
    def test_default_grid_2d(self, two_distances, small_data):
        _, y = small_data
        result = search_weights(
            two_distances, y_true=y, backend="umap", n_neighbors=5
        )
        # 11-point grid → 11 result rows
        assert len(result["all_results"]) == 11
        assert isinstance(result["best_weights"], tuple)
        assert len(result["best_weights"]) == 2
        np.testing.assert_allclose(sum(result["best_weights"]), 1.0, atol=1e-8)
        # NMI is in [0, 1]
        assert 0.0 <= result["best_score"] <= 1.0
        assert result["best_embedding"].shape == (small_data[0].shape[0], 2)
        # kmeans path → no resolution
        assert result["best_resolution"] is None

    def test_custom_grid(self, two_distances, small_data):
        _, y = small_data
        custom = [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]
        result = search_weights(
            two_distances,
            y_true=y,
            cluster="kmeans",
            weights_grid=custom,
            backend="umap",
            n_neighbors=5,
        )
        assert len(result["all_results"]) == 3


class TestSearchWeightsLeiden:
    """Two-layer sweep: 11 weights x 16 resolutions, scored by NMI."""

    def test_leiden_two_layer_sweep(self, two_distances, small_data):
        pytest.importorskip("leidenalg")
        pytest.importorskip("igraph")
        _, y = small_data
        result = search_weights(
            two_distances,
            y_true=y,
            cluster="leiden",
            backend="umap",
            n_neighbors=5,
        )
        # 11 weight points (each with internal resolution sweep)
        assert len(result["all_results"]) == 11
        # best_resolution must be one of the swept values
        assert result["best_resolution"] is not None
        assert 0.0 < result["best_resolution"] <= 0.1
        assert 0.0 <= result["best_score"] <= 1.0

    def test_leiden_custom_resolutions(self, two_distances, small_data):
        pytest.importorskip("leidenalg")
        pytest.importorskip("igraph")
        _, y = small_data
        result = search_weights(
            two_distances,
            y_true=y,
            cluster="leiden",
            leiden_resolutions=[0.001, 0.005, 0.01],
            backend="umap",
            n_neighbors=5,
        )
        assert result["best_resolution"] in (0.001, 0.005, 0.01)

    def test_leiden_requires_umap(self, two_distances, small_data):
        _, y = small_data
        with pytest.raises(ValueError, match="UMAP"):
            search_weights(
                two_distances, y_true=y, cluster="leiden", backend="tsne"
            )


class TestSearchWeightsValidation:
    def test_y_required(self, two_distances):
        with pytest.raises(ValueError, match="y_true"):
            search_weights(two_distances, y_true=None)

    def test_unknown_cluster(self, two_distances, small_data):
        _, y = small_data
        with pytest.raises(ValueError, match="kmeans.*leiden"):
            search_weights(two_distances, y_true=y, cluster="dbscan")

    def test_three_distances_default_grid(self, two_distances, small_data):
        # N=3 distances auto-generates a 2-simplex grid (66 points at n_grid=11)
        X, y = small_data
        D3 = [*two_distances, fractional_distance(X)]
        result = search_weights(
            D3, y_true=y, backend="umap", n_neighbors=5
        )
        assert len(result["all_results"]) == 66
        assert len(result["best_weights"]) == 3
        np.testing.assert_allclose(sum(result["best_weights"]), 1.0, atol=1e-8)
        assert all(w >= 0 for w in result["best_weights"])


class TestDefaultSimplexGrid:
    """Unit tests for _default_simplex_grid (general N-distance lattice)."""

    @pytest.mark.parametrize("n_dist,expected", [(2, 11), (3, 66), (4, 286)])
    def test_point_count_n_grid_11(self, n_dist, expected):
        # C(n_grid + n_dist - 2, n_dist - 1) at n_grid=11
        pts = _default_simplex_grid(n_dist, n_grid=11)
        assert len(pts) == expected

    @pytest.mark.parametrize("n_dist", [2, 3, 4])
    def test_simplex_invariants(self, n_dist):
        # Each point: length n_dist, non-negative entries, sums to exactly 1.0
        for w in _default_simplex_grid(n_dist, n_grid=11):
            assert len(w) == n_dist
            assert all(x >= 0 for x in w)
            np.testing.assert_allclose(sum(w), 1.0, atol=1e-8)

    def test_2dist_matches_legacy(self):
        # N=2 must reproduce the historical 11-point 1-simplex exactly
        pts = _default_simplex_grid(2, n_grid=11)
        step = 0.1
        expected = [(1.0 - i * step, i * step) for i in range(11)]
        for got, exp in zip(pts, expected):
            np.testing.assert_allclose(got, exp, atol=1e-12)

    def test_invalid_args(self):
        with pytest.raises(ValueError, match="n_dist"):
            _default_simplex_grid(1, n_grid=11)
        with pytest.raises(ValueError, match="n_grid"):
            _default_simplex_grid(3, n_grid=1)
