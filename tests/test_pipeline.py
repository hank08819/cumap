"""End-to-end tests for the CTSNE class."""

import numpy as np
import pytest

from cumap import CTSNE


class TestCTSNEFixedWeights:
    def test_fit_transform_shape(self, small_data):
        X, _ = small_data
        model = CTSNE(
            distances=["yule", "l_chebyshev"],
            weights=[0.9, 0.1],
            backend="umap",
            n_neighbors=5,
            l_chebyshev_k=5,
        )
        emb = model.fit_transform(X)
        assert emb.shape == (X.shape[0], 2)
        assert hasattr(model, "weights_")
        assert hasattr(model, "embedding_")
        assert hasattr(model, "fused_distance_")
        assert hasattr(model, "distance_matrices_")
        assert len(model.distance_matrices_) == 2

    def test_tsne_backend(self, small_data):
        X, _ = small_data
        model = CTSNE(
            distances=["yule", "l_chebyshev"],
            weights=[0.9, 0.1],
            backend="tsne",
            perplexity=10,
            l_chebyshev_k=5,
        )
        emb = model.fit_transform(X)
        assert emb.shape == (X.shape[0], 2)

    def test_3_distances(self, small_data):
        X, _ = small_data
        model = CTSNE(
            distances=["yule", "fractional", "l_chebyshev"],
            weights=[0.9, 0.05, 0.05],
            backend="umap",
            n_neighbors=5,
            l_chebyshev_k=5,
        )
        emb = model.fit_transform(X)
        assert emb.shape == (X.shape[0], 2)


class TestCTSNEAuto:
    def test_auto_weights_kmeans(self, small_data):
        X, y = small_data
        model = CTSNE(
            distances=["yule", "l_chebyshev"],
            weights="auto",
            cluster="kmeans",
            backend="umap",
            n_neighbors=5,
            l_chebyshev_k=5,
        )
        emb = model.fit_transform(X, y=y)
        assert emb.shape == (X.shape[0], 2)
        assert isinstance(model.best_score_, float)
        assert len(model.search_results_) == 11
        assert model.best_resolution_ is None

    def test_auto_requires_y(self, small_data):
        X, _ = small_data
        model = CTSNE(
            distances=["yule", "l_chebyshev"],
            weights="auto",
            backend="umap",
            n_neighbors=5,
            l_chebyshev_k=5,
        )
        with pytest.raises(ValueError, match="y"):
            model.fit_transform(X)

    def test_auto_weights_leiden(self, small_data):
        pytest.importorskip("leidenalg")
        pytest.importorskip("igraph")
        X, y = small_data
        model = CTSNE(
            distances=["yule", "l_chebyshev"],
            weights="auto",
            cluster="leiden",
            backend="umap",
            n_neighbors=5,
            l_chebyshev_k=5,
        )
        emb = model.fit_transform(X, y=y)
        assert emb.shape == (X.shape[0], 2)
        # Two-layer sweep populates best_resolution_
        assert model.best_resolution_ is not None
        assert 0.0 < model.best_resolution_ <= 0.1


class TestCTSNEFitPredict:
    def test_kmeans_labels(self, small_data):
        X, y = small_data
        model = CTSNE(
            distances=["yule", "l_chebyshev"],
            weights=[0.9, 0.1],
            cluster="kmeans",
            backend="umap",
            n_neighbors=5,
            l_chebyshev_k=5,
        )
        labels = model.fit_predict(X, y=y)
        assert labels.shape == (X.shape[0],)
        assert set(np.unique(labels).tolist()) <= {0, 1, 2}
        # Synthetic data is well-separated → NMI should be high
        nmi = model.score(y)
        assert nmi > 0.5

    def test_kmeans_needs_n_clusters_without_y(self, small_data):
        X, _ = small_data
        model = CTSNE(
            distances=["yule", "l_chebyshev"],
            weights=[0.9, 0.1],
            cluster="kmeans",
            backend="umap",
            n_neighbors=5,
            l_chebyshev_k=5,
        )
        with pytest.raises(ValueError, match="n_clusters"):
            model.fit_predict(X)

    def test_leiden_labels(self, small_data):
        pytest.importorskip("leidenalg")
        pytest.importorskip("igraph")
        X, y = small_data
        model = CTSNE(
            distances=["yule", "l_chebyshev"],
            weights=[0.9, 0.1],
            cluster="leiden",
            backend="umap",
            n_neighbors=5,
            l_chebyshev_k=5,
            leiden_resolution=0.01,
        )
        labels = model.fit_predict(X, y=y)
        assert labels.shape == (X.shape[0],)


class TestCTSNEValidation:
    def test_invalid_weights_string(self, small_data):
        X, _ = small_data
        model = CTSNE(weights="random", l_chebyshev_k=5)
        with pytest.raises(ValueError, match="'auto'"):
            model.fit(X)

    def test_invalid_cluster_method(self):
        # cluster is validated at __init__ time now
        with pytest.raises(ValueError, match="kmeans.*leiden"):
            CTSNE(cluster="dbscan")

    def test_1d_X_raises(self):
        model = CTSNE()
        with pytest.raises(ValueError, match="2D"):
            model.fit(np.zeros(10))


class TestCTSNECallable:
    """CTSNE should accept callables in the distances list (sklearn-style)."""

    def test_distances_list_with_callable(self, small_data):
        from cumap.distances import yule_distance

        X, _ = small_data

        def my_metric(X):
            return yule_distance(X)

        model = CTSNE(
            distances=["yule", my_metric],
            weights=[0.5, 0.5],
            backend="umap",
            n_neighbors=5,
        )
        emb = model.fit_transform(X)
        assert emb.shape == (X.shape[0], 2)
        assert len(model.distance_matrices_) == 2


class TestCTSNEFitPrecomputed:
    """fit_precomputed should give the same embedding as fit, skipping distance computation."""

    def _make_model(self):
        return CTSNE(
            distances=["yule", "l_chebyshev"],
            weights=[0.9, 0.1],
            backend="umap",
            n_neighbors=5,
            l_chebyshev_k=5,
            random_state=42,
        )

    def test_matches_fit(self, small_data):
        from cumap.distances import l_chebyshev_distance, yule_distance

        X, _ = small_data
        D_y = yule_distance(X)
        D_c = l_chebyshev_distance(X, k=5)

        m_full = self._make_model()
        m_full.fit_transform(X)

        m_pre = self._make_model()
        m_pre.fit_transform_precomputed([D_y, D_c])

        np.testing.assert_allclose(m_full.embedding_, m_pre.embedding_, atol=1e-8)
        np.testing.assert_allclose(
            m_full.fused_distance_, m_pre.fused_distance_, atol=1e-8
        )

    def test_wrong_count_raises(self, small_data):
        from cumap.distances import yule_distance

        X, _ = small_data
        D = yule_distance(X)
        with pytest.raises(ValueError, match="Expected 2 distance"):
            self._make_model().fit_precomputed([D])

    def test_shape_mismatch_raises(self, small_data):
        from cumap.distances import yule_distance

        X, _ = small_data
        D1 = yule_distance(X)
        D2 = np.eye(D1.shape[0] + 1)
        with pytest.raises(ValueError, match="shape"):
            self._make_model().fit_precomputed([D1, D2])

    def test_fit_predict_precomputed(self, small_data):
        from cumap.distances import l_chebyshev_distance, yule_distance

        X, y = small_data
        D_y = yule_distance(X)
        D_c = l_chebyshev_distance(X, k=5)
        m = self._make_model()
        labels = m.fit_predict_precomputed([D_y, D_c], y=y)
        assert labels.shape == (X.shape[0],)
        assert m.score(y) > 0.5

    def test_auto_weights_precomputed(self, small_data):
        from cumap.distances import l_chebyshev_distance, yule_distance

        X, y = small_data
        D_y = yule_distance(X)
        D_c = l_chebyshev_distance(X, k=5)
        m = CTSNE(
            distances=["yule", "l_chebyshev"],
            weights="auto",
            backend="umap",
            n_neighbors=5,
        )
        emb = m.fit_transform_precomputed([D_y, D_c], y=y)
        assert emb.shape == (X.shape[0], 2)
        assert hasattr(m, "best_score_")
        assert len(m.search_results_) == 11
