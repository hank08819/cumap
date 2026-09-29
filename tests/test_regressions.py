"""Regression tests for v0.0.2 bug fixes.

Each test corresponds to a bug found during the v0.0.1 audit. Names reference
the bug numbers in the audit report.
"""

import warnings

import numpy as np
import pytest

from cumap import CTSNE, fuse_distances
from cumap.distances import compute_distance, l_chebyshev_distance


class TestDefaultsRun:
    """Bug #1: CTSNE() with no args must not crash."""

    def test_default_ctor_fit(self, small_data):
        X, y = small_data
        model = CTSNE()  # default distances=(yule, l_chebyshev), weights='auto'
        emb = model.fit_transform(X, y=y)
        assert emb.shape == (X.shape[0], 2)
        assert isinstance(model.weights_, tuple)
        assert len(model.weights_) == 2

    def test_default_ctor_fit_predict(self, small_data):
        X, y = small_data
        model = CTSNE()
        labels = model.fit_predict(X, y=y)
        assert labels.shape == (X.shape[0],)


class TestLChebKAutoCap:
    """Bug #2: l_chebyshev_k=15 default must not crash on small data."""

    def test_k_capped_with_warning(self):
        X = np.random.default_rng(0).poisson(0.5, size=(10, 5)).astype(np.float64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            D = l_chebyshev_distance(X, k=15)
        assert any("capping" in str(rec.message) for rec in w)
        assert D.shape == (10, 10)

    def test_ctsne_default_k_on_small_data(self):
        X = np.random.default_rng(0).poisson(0.5, size=(15, 5)).astype(np.float64)
        y = np.tile([0, 1, 2], 5)
        # l_chebyshev_k=15 (default) but min(X.shape)=5 — should auto-cap
        model = CTSNE(distances=["yule", "l_chebyshev"], weights=[0.5, 0.5])
        emb = model.fit_transform(X, y=y)
        assert emb.shape == (15, 2)


class TestCallableBackendKwargs:
    """Bug #3: callable backend must not receive UMAP-only kwargs."""

    def test_callable_backend_no_n_neighbors(self, small_data):
        X, _ = small_data
        seen_kwargs = {}

        def my_backend(D, random_state=42, **kwargs):
            seen_kwargs.update(kwargs)
            return D[:, :2].copy()

        model = CTSNE(
            distances=["yule", "l_chebyshev"],
            weights=[0.9, 0.1],
            backend=my_backend,
            l_chebyshev_k=5,
        )
        model.fit_transform(X)
        assert "n_neighbors" not in seen_kwargs
        assert "perplexity" not in seen_kwargs


class TestNoUmapWarnings:
    """Bug #4: UMAP fits must not spam the two known-harmless warnings."""

    def test_no_precomputed_warning(self, small_data):
        X, y = small_data
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CTSNE(
                distances=["yule", "l_chebyshev"],
                weights=[0.9, 0.1],
                n_neighbors=5,
                l_chebyshev_k=5,
            ).fit_transform(X, y=y)
        msgs = [str(rec.message) for rec in w]
        assert not any("inverse_transform will be unavailable" in m for m in msgs)
        assert not any("n_jobs value 1 overridden" in m for m in msgs)


class TestFitPredictReusesSearchLabels:
    """Bug #5: fit_predict with auto+kmeans should reuse search's KMeans labels."""

    def test_reuse_when_n_clusters_matches(self, small_data):
        X, y = small_data
        model = CTSNE(
            distances=["yule", "l_chebyshev"],
            weights="auto",
            n_neighbors=5,
            l_chebyshev_k=5,
        )
        model.fit(X, y=y)
        labels_from_search = model._predicted_labels_.copy()
        # _do_cluster should return the cached labels (no new KMeans)
        labels = model._do_cluster(n_clusters=None, y=y)
        np.testing.assert_array_equal(labels, labels_from_search)


class TestPrecomputedValidation:
    """Bug #6: fit_precomputed must reject non-symmetric / non-zero-diagonal matrices."""

    def test_non_symmetric_rejected(self, small_data):
        from cumap.distances import yule_distance

        X, _ = small_data
        D_y = yule_distance(X)
        D_bad = D_y.copy()
        D_bad[0, 1] = 99.0  # break symmetry
        model = CTSNE(distances=["yule", "l_chebyshev"], weights=[0.5, 0.5])
        with pytest.raises(ValueError, match="symmetric"):
            model.fit_precomputed([D_bad, D_y])

    def test_non_zero_diagonal_rejected(self, small_data):
        from cumap.distances import yule_distance

        X, _ = small_data
        D_y = yule_distance(X)
        D_bad = D_y.copy()
        np.fill_diagonal(D_bad, 5.0)
        model = CTSNE(distances=["yule", "l_chebyshev"], weights=[0.5, 0.5])
        with pytest.raises(ValueError, match="diagonal"):
            model.fit_precomputed([D_bad, D_y])

    def test_triu_zeros_auto_symmetrize(self, small_data):
        """Upper-tri-only storage with zeroed lower triangle should be auto-fixed."""
        import warnings
        from cumap.distances import l_chebyshev_distance, yule_distance

        X, _ = small_data
        D_y = yule_distance(X)
        D_c = l_chebyshev_distance(X, k=5)
        D_y_triu = np.triu(D_y)  # lower triangle zeroed
        model = CTSNE(
            distances=["yule", "l_chebyshev"],
            weights=[0.5, 0.5],
            backend="umap",
            n_neighbors=5,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit_precomputed([D_y_triu, D_c])
        assert any("upper-triangular" in str(rec.message) for rec in w)
        # Symmetrized matrix should match the original symmetric one
        np.testing.assert_allclose(model.distance_matrices_[0], D_y, atol=1e-12)

    def test_triu_garbage_lower_auto_symmetrize(self, small_data):
        """Upper-tri-only storage with uninitialized-memory garbage lower
        (very large values, like Cortex P/F dist_fractional.npy) is auto-fixed."""
        import warnings
        from cumap.distances import l_chebyshev_distance, yule_distance

        X, _ = small_data
        D_y = yule_distance(X)
        D_c = l_chebyshev_distance(X, k=5)
        D_y_bad = np.triu(D_y).copy()
        # Stamp garbage on the lower triangle (e+16 like the real-world case)
        n = D_y_bad.shape[0]
        for i in range(1, n):
            for j in range(i):
                D_y_bad[i, j] = 1e16 * (i + j + 1)
        model = CTSNE(
            distances=["yule", "l_chebyshev"],
            weights=[0.5, 0.5],
            backend="umap",
            n_neighbors=5,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit_precomputed([D_y_bad, D_c])
        assert any("upper-triangular" in str(rec.message) for rec in w)
        np.testing.assert_allclose(model.distance_matrices_[0], D_y, atol=1e-12)


class TestCallableDistanceValidation:
    """Bug #7: compute_distance callable must validate return shape / nan."""

    def test_wrong_shape_rejected(self, small_data):
        X, _ = small_data

        def bad_metric(X):
            return np.zeros((X.shape[0] + 1, X.shape[0]))

        with pytest.raises(ValueError, match="shape"):
            compute_distance(X, bad_metric)

    def test_nan_rejected(self, small_data):
        X, _ = small_data

        def nan_metric(X):
            n = X.shape[0]
            D = np.zeros((n, n))
            D[0, 1] = np.nan
            return D

        with pytest.raises(ValueError, match="NaN"):
            compute_distance(X, nan_metric)


class TestSearchReturnsBestFusedAndModel:
    """Bug #9/#10: search_weights must return best_fused and best_umap_model for reuse."""

    def test_keys_present(self, small_data):
        from cumap.distances import l_chebyshev_distance, yule_distance
        from cumap.search import search_weights

        X, y = small_data
        D_y = yule_distance(X)
        D_c = l_chebyshev_distance(X, k=5)
        result = search_weights(
            [D_y, D_c], y_true=y, backend="umap", n_neighbors=5
        )
        assert "best_fused" in result
        assert "best_umap_model" in result
        assert result["best_fused"].shape == D_y.shape
        # UMAP backend → model is the fitted UMAP, exposing graph_
        assert hasattr(result["best_umap_model"], "graph_")


class TestFuseWeightValidation:
    """Bug #11: negative weights raise, non-normalized weights warn."""

    def test_negative_raises(self, tiny_distance):
        with pytest.raises(ValueError, match="Negative"):
            fuse_distances([tiny_distance, tiny_distance], weights=[-0.1, 1.1])

    def test_unnormalized_warns(self, tiny_distance):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fuse_distances([tiny_distance, tiny_distance], weights=[0.3, 0.4])
        assert any("sum to" in str(rec.message) for rec in w)

    def test_disabled_validation(self, tiny_distance):
        # negative weights allowed when validate_weights=False
        fuse_distances(
            [tiny_distance, tiny_distance],
            weights=[-0.5, 1.5],
            normalize=False,
            validate_weights=False,
        )
