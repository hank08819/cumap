"""Tests for cumap.distances."""

import numpy as np
import pytest

from cumap.distances import (
    compute_distance,
    fractional_distance,
    l_chebyshev_distance,
    low_rank_approx,
    yule_distance,
)


class TestYule:
    def test_shape(self, small_data):
        X, _ = small_data
        D = yule_distance(X)
        assert D.shape == (X.shape[0], X.shape[0])

    def test_symmetric(self, small_data):
        X, _ = small_data
        D = yule_distance(X)
        np.testing.assert_allclose(D, D.T)

    def test_zero_diagonal(self, small_data):
        X, _ = small_data
        D = yule_distance(X)
        np.testing.assert_allclose(np.diag(D), 0.0)

    def test_non_negative(self, small_data):
        X, _ = small_data
        D = yule_distance(X)
        assert (D >= 0).all()


class TestFractional:
    def test_shape(self, small_data):
        X, _ = small_data
        D = fractional_distance(X)
        assert D.shape == (X.shape[0], X.shape[0])

    def test_symmetric_and_zero_diag(self, small_data):
        X, _ = small_data
        D = fractional_distance(X)
        np.testing.assert_allclose(D, D.T)
        np.testing.assert_allclose(np.diag(D), 0.0)

    def test_f_param_changes_result(self, small_data):
        X, _ = small_data
        D_quarter = fractional_distance(X, f=0.25)
        D_one = fractional_distance(X, f=1.0)
        assert not np.allclose(D_quarter, D_one)

    def test_invalid_f_raises(self, small_data):
        X, _ = small_data
        with pytest.raises(ValueError):
            fractional_distance(X, f=0.0)


class TestLChebyshev:
    def test_shape(self, small_data):
        X, _ = small_data
        D = l_chebyshev_distance(X, k=5)
        assert D.shape == (X.shape[0], X.shape[0])

    def test_symmetric_and_zero_diag(self, small_data):
        X, _ = small_data
        D = l_chebyshev_distance(X, k=5)
        np.testing.assert_allclose(D, D.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)

    def test_k0_is_chebyshev_on_raw(self, small_data):
        from scipy.spatial.distance import pdist, squareform

        X, _ = small_data
        D_lcheb_k0 = l_chebyshev_distance(X, k=0)
        D_raw = squareform(pdist(X, "chebyshev"))
        np.testing.assert_allclose(D_lcheb_k0, D_raw, atol=1e-10)

    def test_k_too_large_warns_and_caps(self, small_data):
        X, _ = small_data
        with pytest.warns(UserWarning, match="capping"):
            D = l_chebyshev_distance(X, k=10_000)
        # Auto-capped to min(X.shape), so the call still produces a valid matrix
        assert D.shape == (X.shape[0], X.shape[0])


class TestLowRankApprox:
    def test_rank_at_most_k(self, small_data):
        X, _ = small_data
        X_low = low_rank_approx(X, k=5)
        assert np.linalg.matrix_rank(X_low, tol=1e-8) <= 5

    def test_k0_returns_input(self, small_data):
        X, _ = small_data
        X_low = low_rank_approx(X, k=0)
        np.testing.assert_array_equal(X_low, X)

    def test_full_rank_reconstructs(self, small_data):
        X, _ = small_data
        full_k = min(X.shape)
        X_low = low_rank_approx(X, k=full_k)
        np.testing.assert_allclose(X_low, X, atol=1e-8)


class TestComputeDistance:
    def test_dispatch_yule(self, small_data):
        X, _ = small_data
        D1 = compute_distance(X, "yule")
        D2 = yule_distance(X)
        np.testing.assert_allclose(D1, D2)

    def test_dispatch_fractional_kwargs(self, small_data):
        X, _ = small_data
        D1 = compute_distance(X, "fractional", f=0.5)
        D2 = fractional_distance(X, f=0.5)
        np.testing.assert_allclose(D1, D2)

    def test_alias_l_chebyshev(self, small_data):
        X, _ = small_data
        D1 = compute_distance(X, "l_chebyshev", k=5)
        D2 = compute_distance(X, "l-chebyshev", k=5)
        np.testing.assert_allclose(D1, D2)

    def test_unknown_distance_raises(self, small_data):
        X, _ = small_data
        with pytest.raises(ValueError, match="Unknown distance"):
            compute_distance(X, "euclidean_typo")


class TestCallableDistance:
    """compute_distance should accept user-supplied callables (sklearn-style)."""

    def test_callable_dispatched(self, small_data):
        X, _ = small_data

        def my_metric(X):
            # trivial: all-ones distance (but symmetric, zero diag)
            n = X.shape[0]
            D = np.ones((n, n))
            np.fill_diagonal(D, 0.0)
            return D

        D = compute_distance(X, my_metric)
        assert D.shape == (X.shape[0], X.shape[0])
        np.testing.assert_array_equal(np.diag(D), 0.0)

    def test_callable_with_kwargs(self, small_data):
        X, _ = small_data

        def scaled_metric(X, scale=1.0):
            n = X.shape[0]
            D = scale * np.ones((n, n))
            np.fill_diagonal(D, 0.0)
            return D

        D = compute_distance(X, scaled_metric, scale=2.5)
        # off-diagonal entries should be 2.5
        off_diag = D[np.triu_indices_from(D, k=1)]
        np.testing.assert_allclose(off_diag, 2.5)
