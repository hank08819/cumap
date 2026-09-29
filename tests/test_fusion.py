"""Tests for cumap.fusion."""

import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from cumap.fusion import fuse_distances, normalize_distance


class TestNormalize:
    def test_spectral_radius_is_one(self, tiny_distance):
        D_norm = normalize_distance(tiny_distance)
        new_eigmax = eigsh(D_norm, k=1, which="LM", return_eigenvectors=False)[0]
        np.testing.assert_allclose(new_eigmax, 1.0, atol=1e-8)

    def test_preserves_shape(self, tiny_distance):
        D_norm = normalize_distance(tiny_distance)
        assert D_norm.shape == tiny_distance.shape

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            normalize_distance(np.ones((3, 4)))


class TestFuseSum:
    def test_equal_weights_default(self, tiny_distance):
        D2 = tiny_distance * 2.0
        D_fused = fuse_distances([tiny_distance, D2], mode="sum", normalize=False)
        expected = 0.5 * tiny_distance + 0.5 * D2
        np.testing.assert_allclose(D_fused, expected)

    def test_custom_weights(self, tiny_distance):
        D2 = tiny_distance * 2.0
        D_fused = fuse_distances(
            [tiny_distance, D2], weights=[0.9, 0.1], mode="sum", normalize=False
        )
        expected = 0.9 * tiny_distance + 0.1 * D2
        np.testing.assert_allclose(D_fused, expected)

    def test_with_normalize(self, tiny_distance):
        D2 = tiny_distance * 5.0  # different scale
        D_fused = fuse_distances(
            [tiny_distance, D2], weights=[0.5, 0.5], mode="sum", normalize=True
        )
        # After normalization both have spectral radius 1, so 0.5+0.5 fused
        # should also have spectral radius 1.
        eigmax = eigsh(D_fused, k=1, which="LM", return_eigenvectors=False)[0]
        np.testing.assert_allclose(eigmax, 1.0, atol=1e-8)


class TestFuseMax:
    def test_max_basic(self):
        D1 = np.array([[0.0, 0.1, 0.2], [0.1, 0.0, 0.3], [0.2, 0.3, 0.0]])
        D2 = np.array([[0.0, 0.5, 0.05], [0.5, 0.0, 0.1], [0.05, 0.1, 0.0]])
        D_fused = fuse_distances([D1, D2], mode="max", normalize=False)
        # Each cell is max of the two inputs
        expected = np.maximum(D1, D2)
        np.testing.assert_allclose(D_fused, expected)


class TestFuseValidation:
    def test_invalid_mode_raises(self, tiny_distance):
        with pytest.raises(ValueError, match="sum.*max"):
            fuse_distances([tiny_distance, tiny_distance], mode="average")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            fuse_distances([], mode="sum")

    def test_shape_mismatch_raises(self, tiny_distance):
        D_other = np.eye(4)
        with pytest.raises(ValueError, match="shape"):
            fuse_distances([tiny_distance, D_other], mode="sum")

    def test_weights_length_mismatch_raises(self, tiny_distance):
        with pytest.raises(ValueError, match="length"):
            fuse_distances(
                [tiny_distance, tiny_distance], weights=[0.5], mode="sum"
            )
