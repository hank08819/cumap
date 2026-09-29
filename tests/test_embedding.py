"""Tests for cumap.embedding."""

import numpy as np
import pytest

from cumap.distances import yule_distance
from cumap.embedding import embed, embed_tsne, embed_umap


class TestEmbedTsne:
    def test_shape(self, small_data):
        X, _ = small_data
        D = yule_distance(X)
        emb = embed_tsne(D, perplexity=10)
        assert emb.shape == (X.shape[0], 2)

    def test_3d_output(self, small_data):
        X, _ = small_data
        D = yule_distance(X)
        emb = embed_tsne(D, n_components=3, perplexity=10)
        assert emb.shape == (X.shape[0], 3)

    def test_deterministic_with_seed(self, small_data):
        X, _ = small_data
        D = yule_distance(X)
        emb1 = embed_tsne(D, perplexity=10, random_state=42)
        emb2 = embed_tsne(D, perplexity=10, random_state=42)
        np.testing.assert_allclose(emb1, emb2)


class TestEmbedUmap:
    def test_shape(self, small_data):
        X, _ = small_data
        D = yule_distance(X)
        emb = embed_umap(D, n_neighbors=5)
        assert emb.shape == (X.shape[0], 2)

    def test_return_model(self, small_data):
        X, _ = small_data
        D = yule_distance(X)
        emb, model = embed_umap(D, n_neighbors=5, return_model=True)
        assert emb.shape == (X.shape[0], 2)
        # The fitted UMAP model exposes graph_ for downstream Leiden
        assert hasattr(model, "graph_")


class TestEmbedDispatch:
    def test_tsne_backend(self, small_data):
        X, _ = small_data
        D = yule_distance(X)
        emb = embed(D, backend="tsne", perplexity=10)
        assert emb.shape == (X.shape[0], 2)

    def test_umap_backend(self, small_data):
        X, _ = small_data
        D = yule_distance(X)
        emb = embed(D, backend="umap", n_neighbors=5)
        assert emb.shape == (X.shape[0], 2)

    def test_unknown_backend_raises(self, tiny_distance):
        with pytest.raises(ValueError, match="Unknown backend"):
            embed(tiny_distance, backend="lda")

    def test_callable_backend(self, small_data):
        """embed should accept a user-supplied callable as backend."""
        X, _ = small_data
        D = yule_distance(X)

        def my_backend(D, random_state=42, n_components=2):
            # Simple deterministic 'embedding': first two columns of D
            return D[:, :n_components].copy()

        emb = embed(D, backend=my_backend, n_components=2)
        assert emb.shape == (X.shape[0], 2)
