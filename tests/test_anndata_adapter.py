"""Tests for the AnnData / scanpy adapter (#6 from roadmap).

Adapter goals (recap):
- ndarray and scipy-sparse inputs work as before (back-compat).
- AnnData inputs work via duck typing — no hard anndata dependency.
- ``layer`` argument routes to ``adata.X`` / ``adata.raw.X`` / ``adata.layers[k]``.
- Clear errors on missing layers / missing ``.raw`` / ``layer`` given without AnnData.

The AnnData integration tests use ``pytest.importorskip`` so the suite
still passes in environments without ``anndata`` installed.
"""

import numpy as np
import pytest
from scipy import sparse

from cumap import CTSNE
from cumap._adapter import _extract_matrix, _looks_like_anndata


# ---------------------------------------------------------------------------
# Unit tests for _extract_matrix (no anndata dependency)
# ---------------------------------------------------------------------------


class TestExtractMatrixNdarray:
    def test_ndarray_passthrough(self):
        X = np.random.default_rng(0).poisson(1.0, size=(20, 30)).astype(np.float64)
        out = _extract_matrix(X)
        assert isinstance(out, np.ndarray)
        assert out.shape == (20, 30)
        np.testing.assert_array_equal(out, X)

    def test_sparse_csr_densified(self):
        X_dense = np.random.default_rng(0).poisson(0.5, size=(10, 15)).astype(np.float64)
        X_sparse = sparse.csr_matrix(X_dense)
        out = _extract_matrix(X_sparse)
        assert isinstance(out, np.ndarray)
        assert out.shape == (10, 15)
        np.testing.assert_array_equal(out, X_dense)

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            _extract_matrix(np.zeros(10))

    def test_layer_given_on_ndarray_raises(self):
        with pytest.raises(ValueError, match="layer"):
            _extract_matrix(np.zeros((5, 5)), layer="counts")


class TestLooksLikeAnnData:
    def test_ndarray_is_not_anndata(self):
        assert not _looks_like_anndata(np.zeros((5, 5)))

    def test_sparse_is_not_anndata(self):
        assert not _looks_like_anndata(sparse.csr_matrix((5, 5)))

    def test_duck_typed_object_recognized(self):
        class FakeAnnData:
            X = np.zeros((3, 3))
            obs = "not-empty"
            var = "not-empty"

        assert _looks_like_anndata(FakeAnnData())


# ---------------------------------------------------------------------------
# Integration tests requiring anndata
# ---------------------------------------------------------------------------


@pytest.fixture
def adata_small(small_data):
    """Wrap small_data into an AnnData object (skips if anndata missing)."""
    anndata = pytest.importorskip("anndata")
    X, y = small_data
    ad = anndata.AnnData(X=X.copy())
    ad.obs["label"] = y.astype(str)
    return ad, y


class TestAnnDataInputs:
    def test_extract_from_dense_anndata(self, adata_small):
        ad, _ = adata_small
        out = _extract_matrix(ad)
        assert out.shape == ad.X.shape
        np.testing.assert_array_equal(out, ad.X)

    def test_extract_from_sparse_anndata(self, small_data):
        anndata = pytest.importorskip("anndata")
        X, _ = small_data
        ad = anndata.AnnData(X=sparse.csr_matrix(X))
        out = _extract_matrix(ad)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, X)

    def test_extract_layer(self, small_data):
        anndata = pytest.importorskip("anndata")
        X, _ = small_data
        ad = anndata.AnnData(X=np.zeros_like(X))
        ad.layers["counts"] = X.copy()
        out = _extract_matrix(ad, layer="counts")
        np.testing.assert_array_equal(out, X)
        # And the default layer=None still reads adata.X (the zeros).
        out_default = _extract_matrix(ad)
        np.testing.assert_array_equal(out_default, np.zeros_like(X))

    def test_extract_raw(self, small_data):
        anndata = pytest.importorskip("anndata")
        X, _ = small_data
        ad = anndata.AnnData(X=X.copy())
        ad.raw = ad  # stash raw before pretend-normalization
        ad.X = ad.X * 100.0  # simulate scanpy normalize
        out = _extract_matrix(ad, layer="raw")
        np.testing.assert_array_equal(out, X)

    def test_missing_layer_errors_with_available_list(self, small_data):
        anndata = pytest.importorskip("anndata")
        X, _ = small_data
        ad = anndata.AnnData(X=X.copy())
        ad.layers["counts"] = X.copy()
        with pytest.raises(ValueError, match="available layers"):
            _extract_matrix(ad, layer="nonexistent")

    def test_raw_missing_errors(self, small_data):
        anndata = pytest.importorskip("anndata")
        X, _ = small_data
        ad = anndata.AnnData(X=X.copy())
        # No ad.raw set.
        with pytest.raises(ValueError, match="adata.raw is None"):
            _extract_matrix(ad, layer="raw")


# ---------------------------------------------------------------------------
# End-to-end: CTSNE.fit_transform / fit_predict accept AnnData
# ---------------------------------------------------------------------------


class TestCTSNEWithAnnData:
    """CTSNE end-to-end with AnnData input. Compares to ndarray baseline."""

    def _model(self):
        return CTSNE(
            distances=["yule", "l_chebyshev"],
            weights=[0.9, 0.1],
            backend="umap",
            n_neighbors=5,
            l_chebyshev_k=5,
            random_state=42,
        )

    def test_fit_transform_dense_adata(self, adata_small):
        ad, _ = adata_small
        emb = self._model().fit_transform(ad)
        assert emb.shape == (ad.n_obs, 2)

    def test_fit_transform_matches_ndarray(self, small_data):
        anndata = pytest.importorskip("anndata")
        X, _ = small_data
        ad = anndata.AnnData(X=X.copy())

        emb_arr = self._model().fit_transform(X)
        emb_ad = self._model().fit_transform(ad)
        np.testing.assert_allclose(emb_arr, emb_ad, atol=1e-8)

    def test_fit_predict_sparse_adata(self, small_data):
        anndata = pytest.importorskip("anndata")
        X, y = small_data
        ad = anndata.AnnData(X=sparse.csr_matrix(X))
        model = self._model()
        labels = model.fit_predict(ad, y=y)
        assert labels.shape == (X.shape[0],)
        # Synthetic data is well-separated; should still cluster well.
        assert model.score(y) > 0.5

    def test_layer_kwarg_routed_through(self, small_data):
        anndata = pytest.importorskip("anndata")
        X, _ = small_data
        # Two different non-zero matrices: noise in adata.X, real signal in
        # layers['counts']. Both paths must produce valid embeddings, and
        # the two embeddings must differ (otherwise the layer argument is
        # being ignored).
        rng = np.random.default_rng(1)
        X_noise = rng.poisson(1.0, size=X.shape).astype(np.float64)
        ad = anndata.AnnData(X=X_noise)
        ad.layers["counts"] = X.copy()

        m_default = CTSNE(
            distances=["yule", "l_chebyshev"],
            weights=[0.9, 0.1],
            backend="umap",
            n_neighbors=5,
            l_chebyshev_k=5,
        )
        m_counts = CTSNE(
            distances=["yule", "l_chebyshev"],
            weights=[0.9, 0.1],
            backend="umap",
            n_neighbors=5,
            l_chebyshev_k=5,
            layer="counts",
        )
        emb_default = m_default.fit_transform(ad)
        emb_counts = m_counts.fit_transform(ad)
        assert emb_default.shape == emb_counts.shape == (ad.n_obs, 2)
        # Different source matrices ⇒ different embeddings.
        assert not np.allclose(emb_default, emb_counts)
