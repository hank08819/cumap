"""Shared fixtures for cumap tests.

Provides small synthetic scRNA-seq-like data so tests are fast (<10s total)
and don't depend on downloaded GSE84133 data.
"""

import numpy as np
import pytest


@pytest.fixture
def small_data():
    """Synthetic raw counts: 60 cells, 150 genes, 3 clusters.

    Each cluster has 50 marker genes with high mean expression (Poisson lambda
    = 3) and 100 background genes with low expression (lambda = 0.1). Poisson
    sampling naturally produces zeros (dropout), matching scRNA-seq sparsity.

    Returns
    -------
    X : ndarray of shape (60, 150)
        Raw count matrix.
    y : ndarray of shape (60,)
        Cluster labels in {0, 1, 2}.
    """
    rng = np.random.default_rng(42)
    n_per_cluster = 20
    n_clusters = 3
    n_genes = 150
    n_markers = 50

    cells = []
    labels = []
    for k in range(n_clusters):
        means = np.full(n_genes, 0.1)
        means[k * n_markers : (k + 1) * n_markers] = 3.0
        counts = rng.poisson(means, size=(n_per_cluster, n_genes))
        cells.append(counts)
        labels.extend([k] * n_per_cluster)

    X = np.vstack(cells).astype(np.float64)
    y = np.array(labels, dtype=np.int64)
    return X, y


@pytest.fixture
def tiny_distance():
    """A tiny 5x5 symmetric distance matrix for quick fusion/normalization tests."""
    D = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 0.0, 1.5, 2.5, 3.5],
            [2.0, 1.5, 0.0, 1.2, 2.2],
            [3.0, 2.5, 1.2, 0.0, 1.1],
            [4.0, 3.5, 2.2, 1.1, 0.0],
        ]
    )
    return D
