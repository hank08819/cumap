"""cumap quickstart: end-to-end pipeline on synthetic scRNA-seq-like data.

Run from this folder:
    cd scripts && python3 quickstart.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cumap import CTSNE


def make_synthetic_counts(n_per_cluster=40, n_clusters=3, n_genes=200, seed=42):
    """60 cells x 200 genes Poisson counts with 3 well-separated clusters."""
    rng = np.random.default_rng(seed)
    n_markers = n_genes // n_clusters
    X_parts, labels = [], []
    for k in range(n_clusters):
        means = np.full(n_genes, 0.1)
        means[k * n_markers : (k + 1) * n_markers] = 3.0
        X_parts.append(rng.poisson(means, size=(n_per_cluster, n_genes)))
        labels.extend([k] * n_per_cluster)
    return np.vstack(X_parts).astype(np.float64), np.array(labels)


def main():
    X, y = make_synthetic_counts()
    print(f"Input: {X.shape[0]} cells x {X.shape[1]} genes, {len(np.unique(y))} true clusters")

    model = CTSNE(
        distances=["yule", "l_chebyshev"],
        weights="auto",
        backend="umap",
        l_chebyshev_k=10,
        n_neighbors=10,
    )
    labels = model.fit_predict(X, y=y)

    print(f"Embedding shape : {model.embedding_.shape}")
    print(f"Best weights    : Yule={model.weights_[0]:.2f}, L-Chebyshev={model.weights_[1]:.2f}")
    print(f"Best NMI (search)         : {model.best_score_:.4f}")
    print(f"NMI(KMeans labels, y_true): {model.score(y):.4f}")


if __name__ == "__main__":
    main()
