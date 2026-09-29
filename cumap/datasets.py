"""Built-in example datasets bundled with cumap.

Provides one small, real single-cell dataset so the examples and quick checks
run the paper pipeline without downloading anything.
"""
import numpy as np
import scipy.sparse as sp
from importlib.resources import files

__all__ = ["load_pancreas"]


def load_pancreas(return_X_y=True):
    """Load the bundled mouse pancreatic islet demo dataset.

    Sample M1 from Baron et al. (2016), *Cell Systems* (GEO accession GSE84133),
    preprocessed with the standard SCANPY clustering pipeline (quality-control
    filtering, Scrublet doublet removal, total-count normalisation, ``log1p``,
    and selection of the 2,000 most highly variable genes). The result is
    820 cells x 2,000 genes of log-normalised counts with 13 ground-truth
    cell-type labels. The data ships with the package (~0.3 MB, stored sparse).

    Parameters
    ----------
    return_X_y : bool, default=True
        If True, return ``(X, y, target_names)``. If False, return a dict with
        keys ``"X"``, ``"y"`` and ``"target_names"``.

    Returns
    -------
    X : numpy.ndarray of shape (820, 2000), dtype float32
        Log-normalised expression matrix (cells x genes), ready to pass to
        :class:`cumap.CTSNE`.
    y : numpy.ndarray of shape (820,), dtype int64
        Ground-truth cell-type label id for each cell.
    target_names : list of str
        Cell-type name for each label id (``target_names[y[i]]`` names cell i).

    Examples
    --------
    >>> from cumap import CTSNE, load_pancreas
    >>> X, y, names = load_pancreas()
    >>> model = CTSNE(distances=["yule", "l_chebyshev"], weights="auto")
    >>> labels = model.fit_predict(X, y=y)
    """
    path = files("cumap") / "data" / "pancreas_m1.npz"
    with path.open("rb") as fh:
        d = np.load(fh, allow_pickle=True)
        X = sp.csr_matrix(
            (d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"])
        ).toarray().astype(np.float32)
        y = d["y"].astype(np.int64)
        target_names = [str(n) for n in d["target_names"]]
    if return_X_y:
        return X, y, target_names
    return {"X": X, "y": y, "target_names": target_names}
