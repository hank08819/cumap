"""cumap GPU backend (torch).

Optional dependency: pip install cumap[gpu] (installs torch + CUDA).

Public API:
    from cumap.gpu import yule_distance, fractional_distance, l_chebyshev_distance

All functions take np.ndarray X and return np.ndarray distance matrix (n, n),
numerically equivalent to cumap CPU versions within float32 precision.
"""
from .distances import (
    yule_distance,
    fractional_distance,
    l_chebyshev_distance,
    low_rank_approx,
)

__all__ = [
    "yule_distance",
    "fractional_distance",
    "l_chebyshev_distance",
    "low_rank_approx",
]
