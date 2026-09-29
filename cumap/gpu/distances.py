"""GPU implementation of cumap three distances using PyTorch.

数学上跟 cumap.distances 的 CPU 版本完全等价 (yule = pdist 'yule',
fractional = pdist minkowski p=0.25, l_chebyshev = U @ diag(s[:k]) @ V → pdist 'chebyshev').
Float32 精度上 max rel diff <= 1e-3 (实测 H1 1925 × 2000).

输入 X 接受 np.ndarray 或 torch.Tensor; 返回 np.ndarray (n, n) float32, 跟 CPU 版兼容.

大数据 (Cortex Frontal 71k cells) 时 (n, n) float32 距离矩阵 ~20GB, 分块算 + CPU 累积
避免 GPU OOM. GPU 内存峰值大概 5-10GB (chunk=4096 default).
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "cumap.gpu requires PyTorch. Install with: pip install cumap[gpu]"
    ) from e


def _to_tensor(X, device, dtype):
    # 把 np.ndarray 或 torch.Tensor 都转成指定 device + dtype 的 torch.Tensor
    if isinstance(X, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(X)).to(device=device, dtype=dtype)
    if isinstance(X, torch.Tensor):
        return X.to(device=device, dtype=dtype)
    raise TypeError(f"X must be np.ndarray or torch.Tensor, got {type(X)}")


def yule_distance(
    X,
    device: str = "cuda",
    chunk: int = 4096,
    dtype=torch.float32,
) -> np.ndarray:
    """Yule binary dissimilarity on rows of X.

    数学: B = (X != 0); 对每对 (i, j) 算 4 个 binary counts
    cTT/cTF/cFT/cFF, Yule = 2*cTF*cFT / (cTT*cFF + cTF*cFT).

    Parameters
    ----------
    X : (n, d) np.ndarray or torch.Tensor
    device : "cuda" | "cuda:0" | "cpu"
    chunk : int, 行分块大小; 大数据避免一次性 (n, n) 中间张量 OOM
    dtype : torch.float32 (推荐) or torch.float64

    Returns
    -------
    D : (n, n) np.ndarray float32, 对称, 对角 = 0
    """
    X_t = _to_tensor(X, device, dtype)
    B = (X_t != 0).to(dtype)
    not_B = 1.0 - B
    n = B.shape[0]
    # D 放 CPU 内存避免 GPU 装不下 (Cortex F 71k × 71k float32 = 19GB)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(0, n, chunk):
        Bi = B[i : i + chunk]
        not_Bi = not_B[i : i + chunk]
        # 4 个 matmul, 每个 shape (chunk, n)
        cTT = Bi @ B.T
        cTF = Bi @ not_B.T
        cFT = not_Bi @ B.T
        cFF = not_Bi @ not_B.T
        num = 2.0 * cTF * cFT
        denom = cTT * cFF + cTF * cFT
        # +1e-12 防 0/0 (与 scipy yule 行为一致, 全相同 binary 时返回 0)
        D_chunk = (num / (denom + 1e-12)).cpu().numpy().astype(np.float32)
        D[i : i + chunk] = D_chunk
    np.fill_diagonal(D, 0.0)
    return D


def fractional_distance(
    X,
    f: float = 0.25,
    device: str = "cuda",
    chunk: int = 4096,
    dtype=torch.float32,
) -> np.ndarray:
    """Fractional Minkowski distance: D[i,j] = (sum_k |x_ik - x_jk|^f)^(1/f).

    跟 cumap.distances.fractional_distance / scipy pdist(metric='minkowski', p=f) 等价.
    torch.cdist 文档: |x - y|_p = (sum |x_i - y_i|^p)^(1/p), 直接调用即可.
    """
    X_t = _to_tensor(X, device, dtype)
    n = X_t.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(0, n, chunk):
        # torch.cdist 在 GPU 上支持 p<1 (实测 p=0.25 H1 ✓)
        D_chunk = torch.cdist(X_t[i : i + chunk], X_t, p=f).cpu().numpy().astype(np.float32)
        D[i : i + chunk] = D_chunk
    return D


def low_rank_approx(X, k: int, device: str = "cuda", dtype=torch.float32):
    """Rank-k SVD approximation: U[:, :k] @ diag(s[:k]) @ Vh[:k, :].

    返回 torch.Tensor on device (跟 CPU 版返回 np.ndarray 不同, 因为后面 cdist 还在 GPU 上算).
    跟 cumap.distances.low_rank_approx 数学等价: U @ diag(S[:k]) @ V.
    """
    X_t = _to_tensor(X, device, dtype)
    m, n = X_t.shape
    L = min(m, n)
    if k > L:
        raise ValueError(f"Requested rank k={k} exceeds min(m, n)={L}")
    if k == 0:
        return X_t
    try:
        U, s, Vh = torch.linalg.svd(X_t, full_matrices=False)
        # 重建到 (m, n) 原空间, 跟 cumap CPU 一致 (CPU 用 U @ diag(S[:k]) @ V)
        return (U[:, :k] * s[:k]) @ Vh[:k, :]
    except torch._C._LinAlgError:
        # 大矩阵 (如 24k×35k) cusolver full SVD 崩 → randomized truncated SVD (rank-k 近似, 目的一致).
        # 只在 full SVD 失败时触发, 小数据仍走上面的 exact 路径, 结果不变.
        q = min(k + 10, L)
        U, s, V = torch.svd_lowrank(X_t, q=q, niter=4)
        return (U[:, :k] * s[:k]) @ V[:, :k].T


def l_chebyshev_distance(
    X,
    k: int = 15,
    device: str = "cuda",
    chunk: int = 4096,
    dtype=torch.float32,
) -> np.ndarray:
    """SVD rank-k 近似 + Chebyshev (L-inf) distance.

    Step 1: X_low = U @ diag(s[:k]) @ Vh, shape (m, n) 重建到原空间.
    Step 2: D[i,j] = max_k |X_low[i,k] - X_low[j,k]|.

    注意 cumap CPU low_rank_approx 返回 (m, n) 重建而非 (m, k) 主成分, GPU 同步.
    """
    X_t = _to_tensor(X, device, dtype)
    m, n = X_t.shape
    L = min(m, n)
    if k > L:
        import warnings

        warnings.warn(
            f"l_chebyshev k={k} exceeds min(X.shape)={L}; capping to {L}.",
            UserWarning,
            stacklevel=2,
        )
        k = L
    X_low = low_rank_approx(X_t, k=k, device=device, dtype=dtype)  # (m, n) on GPU
    n_cells = X_low.shape[0]
    D = np.zeros((n_cells, n_cells), dtype=np.float32)
    for i in range(0, n_cells, chunk):
        # cdist p=inf = chebyshev = max abs diff
        D_chunk = (
            torch.cdist(X_low[i : i + chunk], X_low, p=float("inf"))
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        D[i : i + chunk] = D_chunk
    return D
