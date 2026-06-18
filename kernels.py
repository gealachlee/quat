#!/usr/bin/env python3
"""Quaternion kernel matrix computations.

Implements kernel functions from:
  Li, Lin & Liu, "Kernel Support Quaternion Matrix Machines with Ramp Loss"
  References: Felipe2014 (QRKHS), Chen2025QSMM (LSQMM)

Kernels (Section 2.2, Felipe2014 Section V):
  (i)  Q-cubic:    k(A,B) = (1+|A|_F^2)(1+<A,B>)(1+|B|_F^2)
  (ii) Q-Gaussian: k(A,B) = exp(-gamma * tr((A-conj(B))^T (A-conj(B))))

where <A,B> = sum_{i,j} A_{ij} * conj(B_{ij})  (Frobenius inner product, Eq 3.1)
"""

import numpy as np
from quaternion import QuatTensor, QuatMatrix, _hamilton, _CONJ

# ---------------------------------------------------------------------------
# Quaternion exponential (vectorized)
# ---------------------------------------------------------------------------
def _qexp(Q):
    """Vectorized quaternion exponential.  Q: (..., 4) array."""
    a = Q[..., 0]
    v = Q[..., 1:4]  # (..., 3)
    v_norm = np.sqrt((v * v).sum(axis=-1))
    ea = np.exp(a)
    out = np.empty_like(Q)
    out[..., 0] = ea * np.cos(v_norm)
    small = v_norm < 1e-15
    s = np.empty_like(v_norm)
    s[~small] = np.sin(v_norm[~small]) / v_norm[~small]
    s[small] = 0.
    s = s[..., None]  # (..., 1)
    out[..., 1:4] = ea[..., None] * s * v
    return out


# ---------------------------------------------------------------------------
# Kernel matrix: Q-cubic
# ---------------------------------------------------------------------------
def cubic_kernel_matrix(X, chunk_size=None):
    r"""Compute the Q-cubic kernel matrix.

    Formula (paper Eq. on line 358):
        K_{ij} = (1 + |X_i|_F^2) * (1 + <X_i, X_j>) * (1 + |X_j|_F^2)

    where <·,·> is the quaternion Frobenius inner product (Eq 3.1):
        <A, B> = \sum_{p,q} A_{pq} conj(B_{pq})  \in \mathbb{H}

    Args:
        X: QuatTensor (N, m, l)  -- N samples, each m×l quaternion
        chunk_size: ignored (cubic kernel is fully vectorized and memory-efficient)

    Returns:
        K: (N, N, 4)  quaternion kernel matrix
    """
    # Flatten spatial dims: (N, M, 4)  where M = m * l
    if isinstance(X, QuatTensor):
        data = X._data.reshape(X.shape[0], -1, 4)
    elif isinstance(X, QuatMatrix):
        data = X._data.reshape(X._m, -1, 4)
    else:
        data = np.asarray(X, dtype=float)
        data = data.reshape(data.shape[0], -1, 4)

    N, M, _ = data.shape

    # --- Frobenius norms squared (real scalars) ---
    # |q|_F^2 = sum_{p,q} |q_{pq}|^2 = sum_{p,q} (q1^2+q2^2+q3^2+q4^2)
    norms_sq = (data * data).sum(axis=(1, 2))  # (N,)

    # --- Frobenius inner product matrix (quaternions) ---
    # conj_data[i,p] = conj(X_i)_p
    conj_data = data * _CONJ  # (N, M, 4)

    # <X_i, X_j> = sum_p  X_i_p * conj(X_j)_p   (Hamilton product per pixel)
    # broadcast: (N, 1, M, 4) vs (1, N, M, 4) -> (N, N, M, 4), sum over M
    inner = _hamilton(
        data[:, None, :, :],       # (N, 1, M, 4)
        conj_data[None, :, :, :],  # (1, N, M, 4)
    ).sum(axis=2)  # (N, N, 4)  -- quaternion-valued

    # --- Cubic kernel ---
    # K_ij = (1 + |X_i|^2) * (1 + <X_i,X_j>) * (1 + |X_j|^2)
    one_plus_inner = inner.copy()
    one_plus_inner[..., 0] += 1.  # add 1 to real part

    K = (1. + norms_sq[:, None, None]) * one_plus_inner * (1. + norms_sq[None, :, None])
    return K  # (N, N, 4)


# ---------------------------------------------------------------------------
# Kernel matrix: Q-Gaussian
# ---------------------------------------------------------------------------
def gaussian_kernel_matrix(X, gamma=1., chunk_size=200):
    r"""Compute the Q-Gaussian kernel matrix.

    Formula (paper Eq. on line 359):
        K_{ij} = exp( -gamma * tr( (X_i - conj(X_j))^T (X_i - conj(X_j)) ) )

    where  tr(C^T C) = \sum_{p,q} C_{pq}^2   (element-wise quaternion square + sum).

    Note: uses chunked computation to avoid (N,N,M,4) intermediate.
    Use ``chunk_size=0`` for fully vectorized (high memory).

    Args:
        X: QuatTensor (N, m, l)
        gamma: kernel bandwidth parameter
        chunk_size: batch size for memory control (default 200)

    Returns:
        K: (N, N, 4)  quaternion kernel matrix
    """
    if isinstance(X, QuatTensor):
        data = X._data.reshape(X.shape[0], -1, 4)
    elif isinstance(X, QuatMatrix):
        data = X._data.reshape(X._m, -1, 4)
    else:
        data = np.asarray(X, dtype=float)
        data = data.reshape(data.shape[0], -1, 4)

    N, M, _ = data.shape
    K = np.empty((N, N, 4))

    conj_data = data * _CONJ  # (N, M, 4)

    if chunk_size and chunk_size > 0:
        chunk_size = min(chunk_size, N)
        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            for j_start in range(0, N, chunk_size):
                j_end = min(j_start + chunk_size, N)

                Di = data[i_start:i_end, None, :, :]      # (chunk_i, 1, M, 4)
                Cj = conj_data[None, j_start:j_end, :, :]  # (1, chunk_j, M, 4)

                # C = X_i - conj(X_j)  element-wise
                diff = Di - Cj  # (chunk_i, chunk_j, M, 4)

                # C^2  element-wise quaternion square
                diff_sq = _hamilton(diff, diff)  # (chunk_i, chunk_j, M, 4)

                # tr(C^T C) = sum_{p,q} C_{pq}^2
                trace_sq = diff_sq.sum(axis=2)  # (chunk_i, chunk_j, 4)

                # exp(-gamma * trace)
                K[i_start:i_end, j_start:j_end] = _qexp(-gamma * trace_sq)
    else:
        diff = data[:, None, :, :] - conj_data[None, :, :, :]  # (N,N,M,4)
        diff_sq = _hamilton(diff, diff).sum(axis=2)            # (N,N,4)
        K = _qexp(-gamma * diff_sq)

    return K


# ---------------------------------------------------------------------------
# Convenience: select kernel by name
# ---------------------------------------------------------------------------
_KERNEL_MAP = {
    'cubic':    cubic_kernel_matrix,
    'gaussian': gaussian_kernel_matrix,
}


def compute_kernel_matrix(X, kernel_type='cubic', **kwargs):
    """Compute quaternion kernel matrix.

    Args:
        X: QuatTensor (N, m, l)
        kernel_type: 'cubic' | 'gaussian'
        **kwargs: forwarded to kernel function (e.g. gamma for gaussian)
    """
    fn = _KERNEL_MAP.get(kernel_type)
    if fn is None:
        raise ValueError(f"Unknown kernel type: {kernel_type}. "
                         f"Available: {list(_KERNEL_MAP.keys())}")
    # Filter kwargs to only what the function accepts
    import inspect
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(X, **filtered)
