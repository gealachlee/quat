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
# Internal helpers
# ---------------------------------------------------------------------------
def _flatten_data(X):
    if isinstance(X, QuatTensor):
        data = X._data.reshape(X.shape[0], -1, 4)
    elif isinstance(X, QuatMatrix):
        data = X._data.reshape(X._m, -1, 4)
    else:
        data = np.asarray(X, dtype=float)
        data = data.reshape(data.shape[0], -1, 4)
    return data, data.shape[0], data.shape[1]


def _cubic_inner(data):
    """Frobenius norms and inner product from flattened (N,M,4) data."""
    N = data.shape[0]
    norms_sq = (data * data).sum(axis=(1, 2))
    conj_data = data * _CONJ
    inner = _hamilton(
        data[:, None, :, :], conj_data[None, :, :, :]
    ).sum(axis=2)
    return norms_sq, inner


def _cubic_assemble(norms_row, inner, norms_col):
    """K_ij = (1+|row_i|^2)(1+inner_ij)(1+|col_j|^2)."""
    one_plus = inner.copy()
    one_plus[..., 0] += 1.
    return (1. + norms_row[:, None, None]) * one_plus * (1. + norms_col[None, :, None])


# ---------------------------------------------------------------------------
# Kernel matrix: Q-cubic (K_train)
# ---------------------------------------------------------------------------
def cubic_kernel_matrix(X):
    r"""Compute the Q-cubic kernel matrix K_train.

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
    data, N, M = _flatten_data(X)
    norms_sq, inner = _cubic_inner(data)
    return _cubic_assemble(norms_sq, inner, norms_sq)


def cubic_kernel_cross(X_test, X_train):
    r"""Compute cross kernel matrix K_test = κ(X_test, X_train).

    K_test[a, i] = (1 + |X_test_a|^2)(1 + <X_test_a, X_train_i>)(1 + |X_train_i|^2)

    Args:
        X_test:  QuatTensor (N_test, m, l)
        X_train: QuatTensor (N_train, m, l)

    Returns:
        K_test: (N_test, N_train, 4)
    """
    data_test,  Nt, M = _flatten_data(X_test)
    data_train, Ntr, _ = _flatten_data(X_train)
    assert M == _, "Spatial dimensions must match"

    norms_sq_test  = (data_test  * data_test).sum(axis=(1, 2))    # (N_test,)
    norms_sq_train = (data_train * data_train).sum(axis=(1, 2))   # (N_train,)

    conj_train = data_train * _CONJ
    inner_cross = _hamilton(
        data_test[:, None, :, :],       # (N_test, 1, M, 4)
        conj_train[None, :, :, :],      # (1, N_train, M, 4)
    ).sum(axis=2)                       # (N_test, N_train, 4)

    return _cubic_assemble(norms_sq_test, inner_cross, norms_sq_train)


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
    data, N, M = _flatten_data(X)
    K = np.empty((N, N, 4))

    conj_data = data * _CONJ

    if chunk_size and chunk_size > 0:
        chunk_size = min(chunk_size, N)
        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            for j_start in range(0, N, chunk_size):
                j_end = min(j_start + chunk_size, N)

                Di = data[i_start:i_end, None, :, :]
                Cj = conj_data[None, j_start:j_end, :, :]
                diff = Di - Cj
                diff_sq = _hamilton(diff, diff)
                trace_sq = diff_sq.sum(axis=2)
                K[i_start:i_end, j_start:j_end] = _qexp(-gamma * trace_sq)
    else:
        diff = data[:, None, :, :] - conj_data[None, :, :, :]
        diff_sq = _hamilton(diff, diff).sum(axis=2)
        K = _qexp(-gamma * diff_sq)

    return K


def gaussian_kernel_cross(X_test, X_train, gamma=1., chunk_size=200):
    r"""Compute cross Q-Gaussian kernel K_test = κ(X_test, X_train).

    K_test[a, i] = exp( -γ·tr((X_test_a - conj(X_train_i))^T(...)) )

    Args:
        X_test:  QuatTensor (N_test, m, l)
        X_train: QuatTensor (N_train, m, l)
        gamma: bandwidth
        chunk_size: batch size for memory control

    Returns:
        K_test: (N_test, N_train, 4)
    """
    data_test,  Nt, M = _flatten_data(X_test)
    data_train, Ntr, _ = _flatten_data(X_train)
    assert M == _, "Spatial dimensions must match"

    K = np.empty((Nt, Ntr, 4))
    conj_train = data_train * _CONJ

    if chunk_size and chunk_size > 0:
        chunk_size = min(chunk_size, max(Nt, Ntr))
        for i_start in range(0, Nt, chunk_size):
            i_end = min(i_start + chunk_size, Nt)
            for j_start in range(0, Ntr, chunk_size):
                j_end = min(j_start + chunk_size, Ntr)

                Di = data_test[i_start:i_end, None, :, :]
                Cj = conj_train[None, j_start:j_end, :, :]
                diff = Di - Cj
                diff_sq = _hamilton(diff, diff)
                trace_sq = diff_sq.sum(axis=2)
                K[i_start:i_end, j_start:j_end] = _qexp(-gamma * trace_sq)
    else:
        diff = data_test[:, None, :, :] - conj_train[None, :, :, :]
        K = _qexp(-gamma * _hamilton(diff, diff).sum(axis=2))

    return K


# ---------------------------------------------------------------------------
# Per-channel max-normalization
# ---------------------------------------------------------------------------
def normalize_kernel(K):
    r"""Normalize quaternion kernel matrix by each channel's max absolute value.

    K_norm[c] = K[c] / max(|K[c]|)  for c = 0,1,2,3  (real,i,j,k)

    Preserves the quaternion structure while preventing overflow.
    
    Args:
        K: (..., 4) quaternion kernel values

    Returns:
        K_norm: same shape, each channel scaled to [-1, 1]
    """
    K = np.asarray(K, dtype=float)
    max_abs = np.abs(K).max(axis=tuple(range(K.ndim - 1)), keepdims=True)  # (1,...,1,4)
    max_abs[max_abs == 0.] = 1.
    return K / max_abs


# ---------------------------------------------------------------------------
# Convenience: select kernel by name
# ---------------------------------------------------------------------------
_KERNEL_MAP = {
    'cubic':    cubic_kernel_matrix,
    'gaussian': gaussian_kernel_matrix,
}

_CROSS_KERNEL_MAP = {
    'cubic':    cubic_kernel_cross,
    'gaussian': gaussian_kernel_cross,
}


def compute_kernel_matrix(X, kernel_type='cubic', **kwargs):
    """Compute quaternion kernel matrix K_train = κ(X, X).

    Args:
        X: QuatTensor (N, m, l)
        kernel_type: 'cubic' | 'gaussian'
        **kwargs: forwarded to kernel function (e.g. gamma for gaussian)

    Returns:
        K: (N, N, 4)
    """
    return _select_kernel(_KERNEL_MAP, kernel_type, X, **kwargs)


def compute_kernel_cross(X_test, X_train, kernel_type='cubic', **kwargs):
    """Compute cross kernel matrix K_test = κ(X_test, X_train).

    Args:
        X_test:  QuatTensor (N_test, m, l)
        X_train: QuatTensor (N_train, m, l)
        kernel_type: 'cubic' | 'gaussian'
        **kwargs: forwarded to kernel function (e.g. gamma for gaussian)

    Returns:
        K_test: (N_test, N_train, 4)
    """
    return _select_kernel(_CROSS_KERNEL_MAP, kernel_type, X_test, X_train, **kwargs)


def _select_kernel(catalog, kernel_type, *args, **kwargs):
    import inspect
    fn = catalog.get(kernel_type)
    if fn is None:
        raise ValueError(f"Unknown kernel type: {kernel_type}. "
                         f"Available: {list(catalog.keys())}")
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(*args, **filtered)
