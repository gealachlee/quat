"""Numpy-vectorized optimizations for quaternion operations.

All functions produce identical results to the standard implementations
but use numpy einsum and broadcasting for efficiency.
"""
import numpy as np


def hamilton_einsum(p, q):
    """Hamilton product using einsum for arbitrary leading dimensions.

    Equivalent to _hamilton(p, q).
    """
    H = np.zeros((4, 4, 4))
    H[0, 0, 0] = 1;    H[0, 1, 1] = -1;   H[0, 2, 2] = -1;   H[0, 3, 3] = -1
    H[1, 0, 1] = 1;    H[1, 1, 0] = 1;    H[1, 2, 3] = 1;    H[1, 3, 2] = -1
    H[2, 0, 2] = 1;    H[2, 1, 3] = -1;   H[2, 2, 0] = 1;    H[2, 3, 1] = 1
    H[3, 0, 3] = 1;    H[3, 1, 2] = 1;    H[3, 2, 1] = -1;   H[3, 3, 0] = 1
    return np.einsum('rck,...c,...k->...r', H, p, q, optimize=True)


_H_TENSOR = np.zeros((4, 4, 4))
_H_TENSOR[0, 0, 0] = 1;    _H_TENSOR[0, 1, 1] = -1;   _H_TENSOR[0, 2, 2] = -1;   _H_TENSOR[0, 3, 3] = -1
_H_TENSOR[1, 0, 1] = 1;    _H_TENSOR[1, 1, 0] = 1;    _H_TENSOR[1, 2, 3] = 1;    _H_TENSOR[1, 3, 2] = -1
_H_TENSOR[2, 0, 2] = 1;    _H_TENSOR[2, 1, 3] = -1;   _H_TENSOR[2, 2, 0] = 1;    _H_TENSOR[2, 3, 1] = 1
_H_TENSOR[3, 0, 3] = 1;    _H_TENSOR[3, 1, 2] = 1;    _H_TENSOR[3, 2, 1] = -1;   _H_TENSOR[3, 3, 0] = 1


def quat_matmul(A_data, B_data):
    """Vectorized quaternion matrix multiplication.

    Args:
        A_data: (m, k, 4) ndarray
        B_data: (k, n, 4) ndarray

    Returns:
        (m, n, 4) ndarray
    """
    return np.einsum('rab,mia,inb->mnr', _H_TENSOR, A_data, B_data, optimize=True)


def conjugate_batch(data):
    """Element-wise quaternion conjugate for (..., 4) data."""
    mask = np.array([1., -1., -1., -1.])
    return data * mask


def norm_squared_batch(data):
    """Element-wise quaternion norm squared for (..., 4) data."""
    return (data * data).sum(axis=-1)


def normalize_batch(data):
    """Batch normalize quaternion vectors. data: (..., 4)."""
    norms = np.sqrt((data * data).sum(axis=-1, keepdims=True))
    norms = np.where(norms == 0, 1., norms)
    return data / norms
