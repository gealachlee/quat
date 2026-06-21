# =========================================================================
#  LLM-generated code.  See README.md for full disclosure.
# =========================================================================

"""Numpy-vectorized optimizations for quaternion operations.

All functions produce identical results to the standard implementations
but use numpy einsum and broadcasting for efficiency.
"""
from __future__ import annotations
import numpy as np
import numpy.typing as npt


def hamilton_einsum(p: npt.NDArray, q: npt.NDArray) -> npt.NDArray:
    """Hamilton product using a single einsum call.

    Equivalent to ``_hamilton(p, q)`` but uses the Hamilton product tensor
    ``H[r,c,k]`` such that ``out[r] = Σ_{c,k} H[r,c,k] * p[c] * q[k]``.

    Supports arbitrary leading-dimension broadcasting.

    Example:
        >>> from quat.algebra import _hamilton
        >>> from quat.optimized import hamilton_einsum
        >>> import numpy as np
        >>> p = np.random.randn(10, 4)
        >>> q = np.random.randn(10, 4)
        >>> np.allclose(_hamilton(p, q), hamilton_einsum(p, q))
        True
    """
    H = np.zeros((4, 4, 4))
    H[0, 0, 0] = 1;    H[0, 1, 1] = -1;   H[0, 2, 2] = -1;   H[0, 3, 3] = -1
    H[1, 0, 1] = 1;    H[1, 1, 0] = 1;    H[1, 2, 3] = 1;    H[1, 3, 2] = -1
    H[2, 0, 2] = 1;    H[2, 1, 3] = -1;   H[2, 2, 0] = 1;    H[2, 3, 1] = 1
    H[3, 0, 3] = 1;    H[3, 1, 2] = 1;    H[3, 2, 1] = -1;   H[3, 3, 0] = 1
    return np.einsum('rck,...c,...k->...r', H, p, q, optimize=True)


_HAMILTON_TENSOR = np.zeros((4, 4, 4))
_HAMILTON_TENSOR[0, 0, 0] = 1;    _HAMILTON_TENSOR[0, 1, 1] = -1;   _HAMILTON_TENSOR[0, 2, 2] = -1;   _HAMILTON_TENSOR[0, 3, 3] = -1
_HAMILTON_TENSOR[1, 0, 1] = 1;    _HAMILTON_TENSOR[1, 1, 0] = 1;    _HAMILTON_TENSOR[1, 2, 3] = 1;    _HAMILTON_TENSOR[1, 3, 2] = -1
_HAMILTON_TENSOR[2, 0, 2] = 1;    _HAMILTON_TENSOR[2, 1, 3] = -1;   _HAMILTON_TENSOR[2, 2, 0] = 1;    _HAMILTON_TENSOR[2, 3, 1] = 1
_HAMILTON_TENSOR[3, 0, 3] = 1;    _HAMILTON_TENSOR[3, 1, 2] = 1;    _HAMILTON_TENSOR[3, 2, 1] = -1;   _HAMILTON_TENSOR[3, 3, 0] = 1


def quat_matmul(A_data: npt.NDArray, B_data: npt.NDArray) -> npt.NDArray:
    """Vectorized quaternion matrix multiplication via einsum.

    Args:
        A_data: ndarray of shape ``(m, k, 4)``.
        B_data: ndarray of shape ``(k, n, 4)``.

    Returns:
        ndarray of shape ``(m, n, 4)``.

    Example:
        >>> from quat.optimized import quat_matmul
        >>> from quat.collections import QuatMatrix
        >>> import numpy as np
        >>> A = QuatMatrix(np.random.randn(3, 4, 4))
        >>> B = QuatMatrix(np.random.randn(4, 5, 4))
        >>> np.allclose((A * B).to_array(), quat_matmul(A._data, B._data))
        True
    """
    return np.einsum('rab,mia,inb->mnr', _HAMILTON_TENSOR, A_data, B_data, optimize=True)


def conjugate_batch(data: npt.NDArray) -> npt.NDArray:
    """Element-wise quaternion conjugate for ``(..., 4)`` data.

    Example:
        >>> from quat.optimized import conjugate_batch
        >>> import numpy as np
        >>> data = np.array([[1., 2., 3., 4.]])
        >>> conjugate_batch(data)
        array([[ 1., -2., -3., -4.]])
    """
    mask = np.array([1., -1., -1., -1.])
    return data * mask


def norm_squared_batch(data: npt.NDArray) -> npt.NDArray:
    """Per-element quaternion norm squared for ``(..., 4)`` data.

    Example:
        >>> from quat.optimized import norm_squared_batch
        >>> data = np.array([[3., 4., 0., 0.]])
        >>> norm_squared_batch(data)
        array([25.])
    """
    return (data * data).sum(axis=-1)


def normalize_batch(data: npt.NDArray) -> npt.NDArray:
    """Batch normalize quaternion vectors (in-place safe).  ``data: (..., 4)``.

    Example:
        >>> from quat.optimized import normalize_batch
        >>> import numpy as np
        >>> data = np.array([[3., 4., 0., 0.]])
        >>> result = normalize_batch(data)
        >>> np.sqrt((result * result).sum(axis=-1))
        array([1.])
    """
    norms = np.sqrt((data * data).sum(axis=-1, keepdims=True))
    norms = np.where(norms == 0, 1., norms)
    return data / norms
