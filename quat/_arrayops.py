"""NumPy interop helpers for quaternion collection types.

Provides shared implementations of ``data``, ``to_array``, ``to_numpy``,
``__array__``, and ``__array_ufunc__`` so QuatVector, QuatMatrix, and
QuatTensor share a single code path for these operations.
"""
from __future__ import annotations
import numpy as np


def _data_copy(data: np.ndarray) -> np.ndarray:
    """Return a defensive copy of the underlying quaternion data."""
    return data.copy()


def _to_numpy(data: np.ndarray, copy: bool = True,
              dtype: np.dtype | None = None) -> np.ndarray:
    """Export quaternion data to an ndarray, optionally with dtype conversion.

    When ``copy=False`` and ``dtype=None``, returns the internal array
    without copying — caller must not mutate it.
    """
    if copy is False and dtype is None:
        return data
    return np.array(data, dtype=dtype, copy=copy)


def _to_array(data: np.ndarray) -> np.ndarray:
    """Return a copy of the quaternion data as an ndarray."""
    return data.copy()


def _dispatch_collection_ufunc(
    self, ufunc, method: str, *inputs, **kwargs
) -> object:
    """Shared ``__array_ufunc__`` dispatch for QuatVector/QuatMatrix/QuatTensor.

    The *self* parameter is unused in the body (dispatch works via *inputs*)
    but is kept for signature compatibility with NumPy's protocol.

    Returns ``NotImplemented`` for unsupported ufuncs or reduction methods.
    """
    if method != '__call__' or kwargs.get('out') is not None:
        return NotImplemented
    a, b = (inputs[0], inputs[1]) if len(inputs) == 2 else (inputs[0], None)
    if ufunc is np.add:
        return a + b
    if ufunc is np.subtract:
        return a - b
    if ufunc is np.multiply:
        return a * b
    if ufunc is np.true_divide or ufunc is np.floor_divide:
        return a / b
    if ufunc is np.negative:
        return -a
    return NotImplemented
