"""NumPy interop helpers for quaternion collection types."""
from __future__ import annotations
import numpy as np


def _data_copy(data: np.ndarray) -> np.ndarray:
    return data.copy()


def _to_numpy(data: np.ndarray, copy: bool = True,
              dtype: np.dtype | None = None) -> np.ndarray:
    if copy is False and dtype is None:
        return data
    return np.array(data, dtype=dtype, copy=copy)


def _to_array(data: np.ndarray) -> np.ndarray:
    return data.copy()


def _dispatch_collection_ufunc(self, ufunc, method, *inputs, **kwargs):
    """Shared __array_ufunc__ dispatch for QuatVector/QuatMatrix/QuatTensor."""
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
