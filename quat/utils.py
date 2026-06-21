"""Data conversion helpers for quaternion types."""
import numpy as np
from quat.core import Quaternion
from quat.collections import QuatVector, QuatMatrix, QuatTensor


def to_ndarray(q):
    if isinstance(q, Quaternion):
        return q._data.copy()
    if isinstance(q, (QuatVector, QuatMatrix, QuatTensor)):
        return q._data.copy()
    raise TypeError(f"Cannot convert {type(q)} to ndarray")


def from_ndarray(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        if arr.size == 4:
            return Quaternion(arr)
        if arr.size == 1:
            return Quaternion(arr[0])
        raise ValueError(f"Cannot infer type from 1-d array of size {arr.size}")
    if arr.ndim == 2:
        return QuatVector(arr)
    if arr.ndim == 3:
        return QuatMatrix(arr)
    if arr.ndim == 4:
        return QuatTensor(arr)
    raise ValueError(f"Cannot infer quaternion type from {arr.ndim}-d array")


def from_components(r, i, j, k):
    r = np.asarray(r, dtype=float)
    i = np.asarray(i, dtype=float)
    j = np.asarray(j, dtype=float)
    k = np.asarray(k, dtype=float)
    data = np.stack([r, i, j, k], axis=-1)
    return from_ndarray(data)


def broadcast_quat_shapes(*shapes):
    return np.broadcast_shapes(*shapes)


def stack_quat(quaternions, axis=0):
    if len(quaternions) == 0:
        raise ValueError("Cannot stack empty sequence")
    first = quaternions[0]
    if isinstance(first, Quaternion):
        data = np.stack([q._data for q in quaternions], axis=axis)
        return from_ndarray(data)
    if isinstance(first, QuatVector):
        data = np.stack([q._data for q in quaternions], axis=axis)
        return from_ndarray(data)
    raise TypeError(f"Cannot stack type {type(first)}")
