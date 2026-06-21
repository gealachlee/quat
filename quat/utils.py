"""Data conversion helpers and numeric validation for quaternion types."""
from __future__ import annotations
from typing import Union
import numpy as np
from quat.core import Quaternion
from quat.collections import QuatVector, QuatMatrix, QuatTensor


QuatLike = Union[Quaternion, QuatVector, QuatMatrix, QuatTensor]


def to_ndarray(q: QuatLike) -> np.ndarray:
    """Extract raw (..., 4) ndarray from any quaternion type.

    Example:
        >>> from quat import Quaternion, to_ndarray
        >>> q = Quaternion(1, 2, 3, 4)
        >>> to_ndarray(q)
        array([1., 2., 3., 4.])
    """
    if isinstance(q, Quaternion):
        return q._data.copy()
    if isinstance(q, (QuatVector, QuatMatrix, QuatTensor)):
        return q._data.copy()
    raise TypeError(f"Cannot convert {type(q)} to ndarray")


def from_ndarray(arr: np.ndarray) -> QuatLike:
    """Infer quaternion type from an ndarray by dimensionality.

    ==========  ===========
    ndim        result
    ==========  ===========
    1 (size 4)  Quaternion
    2           QuatVector
    3           QuatMatrix
    4           QuatTensor
    ==========  ===========

    Example:
        >>> from quat import from_ndarray, Quaternion, QuatVector
        >>> q = from_ndarray(np.array([1., 2., 3., 4.]))
        >>> isinstance(q, Quaternion)
        True
        >>> v = from_ndarray(np.random.randn(5, 4))
        >>> isinstance(v, QuatVector)
        True
    """
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


def from_components(r: np.ndarray, i: np.ndarray, j: np.ndarray, k: np.ndarray) -> QuatLike:
    """Construct quaternion collection from separate component arrays.

    Returns QuatVector / QuatMatrix / QuatTensor depending on component shape.

    Example:
        >>> from quat import from_components, QuatVector
        >>> import numpy as np
        >>> r, i, j, k = np.ones(3), np.zeros(3), np.zeros(3), np.zeros(3)
        >>> v = from_components(r, i, j, k)
        >>> isinstance(v, QuatVector)
        True
        >>> v.shape
        (3,)
    """
    r = np.asarray(r, dtype=float)
    i = np.asarray(i, dtype=float)
    j = np.asarray(j, dtype=float)
    k = np.asarray(k, dtype=float)
    data = np.stack([r, i, j, k], axis=-1)
    return from_ndarray(data)


def broadcast_quat_shapes(*shapes: tuple) -> tuple:
    """Compute the broadcast shape for quaternion collections."""
    return np.broadcast_shapes(*shapes)


def stack_quat(quaternions: list, axis: int = 0) -> QuatLike:
    """Stack a sequence of Quaternion or QuatVector objects.

    Example:
        >>> from quat import Quaternion, stack_quat, QuatVector
        >>> q1, q2 = Quaternion(1,0,0,0), Quaternion(0,1,0,0)
        >>> v = stack_quat([q1, q2])
        >>> isinstance(v, QuatVector)
        True
        >>> v.shape
        (2,)
    """
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


# -- numeric validation --------------------------------------------------------
def isnan(q: QuatLike) -> Union[bool, np.ndarray]:
    """Check for NaN in any quaternion component.  Delegates to ``q.isnan()``.

    Example:
        >>> from quat import Quaternion, isnan
        >>> isnan(Quaternion(1, float('nan'), 3, 4))
        True
        >>> isnan(Quaternion(1, 2, 3, 4))
        False
    """
    return q.isnan()


def isinf(q: QuatLike) -> Union[bool, np.ndarray]:
    """Check for Inf in any quaternion component.  Delegates to ``q.isinf()``.

    Example:
        >>> from quat import Quaternion, isinf
        >>> isinf(Quaternion(1, float('inf'), 3, 4))
        True
    """
    return q.isinf()


def isfinite(q: QuatLike) -> Union[bool, np.ndarray]:
    """Check that all quaternion components are finite.

    Example:
        >>> from quat import Quaternion, isfinite
        >>> isfinite(Quaternion(1, 2, 3, 4))
        True
    """
    return q.isfinite()


def isclose(q1: QuatLike, q2: QuatLike, rtol: float = 1e-05, atol: float = 1e-08) -> Union[bool, np.ndarray]:
    """Element-wise closeness test. Delegates to ``q1.isclose(q2)``.

    Example:
        >>> from quat import Quaternion, isclose
        >>> isclose(Quaternion(1,2,3,4), Quaternion(1,2,3,4))
        True
        >>> isclose(Quaternion(1,2,3,4), Quaternion(5,6,7,8))
        False
    """
    return q1.isclose(q2, rtol=rtol, atol=atol)
