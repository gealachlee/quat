"""Validation helpers for quaternion collection types.

Each function accepts a ``(..., 4)`` ndarray of quaternion component data
and returns a per-element boolean ndarray (or scalar for Quaternion's 1-D
case).  Used by QuatVector, QuatMatrix, and QuatTensor to avoid duplicating
the same numpy calls across three classes.
"""
from __future__ import annotations
import numpy as np


def _vec_isnan(data: np.ndarray) -> np.ndarray:
    """True where any quaternion component is NaN along the last axis."""
    return np.any(np.isnan(data), axis=-1)


def _vec_isinf(data: np.ndarray) -> np.ndarray:
    """True where any quaternion component is infinite along the last axis."""
    return np.any(np.isinf(data), axis=-1)


def _vec_isfinite(data: np.ndarray) -> np.ndarray:
    """True where all quaternion components are finite along the last axis."""
    return np.all(np.isfinite(data), axis=-1)


def _vec_isclose(data: np.ndarray, other_data: np.ndarray,
                 rtol: float, atol: float) -> np.ndarray:
    """Element-wise closeness test — all four components must be close along the last axis."""
    return np.isclose(data, other_data, rtol=rtol, atol=atol).all(axis=-1)
