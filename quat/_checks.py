"""Validation helpers for quaternion collection types."""
from __future__ import annotations
import numpy as np


def _vec_isnan(data: np.ndarray) -> np.ndarray:
    return np.any(np.isnan(data), axis=-1)


def _vec_isinf(data: np.ndarray) -> np.ndarray:
    return np.any(np.isinf(data), axis=-1)


def _vec_isfinite(data: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(data), axis=-1)


def _vec_isclose(data: np.ndarray, other_data: np.ndarray,
                 rtol: float, atol: float) -> np.ndarray:
    return np.isclose(data, other_data, rtol=rtol, atol=atol).all(axis=-1)
