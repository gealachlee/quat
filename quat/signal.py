"""Quaternion signal processing — QFFT, convolution, and filter design."""
from __future__ import annotations
import numpy as np


def qfft(x: np.ndarray, axis: int = -1, side: str = 'left') -> np.ndarray:
    if side not in ('left', 'right'):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    x = np.asarray(x, dtype=float)
    a, b, c, d = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    s1 = a + 1j * b
    s2 = c + 1j * d
    S1 = np.fft.fft(s1, axis=axis)
    S2 = np.fft.fft(s2, axis=axis)
    if side == 'right':
        S2 = S2.conjugate()
    result = np.empty_like(x)
    result[..., 0] = S1.real
    result[..., 1] = S1.imag
    result[..., 2] = S2.real
    result[..., 3] = S2.imag
    return result


def iqfft(X: np.ndarray, axis: int = -1, side: str = 'left') -> np.ndarray:
    if side not in ('left', 'right'):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    X = np.asarray(X, dtype=float)
    a, b, c, d = X[..., 0], X[..., 1], X[..., 2], X[..., 3]
    s1 = a + 1j * b
    s2 = c + 1j * d
    S1 = np.fft.ifft(s1, axis=axis)
    S2 = np.fft.ifft(s2, axis=axis)
    if side == 'right':
        S2 = S2.conjugate()
    result = np.empty_like(X)
    result[..., 0] = S1.real
    result[..., 1] = S1.imag
    result[..., 2] = S2.real
    result[..., 3] = S2.imag
    return result
