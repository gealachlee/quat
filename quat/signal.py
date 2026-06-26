"""Quaternion signal processing — QFFT, convolution, and filter design."""
from __future__ import annotations
import numpy as np
from quat.algebra import _hamilton


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


def qfft2(x: np.ndarray, axes=(-2, -1), side: str = 'left') -> np.ndarray:
    X = qfft(x, axis=axes[0], side=side)
    return qfft(X, axis=axes[1], side=side)


def iqfft2(X: np.ndarray, axes=(-2, -1), side: str = 'left') -> np.ndarray:
    x = iqfft(X, axis=axes[0], side=side)
    return iqfft(x, axis=axes[1], side=side)


def qconv(x: np.ndarray, kernel: np.ndarray, mode: str = 'full') -> np.ndarray:
    x = np.asarray(x, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if x.ndim < 2 or x.shape[-1] != 4:
        raise ValueError(f"x last axis must have size 4 (r,i,j,k), got shape {x.shape}")
    if kernel.ndim < 2 or kernel.shape[-1] != 4:
        raise ValueError(f"kernel last axis must have size 4, got shape {kernel.shape}")
    if mode not in ('full', 'same', 'valid'):
        raise ValueError(f"mode must be 'full', 'same', or 'valid', got {mode!r}")
    n = x.shape[-2]
    k = kernel.shape[-2]
    N = n + k - 1
    x_pad = np.zeros(x.shape[:-2] + (N, 4))
    k_pad = np.zeros(kernel.shape[:-2] + (N, 4))
    x_pad[..., :n, :] = x
    k_pad[..., :k, :] = kernel
    X = qfft(x_pad, axis=-1)
    K = qfft(k_pad, axis=-1)
    Y = _hamilton(X, K)
    y_full = iqfft(Y, axis=-1)
    if mode == 'full':
        return y_full
    elif mode == 'same':
        start = (k - 1) // 2
        return y_full[..., start:start + n, :]
    else:
        return y_full[..., k - 1:n, :]
