# =========================================================================
#  LLM-generated code.  See README.md for full disclosure.
# =========================================================================

"""Low-level quaternion algebra — constants, Hamilton product, real-matrix tensor."""
from __future__ import annotations
import numpy as np
import numpy.typing as npt

_CONJ = np.array([1., -1., -1., -1.])
"""Conjugate mask: ``_CONJ * q`` negates the three imaginary components."""

_REAL_LEFT = np.zeros((4, 4, 4))
"""Left-regular real-representation tensor.

``L(q)[r, c] = Σ_k _REAL_LEFT[r, c, k] * q[k]`` yields the 4×4 real matrix
satisfying ``L(q) @ vec(x) = vec(q * x)``.
"""
_REAL_LEFT[0, 0, 0] = 1;   _REAL_LEFT[0, 1, 1] = -1;  _REAL_LEFT[0, 2, 2] = -1;  _REAL_LEFT[0, 3, 3] = -1
_REAL_LEFT[1, 0, 1] = 1;   _REAL_LEFT[1, 1, 0] = 1;   _REAL_LEFT[1, 2, 3] = -1;  _REAL_LEFT[1, 3, 2] = 1
_REAL_LEFT[2, 0, 2] = 1;   _REAL_LEFT[2, 1, 3] = 1;   _REAL_LEFT[2, 2, 0] = 1;   _REAL_LEFT[2, 3, 1] = -1
_REAL_LEFT[3, 0, 3] = 1;   _REAL_LEFT[3, 1, 2] = -1;  _REAL_LEFT[3, 2, 1] = 1;   _REAL_LEFT[3, 3, 0] = 1

_HAMILTON_TENSOR = np.zeros((4, 4, 4))
_HAMILTON_TENSOR[0, 0, 0] = 1;    _HAMILTON_TENSOR[0, 1, 1] = -1;   _HAMILTON_TENSOR[0, 2, 2] = -1;   _HAMILTON_TENSOR[0, 3, 3] = -1
_HAMILTON_TENSOR[1, 0, 1] = 1;    _HAMILTON_TENSOR[1, 1, 0] = 1;    _HAMILTON_TENSOR[1, 2, 3] = 1;    _HAMILTON_TENSOR[1, 3, 2] = -1
_HAMILTON_TENSOR[2, 0, 2] = 1;    _HAMILTON_TENSOR[2, 1, 3] = -1;   _HAMILTON_TENSOR[2, 2, 0] = 1;    _HAMILTON_TENSOR[2, 3, 1] = 1
_HAMILTON_TENSOR[3, 0, 3] = 1;    _HAMILTON_TENSOR[3, 1, 2] = 1;    _HAMILTON_TENSOR[3, 2, 1] = -1;   _HAMILTON_TENSOR[3, 3, 0] = 1


_SMALL_THRESHOLD = 500
_LARGE_THRESHOLD = 5000


def _hamilton(p: npt.NDArray, q: npt.NDArray) -> npt.NDArray:
    """Vectorized Hamilton (quaternion) product.

    Dispatches to the optimal kernel based on data size:
      - small (<=500 elements): component-wise arithmetic
      - medium (500-5000): einsum without contraction-path optimisation
      - large (>5000): einsum with full contraction-path optimisation

    Supports arbitrary leading-dimension broadcasting.
    """
    total_elements = p.size + q.size
    if total_elements <= _SMALL_THRESHOLD:
        return _hamilton_component(p, q)
    if total_elements <= _LARGE_THRESHOLD:
        return _hamilton_einsum_noopt(p, q)
    return _hamilton_einsum(p, q)


def _hamilton_component(p: npt.NDArray, q: npt.NDArray) -> npt.NDArray:
    """Component-wise Hamilton product — fastest for small batches."""
    a1, b1, c1, d1 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    a2, b2, c2, d2 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    shp = np.broadcast_shapes(p.shape[:-1], q.shape[:-1]) + (4,)
    out = np.empty(shp)
    out[..., 0] = a1*a2 - b1*b2 - c1*c2 - d1*d2
    out[..., 1] = a1*b2 + b1*a2 + c1*d2 - d1*c2
    out[..., 2] = a1*c2 - b1*d2 + c1*a2 + d1*b2
    out[..., 3] = a1*d2 + b1*c2 - c1*b2 + d1*a2
    return out


def _hamilton_einsum_noopt(p: npt.NDArray, q: npt.NDArray) -> npt.NDArray:
    """Einsum without contraction-path optimisation — faster for medium batches
    where the optimisation overhead outweighs its benefit."""
    return np.einsum('rck,...c,...k->...r', _HAMILTON_TENSOR, p, q, optimize=False)


def _hamilton_einsum(p: npt.NDArray, q: npt.NDArray) -> npt.NDArray:
    """Einsum with full contraction-path optimisation — best throughput for large
    batches where the optimisation cost is amortised."""
    return np.einsum('rck,...c,...k->...r', _HAMILTON_TENSOR, p, q, optimize=True)
