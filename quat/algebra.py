"""Low-level quaternion algebra operations — constants, Hamilton product, real-matrix tensor."""
import numpy as np

_CONJ = np.array([1., -1., -1., -1.])

_REAL_LEFT = np.zeros((4, 4, 4))
_REAL_LEFT[0, 0, 0] = 1;   _REAL_LEFT[0, 1, 1] = -1;  _REAL_LEFT[0, 2, 2] = -1;  _REAL_LEFT[0, 3, 3] = -1
_REAL_LEFT[1, 0, 1] = 1;   _REAL_LEFT[1, 1, 0] = 1;   _REAL_LEFT[1, 2, 3] = -1;  _REAL_LEFT[1, 3, 2] = 1
_REAL_LEFT[2, 0, 2] = 1;   _REAL_LEFT[2, 1, 3] = 1;   _REAL_LEFT[2, 2, 0] = 1;   _REAL_LEFT[2, 3, 1] = -1
_REAL_LEFT[3, 0, 3] = 1;   _REAL_LEFT[3, 1, 2] = -1;  _REAL_LEFT[3, 2, 1] = 1;   _REAL_LEFT[3, 3, 0] = 1


def _hamilton(p, q):
    """Vectorized Hamilton product. p, q: broadcastable arrays, last dim == 4."""
    a1, b1, c1, d1 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    a2, b2, c2, d2 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    shp = np.broadcast_shapes(p.shape[:-1], q.shape[:-1]) + (4,)
    out = np.empty(shp)
    out[..., 0] = a1*a2 - b1*b2 - c1*c2 - d1*d2
    out[..., 1] = a1*b2 + b1*a2 + c1*d2 - d1*c2
    out[..., 2] = a1*c2 - b1*d2 + c1*a2 + d1*b2
    out[..., 3] = a1*d2 + b1*c2 - c1*b2 + d1*a2
    return out
