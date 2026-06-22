# =========================================================================
#  LLM-generated code.  See README.md for full disclosure.
# =========================================================================

"""Quaternion interpolation — SLERP and related utilities."""

import numpy as np
from quat.core import Quaternion


def slerp(q0: Quaternion, q1: Quaternion, t: float) -> Quaternion:
    """Spherical linear interpolation between two unit quaternions.

    Always takes the shortest arc on S³ by flipping sign if necessary.

    Args:
        q0, q1: Quaternions (should be unit-norm).
        t: Interpolation parameter in [0, 1].

    Returns:
        Interpolated quaternion.

    Example:
        >>> from quat import Quaternion, slerp
        >>> q0 = Quaternion(1, 0, 0, 0)
        >>> q1 = Quaternion(0, 1, 0, 0)
        >>> mid = slerp(q0, q1, 0.5)
        >>> abs(mid.norm() - 1.0) < 1e-10
        True
    """
    cos_theta = q0.re_inner(q1)
    if cos_theta < 0.0:
        q1 = -q1
        cos_theta = -cos_theta
    if cos_theta > 1.0 - 1e-12:
        return Quaternion(q0._data + t * (q1._data - q0._data))
    theta = np.arccos(float(cos_theta))
    s = float(np.sin(theta))
    w0 = float(np.sin((1.0 - t) * theta) / s)
    w1 = float(np.sin(t * theta) / s)
    return Quaternion(w0 * q0._data + w1 * q1._data)


def slerp_vector(v0: "QuatVector", v1: "QuatVector", t: np.ndarray | float) -> "QuatVector":
    """Batch SLERP for pairs of QuatVector.

    Args:
        v0, v1: QuatVector with same length.
        t: Scalar or array of interpolation parameters.

    Returns:
        QuatVector of interpolated quaternions.

    Example:
        >>> from quat import Quaternion, QuatVector, slerp_vector
        >>> import numpy as np
        >>> v0 = QuatVector([Quaternion(1,0,0,0), Quaternion(0,1,0,0)])
        >>> v1 = QuatVector([Quaternion(0,1,0,0), Quaternion(-1,0,0,0)])
        >>> mid = slerp_vector(v0, v1, 0.5)
        >>> mid.shape
        (2,)
    """
    from quat.collections import QuatVector
    n = len(v0)
    t_arr = np.broadcast_to(np.asarray(t, dtype=float), (n,))
    result = np.empty((n, 4))
    for i in range(n):
        q0 = Quaternion(v0._data[i])
        q1 = Quaternion(v1._data[i])
        result[i] = slerp(q0, q1, float(t_arr[i]))._data
    return QuatVector(result)


def squad(q0: Quaternion, q1: Quaternion, q2: Quaternion, q3: Quaternion,
          t: float) -> Quaternion:
    """Spherical cubic (squad) interpolation between q1 and q2.

    Uses q0 and q3 as boundary tangents for C¹ continuity through the
    four control points.  Based on Shoemake (1987).

    Args:
        q0, q1, q2, q3: Four consecutive unit quaternion keyframes.
        t: Interpolation parameter in [0, 1] (0→q1, 1→q2).

    Returns:
        Interpolated unit quaternion.

    Example:
        >>> from quat import Quaternion, squad
        >>> q0 = Quaternion(1,0,0,0)
        >>> q1 = Quaternion(0,1,0,0)
        >>> q2 = Quaternion(-1,0,0,0)
        >>> q3 = Quaternion(0,0,1,0)
        >>> mid = squad(q0, q1, q2, q3, 0.0)
        >>> mid.isclose(q1)
        True
    """
    log_a = (q1.inverse() * q0).log()
    log_b = (q1.inverse() * q2).log()
    tangent1 = -(log_a._data + log_b._data) / 4.
    s1 = q1 * Quaternion(tangent1).exp()

    log_a = (q2.inverse() * q1).log()
    log_b = (q2.inverse() * q3).log()
    tangent2 = -(log_a._data + log_b._data) / 4.
    s2 = q2 * Quaternion(tangent2).exp()

    t2 = 2.0 * t * (1.0 - t)
    return slerp(slerp(q1, q2, t), slerp(s1, s2, t), t2)
