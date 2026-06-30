# =========================================================================
#  LLM-generated code.  See README.md for full disclosure.
# =========================================================================

"""Quaternion interpolation — SLERP and related utilities."""

import numpy as np
from quat.core import Quaternion
from quat.algebra import _hamilton, _CONJ


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
    theta = np.arccos(cos_theta)
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
    p = v0._data
    q = v1._data.copy()
    cos_theta = (p * q).sum(axis=1)
    flip = cos_theta < 0.0
    q[flip] *= -1.0
    cos_theta[flip] *= -1.0
    near_ident = cos_theta > 1.0 - 1e-12
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    s = np.sin(theta)
    s[near_ident] = 1.0
    w0 = np.sin((1.0 - t_arr) * theta) / s
    w1 = np.sin(t_arr * theta) / s
    result = p + t_arr[:, None] * (q - p)
    not_near = ~near_ident
    if np.any(not_near):
        result[not_near] = w0[not_near, None] * p[not_near] + w1[not_near, None] * q[not_near]
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


# ---------------------------------------------------------------------------
# Kinematics — angular velocity / integration / frame rotation
# ---------------------------------------------------------------------------


def angular_velocity(
    q_seq: "QuatVector",
    t: float | np.ndarray,
) -> np.ndarray:
    """Estimate angular velocity from a quaternion trajectory.

    Uses finite differences on the quaternion manifold: the relative
    rotation between consecutive quaternions is extracted and converted
    to an angular velocity estimate.

    Args:
        q_seq: QuatVector of shape ``(n,)`` — unit quaternion trajectory.
        t: Scalar *dt* for uniform sampling, or ``(n,)`` time array.

    Returns:
        ``ndarray`` of shape ``(n-1, 3)`` — body-frame angular velocity
        in rad/s.
    """
    from quat.collections import QuatVector as _QV
    q_data = q_seq._data  # (n, 4)
    n = q_data.shape[0]
    if np.isscalar(t):
        dt = float(t)
    else:
        t_arr = np.asarray(t, dtype=float).ravel()
        dt = np.mean(np.diff(t_arr)) if len(t_arr) == n else float(t)

    q_conj = q_data[:-1] * _CONJ
    q_diff = _hamilton(q_data[1:], q_conj)  # (n-1, 4)

    norms = np.sqrt(np.sum(q_diff ** 2, axis=-1, keepdims=True))
    zero_mask = (norms < 1e-15).squeeze(-1)
    norms[zero_mask] = 1.0
    q_diff = q_diff / norms

    omega = 2.0 * q_diff[:, 1:] / dt  # (n-1, 3)
    omega[zero_mask] = 0.0
    return omega


def integrate_angular_velocity(
    omega: np.ndarray,
    dt: float,
    q0: Quaternion | None = None,
) -> "QuatVector":
    """Integrate angular velocity samples to a quaternion trajectory.

    Uses Euler integration with re-normalisation at each step.

    Args:
        omega: ``(n, 3)`` array — angular velocity samples.
        dt: Time step.
        q0: Initial quaternion (defaults to identity).

    Returns:
        QuatVector of shape ``(n+1,)``.
    """
    from quat.collections import QuatVector as _QV
    if q0 is None:
        q0 = Quaternion(1, 0, 0, 0)
    n = omega.shape[0]
    q_data = np.empty((n + 1, 4))
    q_data[0] = q0._data.copy()
    for i in range(n):
        w = omega[i]
        p = q_data[i]
        q_dot = np.zeros(4)
        q_dot[0] = -0.5 * (p[1] * w[0] + p[2] * w[1] + p[3] * w[2])
        q_dot[1] =  0.5 * (p[0] * w[0] + p[2] * w[2] - p[3] * w[1])
        q_dot[2] =  0.5 * (p[0] * w[1] + p[3] * w[0] - p[1] * w[2])
        q_dot[3] =  0.5 * (p[0] * w[2] + p[1] * w[1] - p[2] * w[0])
        q_new = p + dt * q_dot
        nrm = np.linalg.norm(q_new)
        q_data[i + 1] = q_new / (nrm if nrm > 0 else 1.0)
    return _QV(q_data)


def rotate_frame(
    q: Quaternion,
    axis,
    angle: float,
) -> Quaternion:
    """Rotate a quaternion reference frame about a given axis.

    Equivalent to pre-multiplying by the axis-angle rotation quaternion.

    Args:
        q: Quaternion representing the current frame.
        axis: 3-vector rotation axis (normalized internally).
        angle: Rotation angle in radians.

    Returns:
        Rotated quaternion (r * q).
    """
    axis = np.asarray(axis, dtype=float)
    nrm = np.linalg.norm(axis)
    if nrm < 1e-15:
        return Quaternion(q._data.copy())
    axis = axis / nrm
    half = angle / 2.0
    s = np.sin(half)
    r = Quaternion(np.cos(half), s * axis[0], s * axis[1], s * axis[2])
    return r * q
