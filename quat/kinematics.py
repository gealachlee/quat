"""Quaternion kinematics — angular velocity, integration, and frame rotation."""
from __future__ import annotations

import numpy as np
from quat.core import Quaternion
from quat.collections import QuatVector
from quat.algebra import _hamilton, _CONJ


def angular_velocity(
    q_seq: QuatVector,
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

    Example:
        >>> from quat.core import Quaternion
        >>> from quat.collections import QuatVector
        >>> from quat.kinematics import angular_velocity
        >>> import numpy as np
        >>> dt = 0.01
        >>> qs = [Quaternion.from_axis_angle((0,0,1), i*dt) for i in range(100)]
        >>> w = angular_velocity(QuatVector(qs), dt)
        >>> w.shape
        (99, 3)
    """
    q_data = q_seq._data  # (n, 4)
    n = q_data.shape[0]
    if np.isscalar(t):
        dt = float(t)
    else:
        t_arr = np.asarray(t, dtype=float).ravel()
        dt = np.mean(np.diff(t_arr)) if len(t_arr) == n else float(t)

    # Δq[i] = q_{i+1} * conjugate(q_i)
    q_conj = q_data[:-1] * _CONJ
    q_diff = _hamilton(q_data[1:], q_conj)  # (n-1, 4)

    norms = np.sqrt(np.sum(q_diff ** 2, axis=-1, keepdims=True))
    norms = np.where(norms < 1e-15, 1.0, norms)
    q_diff = q_diff / norms

    omega = 2.0 * q_diff[:, 1:] / dt  # (n-1, 3)
    return omega


def integrate_angular_velocity(
    omega: np.ndarray,
    dt: float,
    q0: Quaternion | None = None,
) -> QuatVector:
    """Integrate angular velocity samples to a quaternion trajectory.

    Uses Euler integration with re-normalisation at each step.

    Args:
        omega: ``(n, 3)`` array — angular velocity samples.
        dt: Time step.
        q0: Initial quaternion (defaults to identity).

    Returns:
        QuatVector of shape ``(n+1,)``.

    Example:
        >>> from quat.kinematics import integrate_angular_velocity
        >>> import numpy as np
        >>> w = np.zeros((100, 3)); w[:, 2] = 1.0  # 1 rad/s about z
        >>> traj = integrate_angular_velocity(w, 0.01)
        >>> traj.shape
        (101,)
    """
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
    return QuatVector(q_data)


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

    Example:
        >>> from quat.core import Quaternion
        >>> from quat.kinematics import rotate_frame
        >>> q = Quaternion(1, 0, 0, 0)
        >>> result = rotate_frame(q, (0, 0, 1), 0.5)
        >>> result.components  # doctest: +SKIP
        (0.968..., 0.0, 0.0, 0.247...)
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
