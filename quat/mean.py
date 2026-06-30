"""Quaternion mean rotation — SVD-based and Riemannian (Karcher) averaging."""
from __future__ import annotations

import numpy as np
from quat.core import Quaternion
from quat.collections import QuatVector


def mean_rotation(
    qv: QuatVector,
    weights: np.ndarray | None = None,
) -> Quaternion:
    """Average rotation via SVD of the quaternion data matrix.

    The mean is the dominant singular vector of the weighted data
    matrix, maximising :math:`\\operatorname{tr}(Q^\\top W Q_m)`
    over unit quaternions.  This is the closed-form solution of the
    least-squares estimation on S³ (Gramkow 2001 / Markley 2007).

    Args:
        qv: QuatVector of unit quaternions, shape ``(n,)``.
        weights: Optional ``(n,)`` array of positive weights.

    Returns:
        Unit quaternion — the mean rotation.

    Example:
        >>> from quat.collections import QuatVector
        >>> from quat.mean import mean_rotation
        >>> import numpy as np
        >>> data = np.eye(4)[np.random.choice(4, 100)]
        >>> qv = QuatVector(data)
        >>> m = mean_rotation(qv)
        >>> abs(m.norm() - 1.0) < 1e-5
        True
    """
    data = qv._data  # (n, 4)
    if weights is not None:
        data = data * np.asarray(weights, dtype=float)[:, np.newaxis]
    _, _, vh = np.linalg.svd(data, full_matrices=False)
    q = vh[0]
    nrm = np.linalg.norm(q)
    if nrm < 1e-15:
        return Quaternion(1, 0, 0, 0)
    return Quaternion(q / nrm)


def karcher_mean(
    qv: QuatVector,
    weights: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Quaternion:
    """Karcher (Riemannian) mean via gradient descent on S³.

    Iteratively minimises the sum of squared geodesic distances.
    More accurate than ``mean_rotation`` for widely scattered data
    but requires O(n × iterations) computation.

    Args:
        qv: QuatVector of unit quaternions.
        weights: Optional ``(n,)`` array of positive weights.
        tol: Convergence threshold on the gradient norm.
        max_iter: Maximum number of iterations.

    Returns:
        Unit quaternion — the Riemannian centre-of-mass.

    Example:
        >>> from quat.collections import QuatVector
        >>> from quat.mean import karcher_mean
        >>> import numpy as np
        >>> data = np.eye(4)[np.random.choice(4, 100)]
        >>> qv = QuatVector(data)
        >>> m = karcher_mean(qv, max_iter=30)
        >>> abs(m.norm() - 1.0) < 1e-5
        True
    """
    data = qv._data  # (n, 4)
    w = np.asarray(weights, dtype=float) if weights is not None else np.ones(data.shape[0])
    w = w / w.sum()

    mu_data = mean_rotation(qv).normalize()._data
    for _ in range(max_iter):
        dot = np.sum(mu_data * data, axis=-1)
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(np.abs(dot))
        v = data - dot[:, np.newaxis] * mu_data
        v_norms = np.linalg.norm(v, axis=-1)
        mask = v_norms > 1e-15
        delta = np.zeros(4)
        if np.any(mask):
            weighted = (v[mask] / v_norms[mask, np.newaxis]) * (theta[mask, np.newaxis] * w[mask, np.newaxis])
            delta = weighted.sum(axis=0)
        delta_norm = np.linalg.norm(delta)
        if delta_norm < tol:
            break
        mu_data = mu_data + delta
        nrm = np.linalg.norm(mu_data)
        if nrm < 1e-15:
            break
        mu_data = mu_data / nrm
    return Quaternion(mu_data)
