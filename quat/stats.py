"""Quaternion statistics — distance metrics, mean rotation, component-space PCA."""
from __future__ import annotations

import numpy as np
from quat.core import Quaternion
from quat.collections import QuatVector


# =============================================================================
# Distance metrics
# =============================================================================


def _normalized_dot(q1, q2):
    """Compute quaternion dot product, normalized by norms."""
    d = np.sum(q1._data * q2._data, axis=-1)
    n1 = np.sqrt(np.sum(q1._data ** 2, axis=-1))
    n2 = np.sqrt(np.sum(q2._data ** 2, axis=-1))
    n = np.where(n1 * n2 < 1e-15, 1.0, n1 * n2)
    return d / n


class rotation:
    """Distance between rotations — handles ±q double cover."""

    @staticmethod
    def intrinsic(q1: Quaternion, q2: Quaternion) -> float:
        r"""Geodesic distance on SO(3) via the minimal rotation angle.

        Computes :math:`2 \arccos(|\langle q_1, \bar{q}_2 \rangle|)`,
        which is the shortest geodesic distance on :math:`S^3`,
        chosen over the ± covering.

        Args:
            q1, q2: Unit quaternions.

        Returns:
            Distance in radians (range [0, π]).
        """
        dot = float(_normalized_dot(q1, q2))
        dot = abs(dot)
        if dot > 1.0:
            dot = 1.0
        return float(2.0 * np.arccos(dot))

    @staticmethod
    def chordal(q1: Quaternion, q2: Quaternion) -> float:
        """Euclidean (chordal) distance in the embedding space.

        Defined as ``min(‖q₁ - q₂‖, ‖q₁ + q₂‖)``.

        Args:
            q1, q2: Unit quaternions.

        Returns:
            Chordal distance (range [0, √2]).
        """
        dot = float(abs(_normalized_dot(q1, q2)))
        return float(np.sqrt(max(2.0 - 2.0 * dot, 0.0)))

    @staticmethod
    def intrinsic_batch(v1: QuatVector, v2: QuatVector) -> np.ndarray:
        """Batch version of :meth:`intrinsic`."""
        dot = np.abs(_normalized_dot(v1, v2))
        dot = np.clip(dot, 0.0, 1.0)
        return 2.0 * np.arccos(dot)

    @staticmethod
    def chordal_batch(v1: QuatVector, v2: QuatVector) -> np.ndarray:
        """Batch version of :meth:`chordal`."""
        dot = np.abs(_normalized_dot(v1, v2))
        return np.sqrt(np.clip(2.0 - 2.0 * dot, 0.0, None))


class rotor:
    """Distance between rotors — does *not* handle ±q sign."""

    @staticmethod
    def intrinsic(q1: Quaternion, q2: Quaternion) -> float:
        r"""Geodesic distance on :math:`S^3`.

        Computes :math:`\arccos(\langle q_1, \bar{q}_2 \rangle)`,
        the angle between two quaternions as vectors on the 3-sphere.

        Args:
            q1, q2: Unit quaternions.

        Returns:
            Distance in radians (range [0, π]).
        """
        dot = float(_normalized_dot(q1, q2))
        dot = float(np.clip(dot, -1.0, 1.0))
        return float(np.arccos(dot))

    @staticmethod
    def chordal(q1: Quaternion, q2: Quaternion) -> float:
        r"""Euclidean (chordal) distance in :math:`\mathbb{R}^4`.

        Args:
            q1, q2: Unit quaternions.

        Returns:
            Chordal distance (range [0, 2]).
        """
        dot = float(_normalized_dot(q1, q2))
        return float(np.sqrt(max(2.0 - 2.0 * dot, 0.0)))

    @staticmethod
    def intrinsic_batch(v1: QuatVector, v2: QuatVector) -> np.ndarray:
        """Batch version of :meth:`intrinsic`."""
        dot = np.clip(_normalized_dot(v1, v2), -1.0, 1.0)
        return np.arccos(dot)

    @staticmethod
    def chordal_batch(v1: QuatVector, v2: QuatVector) -> np.ndarray:
        """Batch version of :meth:`chordal`."""
        dot = _normalized_dot(v1, v2)
        return np.sqrt(np.clip(2.0 - 2.0 * dot, 0.0, None))


# =============================================================================
# Mean rotation
# =============================================================================


def mean_rotation(
    qv: QuatVector,
    weights: np.ndarray | None = None,
) -> Quaternion:
    """Average rotation via SVD of the quaternion data matrix.

    The mean is the dominant singular vector of the weighted data
    matrix, maximising :math:`\\operatorname{tr}(Q^\\top W Q_m)`
    over unit quaternions (Gramkow 2001 / Markley 2007).

    Args:
        qv: QuatVector of unit quaternions, shape ``(n,)``.
        weights: Optional ``(n,)`` array of positive weights.

    Returns:
        Unit quaternion — the mean rotation.
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


def approximate_karcher_mean(
    qv: QuatVector,
    weights: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Quaternion:
    """Karcher (Riemannian) mean via gradient descent on S³.

    Iteratively minimises the sum of squared geodesic distances.

    Args:
        qv: QuatVector of unit quaternions.
        weights: Optional ``(n,)`` array of positive weights.
        tol: Convergence threshold on the gradient norm.
        max_iter: Maximum number of iterations.

    Returns:
        Unit quaternion — the Riemannian centre-of-mass.
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


# =============================================================================
# Component-space statistics
# =============================================================================


def quaternion_mean(data: np.ndarray) -> Quaternion:
    """Element-wise mean of quaternion component vectors.

    Treats each quaternion as a 4D vector and computes the arithmetic mean.

    Args:
        data: ``ndarray`` of shape ``(..., 4)``.

    Returns:
        Quaternion whose components are the element-wise means.
    """
    return Quaternion(np.mean(data, axis=0))


def quaternion_cov(
    data: np.ndarray,
    mu: Quaternion | None = None,
) -> np.ndarray:
    """Sample covariance matrix of quaternion component vectors.

    Treats the 4 components as a multivariate random vector and
    returns the standard 4×4 sample covariance.

    Args:
        data: ``ndarray`` of shape ``(n, 4)``.
        mu: Pre-computed mean (optional, computed if not given).

    Returns:
        ``ndarray`` of shape ``(4, 4)``.
    """
    if mu is None:
        mu = quaternion_mean(data)
    centered = data - mu._data
    n = data.shape[0]
    if n <= 1:
        return np.zeros((4, 4))
    return centered.T @ centered / (n - 1)


def quaternion_pca(
    data: np.ndarray,
    n_components: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Principal Component Analysis on quaternion component space.

    Decomposes the 4D quaternion-component dataset into principal
    directions via SVD.

    Args:
        data: ``ndarray`` of shape ``(n, 4)``.
        n_components: Number of principal components to return (1-4).

    Returns:
        ``(components, explained_variance)`` where *components* has
        shape ``(n_components, 4)`` and *explained_variance* has
        shape ``(n_components,)``.
    """
    mu = quaternion_mean(data)
    centered = data - mu._data
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    n = data.shape[0]
    if n <= 1:
        return components, np.zeros(n_components)
    explained_variance = (s[:n_components] ** 2) / (n - 1)
    return components, explained_variance
