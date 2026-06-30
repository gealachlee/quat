"""Quaternion distance metrics — geodesic (intrinsic) and chordal (Euclidean)."""
from __future__ import annotations

import numpy as np
from quat.core import Quaternion
from quat.collections import QuatVector


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
        which is the shortest geodesic distance between the two unit
        quaternions on :math:`S^3`, chosen over the ± covering.

        Args:
            q1, q2: Unit quaternions (scalar or :class:`QuatVector`).

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

        Defined as ``min(‖q₁ - q₂‖, ‖q₁ + q₂‖)``, which accounts for
        the ± sign ambiguity of quaternion-encoded rotations.

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
        the angle between the two quaternions as vectors on the
        3-sphere.

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

        Computes ``‖q₁ − q₂‖`` directly.  Does *not* account for
        the ± sign of quaternion representations.

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
