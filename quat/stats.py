"""Quaternion component-space statistics — mean, covariance, PCA."""
from __future__ import annotations

import numpy as np
from quat.core import Quaternion


def quaternion_mean(data: np.ndarray) -> Quaternion:
    """Element-wise mean of quaternion component vectors.

    Treats each quaternion as a 4D vector and computes the arithmetic mean.

    Args:
        data: ``ndarray`` of shape ``(..., 4)``.

    Returns:
        Quaternion whose components are the element-wise means.

    Example:
        >>> from quat.stats import quaternion_mean
        >>> import numpy as np
        >>> data = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])
        >>> mu = quaternion_mean(data)
        >>> mu.components
        (3.0, 4.0, 5.0, 6.0)
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

    Example:
        >>> from quat.stats import quaternion_cov
        >>> import numpy as np
        >>> data = np.random.randn(100, 4)
        >>> C = quaternion_cov(data)
        >>> C.shape
        (4, 4)
    """
    if mu is None:
        mu = quaternion_mean(data)
    centered = data - mu._data
    return centered.T @ centered / (data.shape[0] - 1)


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

    Example:
        >>> from quat.stats import quaternion_pca
        >>> import numpy as np
        >>> data = np.random.randn(100, 4)
        >>> comp, ev = quaternion_pca(data, n_components=2)
        >>> comp.shape
        (2, 4)
    """
    mu = quaternion_mean(data)
    centered = data - mu._data
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    n = data.shape[0]
    explained_variance = (s[:n_components] ** 2) / (n - 1)
    return components, explained_variance
