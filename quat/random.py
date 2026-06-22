# =========================================================================
#  LLM-generated code.  See README.md for full disclosure.
# =========================================================================

"""Random quaternion generators."""
import numpy as np
from quat.core import Quaternion
from quat.collections import QuatVector, QuatMatrix, QuatTensor


def random_quat(rng: np.random.Generator | int | None = None) -> Quaternion:
    """Generate a random quaternion with standard-normal components.

    Args:
        rng: NumPy Generator, seed integer, or None for default.

    Returns:
        Quaternion with random real+imaginary components.

    Example:
        >>> from quat.random import random_quat
        >>> q = random_quat(42)
        >>> q.norm() > 0
        True
    """
    rng = _ensure_generator(rng)
    return Quaternion(rng.standard_normal(4))


def random_unit_quat(rng: np.random.Generator | int | None = None) -> Quaternion:
    """Generate a random unit quaternion (uniform on S³).

    Example:
        >>> from quat.random import random_unit_quat
        >>> q = random_unit_quat(42)
        >>> abs(q.norm() - 1.0) < 1e-10
        True
    """
    rng = _ensure_generator(rng)
    v = rng.standard_normal(4)
    return Quaternion(v / np.linalg.norm(v))


def random_quat_vector(n: int, rng: np.random.Generator | int | None = None) -> QuatVector:
    """Generate a QuatVector of *n* random quaternions.

    Example:
        >>> from quat.random import random_quat_vector
        >>> v = random_quat_vector(5, 42)
        >>> v.shape
        (5,)
    """
    rng = _ensure_generator(rng)
    return QuatVector(rng.standard_normal((n, 4)))


def random_quat_matrix(m: int, n: int, rng: np.random.Generator | int | None = None) -> QuatMatrix:
    """Generate an m×n QuatMatrix of random quaternions.

    Example:
        >>> from quat.random import random_quat_matrix
        >>> M = random_quat_matrix(3, 4, 42)
        >>> M.shape
        (3, 4)
    """
    rng = _ensure_generator(rng)
    return QuatMatrix(rng.standard_normal((m, n, 4)))


def random_quat_tensor(p: int, q: int, r: int, rng: np.random.Generator | int | None = None) -> QuatTensor:
    """Generate a p×q×r QuatTensor of random quaternions.

    Example:
        >>> from quat.random import random_quat_tensor
        >>> T = random_quat_tensor(2, 3, 4, 42)
        >>> T.shape
        (2, 3, 4)
    """
    rng = _ensure_generator(rng)
    return QuatTensor(rng.standard_normal((p, q, r, 4)))


def _ensure_generator(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Coerce seed or None to a NumPy Generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)
