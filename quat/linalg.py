# =========================================================================
#  LLM-generated code.  See README.md for full disclosure.
# =========================================================================

"""Quaternion linear algebra — SVD, rank, condition number, pseudo-inverse."""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from quat.core import Quaternion
from quat.collections import QuatVector, QuatMatrix


def _svd_values(A: QuatMatrix) -> np.ndarray:
    """Compute only singular values (no vectors) via complex representation."""
    A_complex = A.to_complex_matrix()
    s_full = np.linalg.svd(A_complex, compute_uv=False)
    k = min(A.shape[0], A.shape[1])
    return s_full[::2][:k]


def svd(A: QuatMatrix) -> Tuple[QuatMatrix, np.ndarray, QuatMatrix]:
    """Quaternion singular value decomposition.

    Decompose a quaternion matrix A (m×n) into::

        A = U * Σ * V^H

    where U is m×m unitary quaternion matrix, Σ is m×n real diagonal
    (singular values), and V^H is the conjugate-transpose of an n×n
    unitary quaternion matrix.

    Implemented via the real (left-regular) representation: each quaternion
    element is expanded to a 4×4 real block, forming a 4m×4n real matrix
    whose SVD is computed, then the quaternion structure is reconstructed.
    For singular values only use ``_svd_values()`` which uses the faster
    complex representation path.

    Args:
        A: QuatMatrix of shape (m, n).

    Returns:
        ``(U, s, Vh)`` where:
          - *U*:  QuatMatrix (m, m) — left singular vectors.
          - *s*:  ``ndarray`` (k,) where k = min(m, n) — singular values.
          - *Vh*: QuatMatrix (n, n) — right singular vectors (conjugate-transposed).

    Example:
        >>> from quat import QuatMatrix, svd
        >>> import numpy as np
        >>> A = QuatMatrix(np.random.randn(4, 5, 4))
        >>> U, s, Vh = svd(A)
        >>> S = QuatMatrix.zeros(4, 5)
        >>> for i in range(len(s)):
        ...     S._data[i, i, 0] = s[i]
        >>> recon = U * S * Vh
        >>> np.allclose(A.to_array(), recon.to_array())
        True
    """
    A_real = A.to_real_matrix_left()
    U_real, s_full, Vt_real = np.linalg.svd(A_real, full_matrices=True)
    k = min(A.shape[0], A.shape[1])
    s = s_full[::4][:k]

    m, n = A.shape
    U_data = U_real.reshape(m, 4, m, 4)[:, :, :, 0].transpose(0, 2, 1)
    U = QuatMatrix(U_data)

    V_data = Vt_real.reshape(n, 4, n, 4)[:, 0, :, :].transpose(1, 0, 2)
    V = QuatMatrix(V_data)

    return U, s, V.H


def rank(A: QuatMatrix, tol: Optional[float] = None) -> int:
    """Compute the quaternion matrix rank via SVD.

    Args:
        A: QuatMatrix.
        tol: Singular value threshold.  Defaults to
            ``max(m, n) * σ_max * machine_epsilon``.

    Returns:
        int — number of singular values exceeding *tol*.

    Example:
        >>> from quat import QuatMatrix, rank
        >>> I = QuatMatrix.eye(4)
        >>> rank(I)
        4
        >>> Z = QuatMatrix.zeros(3, 3)
        >>> rank(Z)
        0
    """
    s = _svd_values(A)
    if tol is None:
        tol = max(A.shape) * s.max() * np.finfo(float).eps
    return int((s > tol).sum())


def condition_number(A: QuatMatrix) -> float:
    """Condition number σ_max / σ_min of a quaternion matrix.

    Args:
        A: QuatMatrix.

    Returns:
        float — ratio of largest to smallest positive singular value.

    Example:
        >>> from quat import QuatMatrix, condition_number
        >>> import numpy as np
        >>> I = QuatMatrix.eye(3)
        >>> np.testing.assert_almost_equal(condition_number(I), 1.0)
    """
    s = _svd_values(A)
    return float(s.max() / s[s > 1e-15].min())


def pseudo_inverse(A: QuatMatrix, tol: Optional[float] = None) -> QuatMatrix:
    """Moore-Penrose pseudo-inverse A⁺ = V * Σ⁺ * U^H.

    Σ⁺ contains 1/σ for singular values exceeding *tol*, and 0 otherwise.

    Args:
        A: QuatMatrix (m, n).
        tol: Cutoff for singular values.

    Returns:
        QuatMatrix (n, m).

    Example:
        >>> from quat import QuatMatrix, pseudo_inverse
        >>> import numpy as np
        >>> A = QuatMatrix(np.random.randn(3, 4, 4))
        >>> A_pinv = pseudo_inverse(A)
        >>> A_pinv.shape
        (4, 3)
        >>> # Verify M-P condition: A @ A⁺ @ A ≈ A
        >>> recon = A * A_pinv * A
        >>> np.allclose(A.to_array(), recon.to_array(), atol=1e-5)
        True
    """
    U, s, Vh = svd(A)
    if tol is None:
        tol = max(A.shape) * s.max() * np.finfo(float).eps
    m, n = A.shape
    k = len(s)
    S_data = np.zeros((n, m, 4))
    idx = np.arange(k)
    mask = s > tol
    S_data[idx[mask], idx[mask], 0] = 1.0 / s[mask]
    S_pinv = QuatMatrix(S_data)
    return Vh.H * S_pinv * U.H


def trace(A: QuatMatrix) -> Quaternion:
    """Quaternion trace — sum of diagonal elements.

    Args:
        A: Square QuatMatrix.

    Returns:
        Quaternion.

    Raises:
        ValueError: if *A* is not square.

    Example:
        >>> from quat import QuatMatrix, trace
        >>> I = QuatMatrix.eye(3)
        >>> tr = trace(I)
        >>> tr.r, tr.i, tr.j, tr.k
        (3.0, 0.0, 0.0, 0.0)
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square matrix, got {A.shape}")
    n = A.shape[0]
    diag = A._data[np.arange(n), np.arange(n)]
    return Quaternion(diag.sum(axis=0))


def det(A: QuatMatrix) -> float:
    """Study determinant of a quaternion matrix.

    Computed via the real (left-regular) representation: the 4n×4n real
    matrix has determinant ``(det(A))⁴``.  The Study determinant is the
    positive 4th root.

    Args:
        A: Square QuatMatrix.

    Returns:
        float.

    Raises:
        ValueError: if *A* is not square.

    Example:
        >>> from quat import QuatMatrix, det
        >>> I = QuatMatrix.eye(3)
        >>> abs(det(I) - 1.0) < 1e-5
        True
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square matrix, got {A.shape}")
    A_real = A.to_real_matrix_left()
    det_real = np.linalg.det(A_real)
    if det_real < 0:
        return float((-(-det_real) ** (1 / 4)))
    return float(det_real ** (1 / 4))


def norm(A: QuatMatrix, ord: str = 'fro') -> float:
    """Matrix norm of a quaternion matrix.

    Args:
        A: QuatMatrix.
        ord: ``'fro'`` (Frobenius) or ``2`` (spectral).

    Returns:
        float.

    Example:
        >>> from quat import QuatMatrix, norm
        >>> I = QuatMatrix.eye(3)
        >>> norm(I, 'fro')
        1.7320...
        >>> norm(I, 2)
        1.0
    """
    if ord == 'fro' or ord is None:
        return A.norm()
    elif ord == 2:
        s = _svd_values(A)
        return float(s.max())
    else:
        raise ValueError(f"Unsupported norm order: {ord}")


def solve(A: QuatMatrix, b: QuatVector) -> QuatVector:
    """Solve quaternion linear system A * x = b (least-squares).

    Uses the Moore-Penrose pseudo-inverse::

        x = A⁺ * b

    Args:
        A: QuatMatrix (m, n).
        b: QuatVector (m,).

    Returns:
        QuatVector (n,) — solution *x*.

    Example:
        >>> from quat import QuatMatrix, QuatVector, solve
        >>> A = QuatMatrix.eye(3)
        >>> b = QuatVector.zeros(3)       # actually all ones below
        >>> import numpy as np
        >>> b = QuatVector(np.ones((3, 4)))
        >>> x = solve(A, b)
        >>> np.allclose(x.to_array(), b.to_array())
        True
    """
    A_pinv = pseudo_inverse(A)
    return A_pinv * b
