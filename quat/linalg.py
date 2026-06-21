"""Quaternion linear algebra — SVD, rank, condition number, pseudo-inverse."""
import numpy as np
from quat.core import Quaternion
from quat.collections import QuatVector, QuatMatrix


def svd(A):
    """Quaternion singular value decomposition.

    Decompose a quaternion matrix A (m×n) into:
        A = U * Σ * V^H
    where U is m×m unitary quaternion matrix, Σ is m×n real diagonal
    (singular values), and V^H is n×n unitary quaternion matrix.

    Uses the real representation: builds the 4m×4n real matrix of A,
    computes its SVD, then reconstructs quaternion U and V^H from the
    structured real singular vectors.

    Args:
        A: QuatMatrix of shape (m, n)

    Returns:
        (U, s, Vh) where:
            U:  QuatMatrix (m, m) — left singular vectors
            s:  ndarray (k,) where k = min(m, n) — singular values
            Vh: QuatMatrix (n, n) — right singular vectors (conjugate-transposed)
    """
    A_real = A.to_real_matrix_left()  # (4m, 4n)
    U_real, s_full, Vt_real = np.linalg.svd(A_real, full_matrices=True)
    # Singular values come in groups of 4; take every 4th unique value
    k = min(A.shape[0], A.shape[1])
    s = s_full[::4][:k]

    # Reconstruct quaternion U: U_real is (4m, 4m) → U is (m, m)
    U_data = np.empty((A.shape[0], A.shape[0], 4))
    for i in range(0, U_real.shape[1], 4):
        j = i // 4
        block = U_real[:, i:i+4]  # (4m, 4)
        for r in range(A.shape[0]):
            blk = block[4*r:4*r+4, :]
            U_data[r, j] = Quaternion.from_real_matrix_left(blk)._data
    U = QuatMatrix(U_data)

    # Reconstruct quaternion V: Vt_real is (4n, 4n) → V is (n, n)
    n_A = A.shape[1]
    V_data = np.empty((n_A, n_A, 4))
    for i in range(0, Vt_real.shape[0], 4):
        j = i // 4
        block = Vt_real[i:i+4, :]  # (4, 4n)
        for c in range(n_A):
            blk = block[:, 4*c:4*c+4].T  # transpose to left-repr
            V_data[c, j] = Quaternion.from_real_matrix_left(blk)._data
    V = QuatMatrix(V_data)

    return U, s, V.H


def rank(A, tol=None):
    """Compute the quaternion matrix rank via SVD.

    Args:
        A: QuatMatrix
        tol: tolerance for singular values (default: max(m,n) * max(s) * eps)

    Returns:
        int
    """
    _, s, _ = svd(A)
    if tol is None:
        tol = max(A.shape) * s.max() * np.finfo(float).eps
    return int((s > tol).sum())


def condition_number(A):
    """Compute the condition number (σ_max / σ_min) of a quaternion matrix.

    Args:
        A: QuatMatrix

    Returns:
        float
    """
    _, s, _ = svd(A)
    return float(s.max() / s[s > 1e-15].min())


def pseudo_inverse(A, tol=None):
    """Moore-Penrose pseudo-inverse of a quaternion matrix.

    A⁺ = V * Σ⁺ * U^H, where Σ⁺ contains 1/σ for σ > tol.

    Args:
        A: QuatMatrix (m, n)
        tol: singular value cutoff

    Returns:
        QuatMatrix (n, m)
    """
    U, s, Vh = svd(A)
    if tol is None:
        tol = max(A.shape) * s.max() * np.finfo(float).eps
    m, n = A.shape
    k = len(s)
    S_data = np.zeros((n, m, 4))
    for i in range(k):
        if s[i] > tol:
            S_data[i, i, 0] = 1.0 / s[i]
    S_pinv = QuatMatrix(S_data)
    return Vh.H * S_pinv * U.H


def trace(A):
    """Quaternion trace: sum of diagonal elements.

    Args:
        A: QuatMatrix (square)

    Returns:
        Quaternion
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square matrix, got {A.shape}")
    result = Quaternion.zero()
    for i in range(A.shape[0]):
        result = result + A[i, i]
    return result


def det(A):
    """Determinant of a quaternion matrix via its real representation.

    For a quaternion matrix A of size n×n, its real representation is
    4n×4n. The determinant of the real representation is (det(A))⁴ for
    a suitably defined quaternion determinant (Study determinant).

    Args:
        A: QuatMatrix (square)

    Returns:
        float — Study determinant
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square matrix, got {A.shape}")
    A_real = A.to_real_matrix_left()
    det_real = np.linalg.det(A_real)
    if det_real < 0:
        return float((-(-det_real) ** (1/4)))
    return float(det_real ** (1/4))


def norm(A, ord='fro'):
    """Matrix norm of a quaternion matrix.

    Args:
        A: QuatMatrix
        ord: 'fro' (Frobenius) or 2 (spectral)

    Returns:
        float
    """
    if ord == 'fro' or ord is None:
        return A.norm()
    elif ord == 2:
        _, s, _ = svd(A)
        return float(s.max())
    else:
        raise ValueError(f"Unsupported norm order: {ord}")


def solve(A, b):
    """Solve quaternion linear system A * x = b.

    Uses pseudo-inverse for the least-squares solution.

    Args:
        A: QuatMatrix (m, n)
        b: QuatVector (m,)

    Returns:
        QuatVector (n,) — solution x
    """
    A_pinv = pseudo_inverse(A)
    return A_pinv * b
