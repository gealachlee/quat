"""Tests for quat.linalg — SVD, rank, condition number, etc."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestSVD(QuatTestCase):
    def test_svd_reconstruction(self):
        from quat.linalg import svd
        from quat.collections import QuatMatrix
        A = QuatMatrix(np.random.randn(4, 5, 4))
        U, s, Vh = svd(A)
        self.assertEqual(U.shape, (4, 4))
        self.assertEqual(len(s), min(4, 5))
        self.assertEqual(Vh.shape, (5, 5))
        S = QuatMatrix.zeros(4, 5)
        for i in range(len(s)):
            S._data[i, i, 0] = s[i]
        recon = U * S * Vh
        self.assertTrue(np.allclose(A.to_array(), recon.to_array(), atol=1e-6))

    def test_svd_square(self):
        from quat.linalg import svd
        from quat.collections import QuatMatrix
        A = QuatMatrix(np.random.randn(3, 3, 4))
        U, s, Vh = svd(A)
        self.assertEqual(U.shape, (3, 3))
        self.assertEqual(Vh.shape, (3, 3))
        self.assertEqual(len(s), 3)
        self.assertTrue((s >= 0).all())

    def test_svd_singular_values_nonnegative(self):
        from quat.linalg import svd
        from quat.collections import QuatMatrix
        A = QuatMatrix(np.random.randn(5, 3, 4))
        _, s, _ = svd(A)
        self.assertTrue((s >= -1e-10).all())


class TestRank(QuatTestCase):
    def test_full_rank(self):
        from quat.linalg import rank
        from quat.collections import QuatMatrix
        A = QuatMatrix.eye(4)
        self.assertEqual(rank(A), 4)

    def test_rank_deficient(self):
        from quat.linalg import rank
        from quat.collections import QuatMatrix
        A = QuatMatrix(np.zeros((3, 3, 4)))
        self.assertEqual(rank(A), 0)


class TestConditionNumber(QuatTestCase):
    def test_identity(self):
        from quat.linalg import condition_number
        from quat.collections import QuatMatrix
        A = QuatMatrix.eye(3)
        self.assertAlmostEqual(condition_number(A), 1.0, places=5)


class TestPseudoInverse(QuatTestCase):
    def test_moore_penrose_conditions(self):
        from quat.linalg import pseudo_inverse
        from quat.collections import QuatMatrix
        A = QuatMatrix(np.random.randn(3, 4, 4))
        A_pinv = pseudo_inverse(A)
        self.assertEqual(A_pinv.shape, (4, 3))
        # A * A_pinv * A ≈ A
        recon = A * A_pinv * A
        self.assertTrue(np.allclose(A.to_array(), recon.to_array(), atol=1e-5))


class TestTrace(QuatTestCase):
    def test_trace_identity(self):
        from quat.linalg import trace
        from quat.collections import QuatMatrix
        from quat.core import Quaternion
        I = QuatMatrix.eye(3)
        tr = trace(I)
        self.assertTrue(isinstance(tr, Quaternion))
        self.assertAlmostEqual(tr.r, 3.0)
        self.assertAlmostEqual(tr.i, 0.0)

    def test_trace_non_square_raises(self):
        from quat.linalg import trace
        from quat.collections import QuatMatrix
        A = QuatMatrix(np.random.randn(3, 4, 4))
        with self.assertRaises(ValueError):
            trace(A)


class TestDeterminant(QuatTestCase):
    def test_det_identity(self):
        from quat.linalg import det
        from quat.collections import QuatMatrix
        I = QuatMatrix.eye(3)
        self.assertAlmostEqual(det(I), 1.0, places=5)

    def test_det_non_square_raises(self):
        from quat.linalg import det
        from quat.collections import QuatMatrix
        A = QuatMatrix(np.random.randn(3, 4, 4))
        with self.assertRaises(ValueError):
            det(A)


class TestNorm(QuatTestCase):
    def test_norm_fro(self):
        from quat.linalg import norm
        from quat.collections import QuatMatrix
        A = QuatMatrix(np.ones((3, 3, 4)))
        n = norm(A, 'fro')
        self.assertGreater(n, 0)

    def test_norm_spectral(self):
        from quat.linalg import norm
        from quat.collections import QuatMatrix
        A = QuatMatrix.eye(3)
        n = norm(A, 2)
        self.assertAlmostEqual(n, 1.0, places=5)


class TestSolve(QuatTestCase):
    def test_solve_identity(self):
        from quat.linalg import solve
        from quat.collections import QuatMatrix, QuatVector
        A = QuatMatrix.eye(3)
        b = QuatVector(np.ones((3, 4)))
        x = solve(A, b)
        self.assertEqual(x.shape, (3,))
        self.assertTrue(np.allclose(x.to_array(), b.to_array(), atol=1e-5))
