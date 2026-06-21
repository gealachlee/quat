"""Tests for quat.algebra module."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestHamiltonProduct(QuatTestCase):
    def test_two_vectors(self):
        from quat.algebra import _hamilton
        p = np.array([[1., 2., 3., 4.]])
        q = np.array([[5., 6., 7., 8.]])
        r = _hamilton(p, q)
        self.assertEqual(r.shape, (1, 4))
        expected = np.array([[-60., 12., 30., 24.]])
        self.assertTrue(np.allclose(r, expected))

    def test_batch_square(self):
        from quat.algebra import _hamilton
        p = np.random.randn(5, 4)
        q = np.random.randn(1, 4)
        r = _hamilton(p, q)
        self.assertEqual(r.shape, (5, 4))

    def test_broadcasting(self):
        from quat.algebra import _hamilton
        p = np.random.randn(3, 1, 4)
        q = np.random.randn(1, 5, 4)
        r = _hamilton(p, q)
        self.assertEqual(r.shape, (3, 5, 4))

    def test_identity(self):
        from quat.algebra import _hamilton
        one = np.array([1., 0., 0., 0.])
        q = np.random.randn(10, 4)
        r1 = _hamilton(one[None, :], q)
        r2 = _hamilton(q, one[None, :])
        self.assertTrue(np.allclose(r1, q))
        self.assertTrue(np.allclose(r2, q))

    def test_basis_multiplication(self):
        from quat.algebra import _hamilton
        i = np.array([0., 1., 0., 0.])
        j = np.array([0., 0., 1., 0.])
        k = np.array([0., 0., 0., 1.])
        one = np.array([1., 0., 0., 0.])
        self.assertTrue(np.allclose(_hamilton(i, i), -one))
        self.assertTrue(np.allclose(_hamilton(i, j), k))
        self.assertTrue(np.allclose(_hamilton(j, k), i))
        self.assertTrue(np.allclose(_hamilton(k, i), j))
        self.assertTrue(np.allclose(_hamilton(i, j), -_hamilton(j, i)))


class TestConstants(QuatTestCase):
    def test_conj_mask(self):
        from quat.algebra import _CONJ
        self.assertTrue(np.allclose(_CONJ, [1., -1., -1., -1.]))

    def test_real_left_tensor(self):
        from quat.algebra import _REAL_LEFT
        self.assertEqual(_REAL_LEFT.shape, (4, 4, 4))
        q = np.array([1., 2., 3., 4.])
        R = np.einsum('rck,k->rc', _REAL_LEFT, q)
        expected = np.array([
            [1., -2., -3., -4.],
            [2.,  1., -4.,  3.],
            [3.,  4.,  1., -2.],
            [4., -3.,  2.,  1.]
        ])
        self.assertTrue(np.allclose(R, expected))
