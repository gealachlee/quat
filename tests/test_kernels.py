"""Tests for quaternion kernel computations."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestCubicKernel(QuatTestCase):
    def test_shape(self):
        from quat.algorithms.kernels import cubic_kernel_matrix
        from quat.collections import QuatTensor
        X = QuatTensor(np.random.randn(5, 3, 4, 4))
        K = cubic_kernel_matrix(X)
        self.assertEqual(K.shape, (5, 5, 4))

    def test_identity_property(self):
        from quat.algorithms.kernels import cubic_kernel_matrix
        from quat.collections import QuatTensor
        X = QuatTensor(np.zeros((3, 1, 1, 4)))
        K = cubic_kernel_matrix(X)
        expected = np.zeros((3, 3, 4))
        expected[..., 0] = 1.0
        self.assertTrue(np.allclose(K, expected))


class TestGaussianKernel(QuatTestCase):
    def test_shape(self):
        from quat.algorithms.kernels import gaussian_kernel_matrix
        from quat.collections import QuatTensor
        X = QuatTensor(np.random.randn(4, 2, 2, 4))
        K = gaussian_kernel_matrix(X)
        self.assertEqual(K.shape, (4, 4, 4))

    def test_self_kernel(self):
        from quat.algorithms.kernels import gaussian_kernel_matrix
        from quat.collections import QuatTensor
        X = QuatTensor(np.zeros((2, 1, 1, 4)))
        K = gaussian_kernel_matrix(X)
        expected = np.zeros((2, 2, 4))
        expected[..., 0] = 1.0
        self.assertTrue(np.allclose(K, expected))


class TestNormalizeKernel(QuatTestCase):
    def test_range(self):
        from quat.algorithms.kernels import normalize_kernel
        K = np.random.randn(10, 10, 4) * 5
        K_norm = normalize_kernel(K)
        self.assertLessEqual(np.abs(K_norm).max(), 1.0)
