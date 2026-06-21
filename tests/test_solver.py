"""Tests for Q-ADMM solver."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestProxRamp(QuatTestCase):
    def test_prox_ramp_basic(self):
        from quat.algorithms.solver import _prox_ramp
        t = np.array([[2.0, 0.5, -0.1, 3.0]])
        beta = np.array([0.5, 0.5, 0.5, 0.5])
        result = _prox_ramp(t, beta)
        self.assertEqual(result.shape, (1, 4))


class TestPredict(QuatTestCase):
    def test_predict_shape(self):
        from quat.algorithms.solver import predict_ksqmm
        K_test = np.ones((3, 5, 4))
        alpha = np.ones((5, 4))
        b = np.zeros(4)
        y_pred = predict_ksqmm(K_test, alpha, b)
        self.assertEqual(y_pred.shape, (3, 4))


class TestSolveKSQMM(QuatTestCase):
    def test_solve_smoke(self):
        from quat.algorithms.solver import solve_ksqmm
        from quat.algorithms.kernels import cubic_kernel_matrix
        from quat.collections import QuatTensor
        N = 10
        X = QuatTensor(np.random.randn(N, 1, 1, 4))
        K = cubic_kernel_matrix(X)
        y = np.ones((N, 4))
        y[5:, 0] = -1
        alpha, b, info = solve_ksqmm(K, y, max_iter=50, tol=1e-3, verbose=False)
        self.assertEqual(alpha.shape, (N, 4))
        self.assertEqual(b.shape, (4,))
        self.assertIn('n_iter', info)
