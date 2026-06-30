# =========================================================================
#  LLM-generated code.  See README.md for full disclosure.
# =========================================================================

"""Tests for quat.optimized — correctness vs standard implementations."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestOptimizedHamilton(QuatTestCase):
    def test_correctness_vs_standard(self):
        from quat.algebra import _hamilton
        from quat.algebra import hamilton_einsum
        p = np.random.randn(10, 4)
        q = np.random.randn(10, 4)
        r1 = _hamilton(p, q)
        r2 = hamilton_einsum(p, q)
        self.assertTrue(np.allclose(r1, r2))

    def test_broadcasting(self):
        from quat.algebra import _hamilton
        from quat.algebra import hamilton_einsum
        p = np.random.randn(3, 1, 4)
        q = np.random.randn(1, 5, 4)
        r1 = _hamilton(p, q)
        r2 = hamilton_einsum(p, q)
        self.assertTrue(np.allclose(r1, r2))


class TestOptimizedMatrixMultiply(QuatTestCase):
    def test_correctness(self):
        from quat.algebra import quat_matmul
        from quat.collections import QuatMatrix
        A = QuatMatrix(np.random.randn(3, 4, 4))
        B = QuatMatrix(np.random.randn(4, 5, 4))
        C1 = (A * B).to_array()
        C2 = quat_matmul(A._data, B._data)
        self.assertTrue(np.allclose(C1, C2))


class TestOptimizedOperations(QuatTestCase):
    def test_conjugate_batch(self):
        from quat.algebra import conjugate_batch
        data = np.random.randn(100, 4)
        expected = data * np.array([1., -1., -1., -1.])
        result = conjugate_batch(data)
        self.assertTrue(np.allclose(result, expected))

    def test_norm_squared_batch(self):
        from quat.algebra import norm_squared_batch
        data = np.random.randn(50, 4)
        result = norm_squared_batch(data)
        expected = (data * data).sum(axis=-1)
        self.assertTrue(np.allclose(result, expected))

    def test_normalize_batch(self):
        from quat.algebra import normalize_batch
        data = np.array([[3., 4., 0., 0.], [0., 0., 5., 12.]])
        result = normalize_batch(data)
        norms = np.sqrt((result * result).sum(axis=-1))
        self.assertTrue(np.allclose(norms, [1., 1.]))
