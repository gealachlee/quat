"""Tests for quat.stats — quaternion_mean, quaternion_cov, quaternion_pca."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestQuaternionMean(QuatTestCase):
    def test_mean_of_identical(self):
        from quat.stats import quaternion_mean
        data = np.array([[1., 2., 3., 4.]] * 10)
        mu = quaternion_mean(data)
        self.assertTrue(np.allclose(mu._data, [1., 2., 3., 4.]))

    def test_mean_basic(self):
        from quat.stats import quaternion_mean
        data = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        mu = quaternion_mean(data)
        expected = np.array([0.5, 0.5, 0., 0.])
        self.assertTrue(np.allclose(mu._data, expected))


class TestQuaternionCov(QuatTestCase):
    def test_cov_shape(self):
        from quat.stats import quaternion_cov
        data = np.random.randn(100, 4)
        C = quaternion_cov(data)
        self.assertEqual(C.shape, (4, 4))

    def test_cov_positive_semidefinite(self):
        from quat.stats import quaternion_cov
        data = np.random.randn(50, 4)
        C = quaternion_cov(data)
        eigvals = np.linalg.eigvalsh(C)
        self.assertTrue(np.all(eigvals >= -1e-10))


class TestQuaternionPCA(QuatTestCase):
    def test_pca_shape(self):
        from quat.stats import quaternion_pca
        data = np.random.randn(100, 4)
        comp, ev = quaternion_pca(data, n_components=2)
        self.assertEqual(comp.shape, (2, 4))
        self.assertEqual(ev.shape, (2,))

    def test_pca_explained_variance_nonnegative(self):
        from quat.stats import quaternion_pca
        data = np.random.randn(100, 4)
        _, ev = quaternion_pca(data)
        self.assertTrue(np.all(ev >= 0))
