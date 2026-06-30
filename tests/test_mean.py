"""Tests for quat.mean — mean_rotation, karcher_mean."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestMeanRotation(QuatTestCase):
    def test_identical_quaternions(self):
        from quat.core import Quaternion
        from quat.collections import QuatVector
        from quat.stats import mean_rotation
        q = Quaternion(1, 0, 0, 0)
        qv = QuatVector([q, q, q])
        m = mean_rotation(qv)
        self.assertTrue(m.isclose(q))

    def test_antipodal_mean_unit_norm(self):
        from quat.core import Quaternion
        from quat.collections import QuatVector
        from quat.stats import mean_rotation
        qv = QuatVector([
            Quaternion(1, 0, 0, 0),
            Quaternion(-1, 0, 0, 0),
        ])
        m = mean_rotation(qv)
        self.assertAlmostEqual(m.norm(), 1.0)

    def test_weighted_mean(self):
        from quat.core import Quaternion
        from quat.collections import QuatVector
        from quat.stats import mean_rotation
        qv = QuatVector([
            Quaternion(1, 0, 0, 0),
            Quaternion(0, 1, 0, 0),
        ])
        m = mean_rotation(qv, weights=np.array([1.0, 0.0]))
        self.assertTrue(m.isclose(Quaternion(1, 0, 0, 0)))

    def test_valid_unit_output(self):
        from quat.core import Quaternion
        from quat.collections import QuatVector
        from quat.stats import mean_rotation
        rng = np.random.default_rng(42)
        data = rng.normal(size=(50, 4))
        data = data / np.linalg.norm(data, axis=-1, keepdims=True)
        qv = QuatVector(data)
        m = mean_rotation(qv)
        self.assertAlmostEqual(m.norm(), 1.0, places=5)


class TestKarcherMean(QuatTestCase):
    def test_converges_to_unit(self):
        from quat.collections import QuatVector
        from quat.stats import karcher_mean
        rng = np.random.default_rng(42)
        data = rng.normal(size=(100, 4))
        qv = QuatVector(data / np.linalg.norm(data, axis=-1, keepdims=True))
        km = karcher_mean(qv, max_iter=50)
        self.assertAlmostEqual(km.norm(), 1.0, places=5)
