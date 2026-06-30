"""Tests for quat.distance — rotation/rotor intrinsic/chordal distance metrics."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestDistanceRotationIntrinsic(QuatTestCase):
    def test_same_quaternion_zero_distance(self):
        from quat.core import Quaternion
        from quat.stats import rotation
        q = Quaternion(1, 0, 0, 0)
        d = rotation.intrinsic(q, q)
        self.assertAlmostEqual(d, 0.0)

    def test_opposite_quaternion_zero_distance(self):
        from quat.core import Quaternion
        from quat.stats import rotation
        q = Quaternion(1, 0, 0, 0)
        d = rotation.intrinsic(q, -q)
        self.assertAlmostEqual(d, 0.0)

    def test_orthogonal_quaternion(self):
        from quat.core import Quaternion
        from quat.stats import rotation
        q1 = Quaternion(1, 0, 0, 0)
        q2 = Quaternion(0, 1, 0, 0)
        d = rotation.intrinsic(q1, q2)
        self.assertAlmostEqual(d, np.pi, places=5)

    def test_batch_vs_scalar(self):
        from quat.core import Quaternion
        from quat.collections import QuatVector
        from quat.stats import rotation
        q1 = Quaternion(1, 0, 0, 0)
        q2 = Quaternion(0, 1, 0, 0)
        v1 = QuatVector([q1, q2])
        v2 = QuatVector([q2, q1])
        result = rotation.intrinsic_batch(v1, v2)
        self.assertEqual(result.shape, (2,))
        scalar_0 = rotation.intrinsic(Quaternion(v1._data[0]), Quaternion(v2._data[0]))
        self.assertAlmostEqual(result[0], scalar_0)


class TestDistanceRotorIntrinsic(QuatTestCase):
    def test_same_quaternion_zero(self):
        from quat.core import Quaternion
        from quat.stats import rotor
        q = Quaternion(1, 0, 0, 0)
        self.assertAlmostEqual(rotor.intrinsic(q, q), 0.0)

    def test_opposite_not_zero(self):
        from quat.core import Quaternion
        from quat.stats import rotor
        q = Quaternion(1, 0, 0, 0)
        d = rotor.intrinsic(q, -q)
        self.assertGreater(d, 0.0)


class TestDistanceRotationChordal(QuatTestCase):
    def test_same_quaternion_zero(self):
        from quat.core import Quaternion
        from quat.stats import rotation
        q = Quaternion(1, 0, 0, 0)
        self.assertAlmostEqual(rotation.chordal(q, q), 0.0)

    def test_opposite_zero(self):
        from quat.core import Quaternion
        from quat.stats import rotation
        q = Quaternion(1, 0, 0, 0)
        self.assertAlmostEqual(rotation.chordal(q, -q), 0.0)


class TestDistanceRotorChordal(QuatTestCase):
    def test_identity_properties(self):
        from quat.core import Quaternion
        from quat.stats import rotor
        q1 = Quaternion(1, 0, 0, 0)
        q2 = Quaternion(0, 1, 0, 0)
        d12 = rotor.chordal(q1, q2)
        d21 = rotor.chordal(q2, q1)
        self.assertAlmostEqual(d12, d21)

    def test_self_distance_zero(self):
        from quat.core import Quaternion
        from quat.stats import rotor
        q = Quaternion(1, 0, 0, 0)
        self.assertAlmostEqual(rotor.chordal(q, q), 0.0)
