"""Tests for quat.kinematics — angular velocity, integration, frame rotation."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestAngularVelocity(QuatTestCase):
    def test_constant_rotation_z(self):
        from quat.core import Quaternion
        from quat.collections import QuatVector
        from quat.kinematics import angular_velocity
        dt = 0.01
        t = np.arange(100) * dt
        q_list = [Quaternion.from_axis_angle((0, 0, 1), ti) for ti in t]
        q_seq = QuatVector(q_list)
        omega = angular_velocity(q_seq, dt)
        self.assertEqual(omega.shape, (99, 3))
        omega_z = np.mean(omega[:, 2])
        self.assertAlmostEqual(omega_z, 1.0, places=1)

    def test_zero_velocity(self):
        from quat.core import Quaternion
        from quat.collections import QuatVector
        from quat.kinematics import angular_velocity
        q_fixed = Quaternion(1, 0, 0, 0)
        q_seq = QuatVector([q_fixed] * 10)
        omega = angular_velocity(q_seq, 0.01)
        self.assertTrue(np.allclose(omega, 0, atol=1e-10))


class TestIntegrateAngularVelocity(QuatTestCase):
    def test_roundtrip(self):
        from quat.core import Quaternion
        from quat.collections import QuatVector
        from quat.kinematics import angular_velocity, integrate_angular_velocity
        dt = 0.01
        t = np.arange(100) * dt
        q_list = [Quaternion.from_axis_angle((0, 0, 1), ti) for ti in t]
        q_seq = QuatVector(q_list)
        omega = angular_velocity(q_seq, dt)
        q_reint = integrate_angular_velocity(omega, dt, q0=q_list[0])
        self.assertEqual(len(q_reint), 100)
        diff = q_seq._data - q_reint._data
        self.assertLess(np.max(np.abs(diff)), 0.05)


class TestRotateFrame(QuatTestCase):
    def test_rotate_about_z(self):
        from quat.core import Quaternion
        from quat.kinematics import rotate_frame
        q = Quaternion(1, 0, 0, 0)
        result = rotate_frame(q, (0, 0, 1), np.pi / 2)
        expected = Quaternion.from_axis_angle((0, 0, 1), np.pi / 2)
        self.assertTrue(result.isclose(expected))
