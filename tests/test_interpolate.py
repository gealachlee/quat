"""Tests for quat.interpolate — SLERP and squad."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestSlerp(QuatTestCase):
    def test_slerp_endpoints(self):
        from quat.interpolate import slerp
        from quat.core import Quaternion
        q0 = Quaternion(1, 0, 0, 0)
        q1 = Quaternion(0, 1, 0, 0)
        r0 = slerp(q0, q1, 0.0)
        r1 = slerp(q0, q1, 1.0)
        self.assertTrue(r0.isclose(q0))
        self.assertTrue(r1.isclose(q1))

    def test_slerp_midpoint(self):
        from quat.interpolate import slerp
        from quat.core import Quaternion
        q0 = Quaternion(1, 0, 0, 0)
        q1 = Quaternion(0, 1, 0, 0)
        mid = slerp(q0, q1, 0.5)
        self.assertAlmostEqual(mid.norm(), 1.0, places=10)
        self.assertAlmostEqual(float(mid.r), np.sqrt(2) / 2, places=8)
        self.assertAlmostEqual(float(mid.i), np.sqrt(2) / 2, places=8)

    def test_slerp_shortest_arc(self):
        from quat.interpolate import slerp
        from quat.core import Quaternion
        q0 = Quaternion.from_axis_angle((0, 0, 1), 0)
        q1 = Quaternion.from_axis_angle((0, 0, 1), np.pi)
        mid = slerp(q0, q1, 0.3)
        self.assertAlmostEqual(mid.norm(), 1.0, places=8)

    def test_slerp_near_identical(self):
        from quat.interpolate import slerp
        from quat.core import Quaternion
        q0 = Quaternion(1, 0, 0, 0)
        q1 = Quaternion(0.999999, 0.001, 0, 0)
        r = slerp(q0, q1, 0.5)
        self.assertAlmostEqual(r.norm(), 1.0, places=6)

    def test_slerp_constant_norm(self):
        from quat.interpolate import slerp
        from quat.random import random_unit_quat
        np.random.seed(42)
        for _ in range(10):
            q0 = random_unit_quat()
            q1 = random_unit_quat()
            t = np.random.uniform(0, 1)
            r = slerp(q0, q1, t)
            self.assertAlmostEqual(r.norm(), 1.0, places=8)


class TestSquad(QuatTestCase):
    def test_squad_endpoints(self):
        from quat.interpolate import squad
        from quat.core import Quaternion, _quat_to_rotmat
        from quat.random import random_unit_quat
        np.random.seed(1)
        q0 = random_unit_quat()
        q1 = random_unit_quat()
        q2 = random_unit_quat()
        q3 = random_unit_quat()
        r0 = squad(q0, q1, q2, q3, 0.0)
        r1 = squad(q0, q1, q2, q3, 1.0)
        self.assertAlmostEqual(r0.norm(), 1.0, places=8)
        self.assertAlmostEqual(r1.norm(), 1.0, places=8)
        self.assertTrue(np.allclose(_quat_to_rotmat(r0._data),
                                    _quat_to_rotmat(q1._data)))
        self.assertTrue(np.allclose(_quat_to_rotmat(r1._data),
                                    _quat_to_rotmat(q2._data)))

    def test_squad_constant_norm(self):
        from quat.interpolate import squad
        from quat.random import random_unit_quat
        np.random.seed(7)
        for _ in range(10):
            q0 = random_unit_quat()
            q1 = random_unit_quat()
            q2 = random_unit_quat()
            q3 = random_unit_quat()
            for t in [0, 0.25, 0.5, 0.75, 1.0]:
                r = squad(q0, q1, q2, q3, t)
                self.assertAlmostEqual(r.norm(), 1.0, places=6,
                    msg=f'squad norm failed at t={t}')

    def test_squad_straight_line(self):
        from quat.interpolate import slerp, squad
        from quat.core import Quaternion
        q1 = Quaternion(1, 0, 0, 0)
        q2 = Quaternion(0, 1, 0, 0)
        q0 = Quaternion(0.92388, 0.38268, 0, 0)
        q3 = Quaternion(-0.38268, 0.92388, 0, 0)
        for t in [0.2, 0.5, 0.8]:
            s = squad(q0, q1, q2, q3, t)
            self.assertAlmostEqual(s.norm(), 1.0, places=6)
