"""Tests for Quaternion class."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestQuaternionConstruction(QuatTestCase):
    def test_four_args(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertAlmostEqual(q.r, 1.0)
        self.assertAlmostEqual(q.i, 2.0)
        self.assertAlmostEqual(q.j, 3.0)
        self.assertAlmostEqual(q.k, 4.0)

    def test_scalar_only(self):
        from quat.core import Quaternion
        q = Quaternion(5.0)
        self.assertAlmostEqual(q.r, 5.0)
        self.assertAlmostEqual(q.i, 0.0)

    def test_copy_constructor(self):
        from quat.core import Quaternion
        q1 = Quaternion(1, 2, 3, 4)
        q2 = Quaternion(q1)
        self.assertEqual(q1, q2)
        self.assertIsNot(q1, q2)

    def test_from_ndarray(self):
        from quat.core import Quaternion
        q = Quaternion(np.array([1., 2., 3., 4.]))
        self.assertAlmostEqual(q.r, 1.0)

    def test_zero_factory(self):
        from quat.core import Quaternion
        q = Quaternion.zero()
        self.assertAlmostEqual(q.r, 0.0)

    def test_from_axis_angle(self):
        from quat.core import Quaternion
        q = Quaternion.from_axis_angle((0, 0, 1), np.pi / 2)
        expected = np.cos(np.pi / 4)
        self.assertAlmostEqual(q.r, expected, places=10)

    def test_invalid_ndarray_size(self):
        from quat.core import Quaternion
        with self.assertRaises(ValueError):
            Quaternion(np.array([1., 2., 3.]))


class TestQuaternionAccessors(QuatTestCase):
    def test_components(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertEqual(q.components, (1.0, 2.0, 3.0, 4.0))

    def test_scalar_vector(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertAlmostEqual(q.scalar, 1.0)
        self.assertTrue(np.allclose(q.vector, [2., 3., 4.]))

    def test_getitem(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertEqual(q[0], 1.0)
        self.assertEqual(q[3], 4.0)

    def test_iter(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertEqual(list(q), [1.0, 2.0, 3.0, 4.0])


class TestQuaternionArithmetic(QuatTestCase):
    def test_add(self):
        from quat.core import Quaternion
        a = Quaternion(1, 2, 3, 4)
        b = Quaternion(5, 6, 7, 8)
        c = a + b
        self.assertTrue(np.allclose(c._data, [6., 8., 10., 12.]))

    def test_add_scalar(self):
        from quat.core import Quaternion
        a = Quaternion(1, 2, 3, 4)
        c = a + 3.0
        self.assertTrue(np.allclose(c._data, [4., 2., 3., 4.]))

    def test_neg(self):
        from quat.core import Quaternion
        q = Quaternion(1, -2, 3, -4)
        n = -q
        self.assertTrue(np.allclose(n._data, [-1., 2., -3., 4.]))

    def test_mul_quaternion(self):
        from quat.core import Quaternion
        i = Quaternion(0, 1, 0, 0)
        j = Quaternion(0, 0, 1, 0)
        k = Quaternion(0, 0, 0, 1)
        self.assertEqual(i * i, Quaternion(-1, 0, 0, 0))
        self.assertEqual(i * j, k)

    def test_mul_scalar(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        r = q * 2.0
        self.assertTrue(np.allclose(r._data, [2., 4., 6., 8.]))

    def test_rmul_scalar(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        r = 3.0 * q
        self.assertTrue(np.allclose(r._data, [3., 6., 9., 12.]))

    def test_div_scalar(self):
        from quat.core import Quaternion
        q = Quaternion(2, 4, 6, 8)
        r = q / 2.0
        self.assertTrue(np.allclose(r._data, [1., 2., 3., 4.]))

    def test_div_quaternion(self):
        from quat.core import Quaternion
        a = Quaternion(2, 0, 0, 0)
        b = Quaternion(1, 0, 0, 0)
        self.assertEqual(a / b, Quaternion(2, 0, 0, 0))

    def test_commutator(self):
        from quat.core import Quaternion
        i = Quaternion(0, 1, 0, 0)
        j = Quaternion(0, 0, 1, 0)
        self.assertEqual(i.commutator(j), Quaternion(0, 0, 0, 2))


class TestQuaternionAlgebra(QuatTestCase):
    def test_conjugate(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        c = q.conjugate()
        self.assertTrue(np.allclose(c._data, [1., -2., -3., -4.]))

    def test_norm(self):
        from quat.core import Quaternion
        q = Quaternion(3, 4, 0, 0)
        self.assertAlmostEqual(q.norm(), 5.0)

    def test_norm_squared(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 2, 0)
        self.assertAlmostEqual(q.norm_squared(), 9.0)

    def test_normalize(self):
        from quat.core import Quaternion
        q = Quaternion(3, 4, 0, 0)
        n = q.normalize()
        self.assertAlmostEqual(n.norm(), 1.0)

    def test_normalize_zero_raises(self):
        from quat.core import Quaternion
        with self.assertRaises(ZeroDivisionError):
            Quaternion.zero().normalize()

    def test_inverse(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        inv = q.inverse()
        self.assertAlmostEqual((q * inv).r, 1.0, places=10)

    def test_inverse_zero_raises(self):
        from quat.core import Quaternion
        with self.assertRaises(ZeroDivisionError):
            Quaternion.zero().inverse()

    def test_exp_log_roundtrip(self):
        from quat.core import Quaternion
        q = Quaternion(0, 0.3, 0.4, 0)
        roundtrip = q.exp().log()
        self.assertAlmostEqual(q.r, roundtrip.r, places=10)

    def test_pow_integer(self):
        from quat.core import Quaternion
        q = Quaternion(1, 0, 0, 0)
        self.assertEqual(q.pow(3), Quaternion(1, 0, 0, 0))

    def test_dot(self):
        from quat.core import Quaternion
        a = Quaternion(1, 2, 3, 4)
        b = Quaternion(4, 3, 2, 1)
        self.assertAlmostEqual(a.dot(b), 20.0)


class TestQuaternionRotation(QuatTestCase):
    def test_rotate_vector(self):
        from quat.core import Quaternion
        q_rot = Quaternion.from_axis_angle((0, 0, 1), np.pi / 2)
        v = (1.0, 0.0, 0.0)
        vr = q_rot.rotate_vector(v)
        self.assertAlmostEqual(vr[0], 0.0, places=10)
        self.assertAlmostEqual(vr[1], 1.0, places=10)


class TestQuaternionMatrixRepresentations(QuatTestCase):
    def test_complex_roundtrip(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertEqual(q, Quaternion.from_complex_matrix(q.to_complex_matrix()))

    def test_real_matrix(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        M = q.to_real_matrix()
        self.assertEqual(M.shape, (4, 4))
        self.assertTrue(np.allclose(M.T @ M, q.norm_squared() * np.eye(4)))

    def test_real_roundtrip(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertEqual(q, Quaternion.from_real_matrix(q.to_real_matrix()))

    def test_to_array(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        arr = q.to_array()
        self.assertTrue(np.allclose(arr, [1., 2., 3., 4.]))


class TestQuaternionTypeConversions(QuatTestCase):
    def test_float(self):
        from quat.core import Quaternion
        self.assertEqual(float(Quaternion(3.5)), 3.5)

    def test_float_nonreal_raises(self):
        from quat.core import Quaternion
        with self.assertRaises(ValueError):
            float(Quaternion(1, 2, 3, 4))

    def test_int(self):
        from quat.core import Quaternion
        self.assertEqual(int(Quaternion(3)), 3)

    def test_complex(self):
        from quat.core import Quaternion
        c = complex(Quaternion(2, 3))
        self.assertEqual(c, 2 + 3j)

    def test_bool(self):
        from quat.core import Quaternion
        self.assertTrue(bool(Quaternion(1, 2, 3, 4)))
        self.assertFalse(bool(Quaternion(0, 0, 0, 0)))

    def test_abs(self):
        from quat.core import Quaternion
        q = Quaternion(3, 4, 0, 0)
        self.assertAlmostEqual(abs(q), 5.0)


class TestQuaternionHashEquality(QuatTestCase):
    def test_equal(self):
        from quat.core import Quaternion
        q1 = Quaternion(1, 2, 3, 4)
        q2 = Quaternion(1, 2, 3, 4)
        self.assertEqual(q1, q2)

    def test_not_equal(self):
        from quat.core import Quaternion
        self.assertNotEqual(Quaternion(1, 0, 0, 0), Quaternion(2, 0, 0, 0))

    def test_hash(self):
        from quat.core import Quaternion
        q1 = Quaternion(1.5, 2.5, 3.5, 4.5)
        q2 = Quaternion(1.5, 2.5, 3.5, 4.5)
        self.assertEqual(hash(q1), hash(q2))

    def test_representation(self):
        from quat.core import Quaternion
        q = Quaternion(1, -2, 3, -4)
        r = repr(q)
        self.assertIn("1", r)
        self.assertIn("-2", r)


class TestQuatFactory(QuatTestCase):
    def test_quat_function(self):
        from quat.core import quat, Quaternion
        q = quat(1, 2, 3, 4)
        self.assertIsInstance(q, Quaternion)
        self.assertEqual(q, Quaternion(1, 2, 3, 4))

    def test_basis_constants(self):
        from quat.core import _I, _J, _K, _ZERO, _R, _ONE, Quaternion
        self.assertEqual(_I, Quaternion(0, 1, 0, 0))
        self.assertEqual(_J, Quaternion(0, 0, 1, 0))
        self.assertEqual(_K, Quaternion(0, 0, 0, 1))
        self.assertEqual(_ZERO, Quaternion(0, 0, 0, 0))
        self.assertEqual(_R, Quaternion(1, 0, 0, 0))
        self.assertEqual(_ONE, Quaternion(1, 1, 1, 1))
