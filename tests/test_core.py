# =========================================================================
#  LLM-generated code.  See README.md for full disclosure.
# =========================================================================

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

    def test_real_imag(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertAlmostEqual(q.real, 1.0)
        self.assertTrue(np.allclose(q.imag, [2., 3., 4.]))

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
        self.assertAlmostEqual(a.re_inner(b), 20.0)


class TestQuaternionRotation(QuatTestCase):
    def test_rotate_vector(self):
        from quat.core import Quaternion
        q_rot = Quaternion.from_axis_angle((0, 0, 1), np.pi / 2)
        v = (1.0, 0.0, 0.0)
        vr = q_rot.rotate_vector(v)
        self.assertAlmostEqual(vr[0], 0.0, places=10)
        self.assertAlmostEqual(vr[1], 1.0, places=10)

    def test_axis_angle_roundtrip(self):
        from quat.core import Quaternion
        q = Quaternion.from_axis_angle((0, 0, 1), 1.2)
        axis, angle = q.to_axis_angle()
        self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=10)
        self.assertAlmostEqual(angle, 1.2, places=10)

    def test_axis_angle_identity(self):
        from quat.core import Quaternion
        q = Quaternion(1, 0, 0, 0)
        axis, angle = q.to_axis_angle()
        self.assertAlmostEqual(angle, 0.0, places=10)
        self.assertEqual(float(np.linalg.norm(axis)), 1.0)

    def test_axis_angle_x_axis(self):
        from quat.core import Quaternion
        q = Quaternion.from_axis_angle((1, 0, 0), 0.5)
        axis, angle = q.to_axis_angle()
        self.assertAlmostEqual(angle, 0.5, places=10)
        self.assertTrue(np.allclose(axis, [1., 0., 0.], atol=1e-10))


class TestQuaternionEuler(QuatTestCase):
    def test_zyx_roundtrip(self):
        from quat.core import Quaternion
        np.random.seed(42)
        for _ in range(10):
            phi = np.random.uniform(-np.pi, np.pi)
            theta = np.random.uniform(-np.pi / 2 + 0.05, np.pi / 2 - 0.05)
            psi = np.random.uniform(-np.pi, np.pi)
            angles = np.array([phi, theta, psi])
            q = Quaternion.from_euler(angles, 'zyx')
            out = q.to_euler('zyx')
            q2 = Quaternion.from_euler(out, 'zyx')
            self.assertTrue(q.isclose(q2), f'zyx failed: {angles} -> {out}')

    def test_xyz_roundtrip(self):
        from quat.core import Quaternion
        np.random.seed(99)
        for _ in range(10):
            angles = np.random.uniform(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, 3)
            q = Quaternion.from_euler(angles, 'xyz')
            out = q.to_euler('xyz')
            q2 = Quaternion.from_euler(out, 'xyz')
            self.assertTrue(q.isclose(q2))

    def test_zxz_roundtrip(self):
        from quat.core import Quaternion
        for _ in range(10):
            a = np.random.uniform(-np.pi, np.pi)
            b = np.random.uniform(0.1, np.pi - 0.1)
            c = np.random.uniform(-np.pi, np.pi)
            angles = np.array([a, b, c])
            q = Quaternion.from_euler(angles, 'zxz')
            out = q.to_euler('zxz')
            q2 = Quaternion.from_euler(out, 'zxz')
            self.assertTrue(q.isclose(q2))

    def test_xzx_roundtrip(self):
        from quat.core import Quaternion
        for _ in range(10):
            a = np.random.uniform(-np.pi, np.pi)
            b = np.random.uniform(0.1, np.pi - 0.1)
            c = np.random.uniform(-np.pi, np.pi)
            angles = np.array([a, b, c])
            q = Quaternion.from_euler(angles, 'xzx')
            out = q.to_euler('xzx')
            q2 = Quaternion.from_euler(out, 'xzx')
            self.assertTrue(q.isclose(q2))

    def test_zero_angles(self):
        from quat.core import Quaternion
        q = Quaternion.from_euler([0, 0, 0], 'zyx')
        self.assertEqual(q, Quaternion(1, 0, 0, 0))
        out = q.to_euler('zyx')
        self.assertTrue(np.allclose(out, [0, 0, 0]))

    def test_extrinsic_roundtrip(self):
        from quat.core import Quaternion
        angles = np.array([0.3, 0.5, 0.7])
        q = Quaternion.from_euler(angles, 'zyx', intrinsic=False)
        out = q.to_euler('zyx', intrinsic=False)
        q2 = Quaternion.from_euler(out, 'zyx', intrinsic=False)
        self.assertTrue(q.isclose(q2))

    def test_invalid_sequence(self):
        from quat.core import Quaternion
        with self.assertRaises(ValueError):
            Quaternion.from_euler([0, 0, 0], 'xwz')


class TestQuaternionNumPy(QuatTestCase):
    def test_np_add_preserves_type(self):
        from quat.core import Quaternion
        import numpy as np
        q1 = Quaternion(1, 2, 3, 4)
        q2 = Quaternion(5, 6, 7, 8)
        r = np.add(q1, q2)
        self.assertIsInstance(r, Quaternion)
        self.assertTrue(np.array_equal(r._data, [6., 8., 10., 12.]))

    def test_np_multiply_scalar(self):
        from quat.core import Quaternion
        import numpy as np
        q = Quaternion(1, 2, 3, 4)
        r = np.multiply(3.0, q)
        self.assertIsInstance(r, Quaternion)
        self.assertTrue(np.array_equal(r._data, [3., 6., 9., 12.]))

    def test_np_multiply_quat(self):
        from quat.core import Quaternion
        import numpy as np
        i = Quaternion(0, 1, 0, 0)
        j = Quaternion(0, 0, 1, 0)
        r = np.multiply(i, j)
        self.assertIsInstance(r, Quaternion)
        self.assertTrue(np.array_equal(r._data, Quaternion(0, 0, 0, 1)._data))

    def test_np_negative(self):
        from quat.core import Quaternion
        import numpy as np
        q = Quaternion(1, -2, 3, -4)
        r = np.negative(q)
        self.assertIsInstance(r, Quaternion)
        self.assertTrue(np.array_equal(r._data, [-1., 2., -3., 4.]))

    def test_np_absolute(self):
        from quat.core import Quaternion
        import numpy as np
        q = Quaternion(3, 4, 0, 0)
        r = np.absolute(q)
        self.assertAlmostEqual(r, 5.0)

    def test_np_conjugate(self):
        from quat.core import Quaternion
        import numpy as np
        q = Quaternion(1, 2, 3, 4)
        r = np.conjugate(q)
        self.assertIsInstance(r, Quaternion)
        self.assertTrue(np.array_equal(r._data, [1., -2., -3., -4.]))

    def test_np_negative_positive(self):
        from quat.core import Quaternion
        import numpy as np
        q = Quaternion(1, 2, 3, 4)
        self.assertTrue(np.array_equal(np.positive(q)._data, q._data))

    def test_np_array_creation(self):
        from quat.core import Quaternion
        import numpy as np
        q = Quaternion(1, 2, 3, 4)
        arr = np.array(q)
        self.assertEqual(arr.shape, (4,))
        self.assertTrue(np.array_equal(arr, [1., 2., 3., 4.]))


class TestQuaternionMatrixRepresentations(QuatTestCase):
    def test_complex_roundtrip(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertEqual(q, Quaternion.from_complex_matrix(q.to_complex_matrix()))

    def test_real_matrix(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        M = q.to_real_matrix_left()
        self.assertEqual(M.shape, (4, 4))
        self.assertTrue(np.allclose(M.T @ M, q.norm_squared() * np.eye(4)))

    def test_real_roundtrip(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertEqual(q, Quaternion.from_real_matrix_left(q.to_real_matrix_left()))

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
        from quat.core import _I, _J, _K, _ZERO, _R, _ONE_Q, Quaternion
        self.assertEqual(_I, Quaternion(0, 1, 0, 0))
        self.assertEqual(_J, Quaternion(0, 0, 1, 0))
        self.assertEqual(_K, Quaternion(0, 0, 0, 1))
        self.assertEqual(_ZERO, Quaternion(0, 0, 0, 0))
        self.assertEqual(_R, Quaternion(1, 0, 0, 0))
        self.assertEqual(_ONE_Q, Quaternion(1, 1, 1, 1))


class TestQuaternionToNumpy(QuatTestCase):
    def test_to_numpy_default_copy(self):
        from quat.core import Quaternion
        import numpy as np
        q = Quaternion(1, 2, 3, 4)
        arr = q.to_numpy()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (4,))
        self.assertTrue(np.allclose(arr, [1., 2., 3., 4.]))
        arr[0] = 99.0
        self.assertNotEqual(q.r, 99.0)

    def test_to_numpy_no_copy(self):
        from quat.core import Quaternion
        import numpy as np
        q = Quaternion(1, 2, 3, 4)
        arr = q.to_numpy(copy=False)
        self.assertTrue(arr is q._data)
        arr[0] = 99.0
        self.assertEqual(q.r, 99.0)

    def test_to_numpy_dtype(self):
        from quat.core import Quaternion
        import numpy as np
        q = Quaternion(1, 2, 3, 4)
        arr = q.to_numpy(dtype=np.float32)
        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(np.allclose(arr, [1., 2., 3., 4.]))
