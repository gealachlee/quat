"""Tests for quat.utils data conversion helpers."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestToNdarray(QuatTestCase):
    def test_quaternion_to_ndarray(self):
        from quat.utils import to_ndarray
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        arr = to_ndarray(q)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (4,))
        self.assertTrue(np.allclose(arr, [1., 2., 3., 4.]))

    def test_quatvector_to_ndarray(self):
        from quat.utils import to_ndarray
        from quat.collections import QuatVector
        v = QuatVector(np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]))
        arr = to_ndarray(v)
        self.assertEqual(arr.shape, (2, 4))

    def test_quatmatrix_to_ndarray(self):
        from quat.utils import to_ndarray
        from quat.collections import QuatMatrix
        M = QuatMatrix(np.ones((2, 3, 4)))
        arr = to_ndarray(M)
        self.assertEqual(arr.shape, (2, 3, 4))

    def test_quattensor_to_ndarray(self):
        from quat.utils import to_ndarray
        from quat.collections import QuatTensor
        T = QuatTensor(np.ones((2, 3, 4, 4)))
        arr = to_ndarray(T)
        self.assertEqual(arr.shape, (2, 3, 4, 4))


class TestFromNdarray(QuatTestCase):
    def test_quaternion(self):
        from quat.utils import from_ndarray
        from quat.core import Quaternion
        q = from_ndarray(np.array([1., 2., 3., 4.]))
        self.assertIsInstance(q, Quaternion)
        self.assertEqual(q.r, 1.0)

    def test_vector(self):
        from quat.utils import from_ndarray
        from quat.collections import QuatVector
        v = from_ndarray(np.random.randn(5, 4))
        self.assertIsInstance(v, QuatVector)

    def test_matrix(self):
        from quat.utils import from_ndarray
        from quat.collections import QuatMatrix
        M = from_ndarray(np.random.randn(3, 4, 4))
        self.assertIsInstance(M, QuatMatrix)

    def test_tensor(self):
        from quat.utils import from_ndarray
        from quat.collections import QuatTensor
        T = from_ndarray(np.random.randn(2, 3, 4, 4))
        self.assertIsInstance(T, QuatTensor)

    def test_scalar_to_quaternion(self):
        from quat.utils import from_ndarray
        from quat.core import Quaternion
        q = from_ndarray(np.array([5.0]))
        self.assertIsInstance(q, Quaternion)
        self.assertAlmostEqual(q.r, 5.0)


class TestFromComponents(QuatTestCase):
    def test_from_components(self):
        from quat.utils import from_components
        from quat.collections import QuatVector
        r = np.array([1., 2., 3.])
        i = np.array([4., 5., 6.])
        j = np.array([7., 8., 9.])
        k = np.array([0., 0., 0.])
        v = from_components(r, i, j, k)
        self.assertIsInstance(v, QuatVector)
        self.assertTrue(np.allclose(v.real, r))

    def test_matrix(self):
        from quat.utils import from_components
        from quat.collections import QuatMatrix
        r = np.ones((2, 3))
        i = np.ones((2, 3)) * 2
        j = np.ones((2, 3)) * 3
        k = np.ones((2, 3)) * 4
        M = from_components(r, i, j, k)
        self.assertIsInstance(M, QuatMatrix)
        self.assertEqual(M.shape, (2, 3))


class TestStackQuat(QuatTestCase):
    def test_stack_vectors(self):
        from quat.utils import stack_quat
        from quat.collections import QuatVector
        from quat.core import Quaternion
        q1 = Quaternion(1, 0, 0, 0)
        q2 = Quaternion(0, 1, 0, 0)
        q3 = Quaternion(0, 0, 1, 0)
        v = stack_quat([q1, q2, q3])
        self.assertIsInstance(v, QuatVector)
        self.assertEqual(v.shape, (3,))


class TestBroadcastHelpers(QuatTestCase):
    def test_broadcast_shapes(self):
        from quat.utils import broadcast_quat_shapes
        shapes = broadcast_quat_shapes((3, 1), (1, 4))
        self.assertEqual(shapes, (3, 4))

    def test_broadcast_shapes_identical(self):
        from quat.utils import broadcast_quat_shapes
        shapes = broadcast_quat_shapes((2, 3), (2, 3))
        self.assertEqual(shapes, (2, 3))


class TestNumericValidation(QuatTestCase):
    def test_isnan_quaternion_method(self):
        from quat.core import Quaternion
        self.assertFalse(Quaternion(1, 2, 3, 4).isnan())
        self.assertTrue(Quaternion(1, float('nan'), 3, 4).isnan())

    def test_isnan_quaternion_func(self):
        from quat.utils import isnan
        from quat.core import Quaternion
        self.assertTrue(isnan(Quaternion(1, float('nan'), 3, 4)))

    def test_isnan_vector(self):
        from quat.collections import QuatVector
        v = QuatVector(np.array([[1., 2., 3., 4.], [1., float('nan'), 3., 4.]]))
        result = v.isnan()
        self.assertFalse(result[0])
        self.assertTrue(result[1])

    def test_isinf_quaternion(self):
        from quat.core import Quaternion
        self.assertFalse(Quaternion(1, 2, 3, 4).isinf())
        self.assertTrue(Quaternion(1, float('inf'), 3, 4).isinf())

    def test_isinf_matrix(self):
        from quat.collections import QuatMatrix
        M = QuatMatrix(np.ones((2, 2, 4)))
        M._data[0, 1, 2] = float('inf')
        result = M.isinf()
        self.assertFalse(result[0, 0])
        self.assertTrue(result[0, 1])

    def test_isfinite_quaternion(self):
        from quat.core import Quaternion
        self.assertTrue(Quaternion(1, 2, 3, 4).isfinite())
        self.assertFalse(Quaternion(1, float('inf'), 3, 4).isfinite())

    def test_isfinite_tensor(self):
        from quat.collections import QuatTensor
        T = QuatTensor(np.ones((2, 3, 4, 4)))
        self.assertTrue(T.isfinite().all())

    def test_isclose_quaternion(self):
        from quat.core import Quaternion
        self.assertTrue(Quaternion(1, 2, 3, 4).isclose(Quaternion(1, 2, 3, 4)))
        self.assertFalse(Quaternion(1, 2, 3, 4).isclose(Quaternion(5, 6, 7, 8)))

    def test_isclose_vector(self):
        from quat.collections import QuatVector
        v1 = QuatVector(np.ones((3, 4)))
        v2 = QuatVector(np.ones((3, 4)))
        v2._data[1, 2] = 100.0
        result = v1.isclose(v2)
        self.assertTrue(result[0])
        self.assertFalse(result[1])
        self.assertTrue(result[2])

    def test_isclose_func(self):
        from quat.utils import isclose
        from quat.core import Quaternion
        self.assertTrue(isclose(Quaternion(1, 2, 3, 4), Quaternion(1, 2, 3, 4)))
