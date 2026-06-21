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


class TestBatchQuat(QuatTestCase):
    def test_batch_from_components(self):
        from quat.utils import batch_quat
        from quat.collections import QuatVector
        r = np.array([1., 2., 3.])
        i = np.array([4., 5., 6.])
        j = np.array([7., 8., 9.])
        k = np.array([0., 0., 0.])
        v = batch_quat(r, i, j, k)
        self.assertIsInstance(v, QuatVector)
        self.assertTrue(np.allclose(v.real, r))

    def test_batch_matrix(self):
        from quat.utils import batch_quat
        from quat.collections import QuatMatrix
        r = np.ones((2, 3))
        i = np.ones((2, 3)) * 2
        j = np.ones((2, 3)) * 3
        k = np.ones((2, 3)) * 4
        M = batch_quat(r, i, j, k)
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
