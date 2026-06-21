"""Tests for QuatVector, QuatMatrix, QuatTensor classes."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestQuatVectorConstruction(QuatTestCase):
    def test_from_ndarray(self):
        from quat.collections import QuatVector
        data = np.random.randn(5, 4)
        v = QuatVector(data)
        self.assertEqual(v.shape, (5,))
        self.assertEqual(len(v), 5)

    def test_from_list(self):
        from quat.collections import QuatVector
        from quat.core import Quaternion
        v = QuatVector([Quaternion(1, 0, 0, 0), Quaternion(0, 1, 0, 0)])
        self.assertEqual(len(v), 2)

    def test_zeros(self):
        from quat.collections import QuatVector
        v = QuatVector.zeros(3)
        self.assertEqual(len(v), 3)

    def test_from_real_matrix(self):
        from quat.collections import QuatVector
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        M = q.to_real_matrix()
        v = QuatVector.from_real_matrix(M)
        self.assertEqual(len(v), 1)


class TestQuatVectorOperations(QuatTestCase):
    def test_add(self):
        from quat.collections import QuatVector
        a = QuatVector(np.ones((3, 4)))
        b = QuatVector(np.ones((3, 4)) * 2)
        c = a + b
        self.assertTrue(np.allclose(c.to_array(), np.ones((3, 4)) * 3))

    def test_mul_scalar(self):
        from quat.collections import QuatVector
        v = QuatVector(np.ones((3, 4)))
        r = v * 3.0
        self.assertTrue(np.allclose(r.to_array(), np.ones((3, 4)) * 3))

    def test_mul_quaternion(self):
        from quat.collections import QuatVector
        from quat.core import Quaternion
        v = QuatVector(np.eye(4))
        q = Quaternion(2, 0, 0, 0)
        r = v * q
        self.assertTrue(np.allclose(r.to_array(), np.eye(4) * 2))

    def test_inner(self):
        from quat.collections import QuatVector
        a = QuatVector(np.eye(4)[:1])
        b = QuatVector(np.eye(4)[:1])
        result = a.inner(b)
        self.assertAlmostEqual(result.r, 1.0)

    def test_norm(self):
        from quat.collections import QuatVector
        v = QuatVector(np.array([[3., 4., 0., 0.]]))
        self.assertAlmostEqual(v.norm(), 5.0)

    def test_getitem_int(self):
        from quat.collections import QuatVector
        from quat.core import Quaternion
        v = QuatVector(np.eye(4))
        self.assertIsInstance(v[0], Quaternion)

    def test_components(self):
        from quat.collections import QuatVector
        v = QuatVector(np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]))
        self.assertTrue(np.allclose(v.real, [1., 5.]))
        self.assertTrue(np.allclose(v.i, [2., 6.]))


class TestQuatMatrixConstruction(QuatTestCase):
    def test_from_ndarray(self):
        from quat.collections import QuatMatrix
        data = np.random.randn(3, 4, 4)
        M = QuatMatrix(data)
        self.assertEqual(M.shape, (3, 4))

    def test_eye(self):
        from quat.collections import QuatMatrix
        I = QuatMatrix.eye(3)
        self.assertEqual(I.shape, (3, 3))

    def test_zeros(self):
        from quat.collections import QuatMatrix
        Z = QuatMatrix.zeros(2, 3)
        self.assertEqual(Z.shape, (2, 3))

    def test_from_real_matrix(self):
        from quat.collections import QuatMatrix
        data = np.random.randn(2, 2, 4)
        M = QuatMatrix(data)
        R = M.to_real_matrix()
        M2 = QuatMatrix.from_real_matrix(R)
        self.assertEqual(M.shape, M2.shape)
        self.assertTrue(np.allclose(M.to_array(), M2.to_array()))


class TestQuatMatrixOperations(QuatTestCase):
    def test_matmul_mat(self):
        from quat.collections import QuatMatrix
        A = QuatMatrix.eye(2)
        B = QuatMatrix(np.random.randn(2, 3, 4))
        C = A * B
        self.assertEqual(C.shape, (2, 3))

    def test_matmul_vec(self):
        from quat.collections import QuatMatrix, QuatVector
        M = QuatMatrix.eye(2)
        v = QuatVector(np.ones((2, 4)))
        r = M * v
        self.assertEqual(len(r), 2)

    def test_transpose(self):
        from quat.collections import QuatMatrix
        M = QuatMatrix(np.random.randn(3, 5, 4))
        MT = M.T
        self.assertEqual(MT.shape, (5, 3))

    def test_adjoint(self):
        from quat.collections import QuatMatrix
        M = QuatMatrix(np.random.randn(3, 5, 4))
        MH = M.H
        self.assertEqual(MH.shape, (5, 3))

    def test_row_col(self):
        from quat.collections import QuatMatrix
        M = QuatMatrix(np.random.randn(3, 4, 4))
        row = M.row(0)
        col = M.col(0)
        self.assertEqual(len(row), 4)
        self.assertEqual(len(col), 3)

    def test_getitem_tuple(self):
        from quat.collections import QuatMatrix
        from quat.core import Quaternion
        M = QuatMatrix(np.random.randn(3, 4, 4))
        self.assertIsInstance(M[0, 0], Quaternion)

    def test_components(self):
        from quat.collections import QuatMatrix
        M = QuatMatrix(np.ones((2, 3, 4)))
        self.assertEqual(M.real.shape, (2, 3))
        self.assertEqual(M.i.shape, (2, 3))


class TestQuatTensorConstruction(QuatTestCase):
    def test_from_ndarray(self):
        from quat.collections import QuatTensor
        data = np.random.randn(2, 3, 4, 4)
        T = QuatTensor(data)
        self.assertEqual(T.shape, (2, 3, 4))

    def test_zeros(self):
        from quat.collections import QuatTensor
        T = QuatTensor.zeros(2, 3, 4)
        self.assertEqual(T.shape, (2, 3, 4))


class TestQuatTensorOperations(QuatTestCase):
    def test_add(self):
        from quat.collections import QuatTensor
        a = QuatTensor(np.ones((2, 3, 4, 4)))
        b = QuatTensor(np.ones((2, 3, 4, 4)))
        c = a + b
        self.assertTrue(np.allclose(c.real, np.ones((2, 3, 4)) * 2))

    def test_inner(self):
        from quat.collections import QuatTensor
        a = QuatTensor(np.array([[[[1., 0., 0., 0.]]]]))
        result = a.inner(a)
        self.assertAlmostEqual(result.r, 1.0)

    def test_norm(self):
        from quat.collections import QuatTensor
        T = QuatTensor(np.array([[[[3., 4., 0., 0.]]]]))
        self.assertAlmostEqual(T.norm(), 5.0)

    def test_unfold_mode1(self):
        from quat.collections import QuatTensor
        T = QuatTensor(np.random.randn(3, 4, 5, 4))
        U = T.unfold(1)
        self.assertEqual(U.shape, (3, 20))

    def test_unfold_mode2(self):
        from quat.collections import QuatTensor
        T = QuatTensor(np.random.randn(3, 4, 5, 4))
        U = T.unfold(2)
        self.assertEqual(U.shape, (4, 15))

    def test_unfold_mode3(self):
        from quat.collections import QuatTensor
        T = QuatTensor(np.random.randn(3, 4, 5, 4))
        U = T.unfold(3)
        self.assertEqual(U.shape, (5, 12))

    def test_mode_product(self):
        from quat.collections import QuatTensor, QuatMatrix
        T = QuatTensor(np.random.randn(3, 4, 5, 4))
        A = QuatMatrix(np.random.randn(6, 3, 4))
        result = T.mode_1_product(A)
        self.assertEqual(result.shape, (6, 4, 5))

    def test_getitem_tuple3(self):
        from quat.collections import QuatTensor
        from quat.core import Quaternion
        T = QuatTensor(np.random.randn(2, 3, 4, 4))
        self.assertIsInstance(T[0, 0, 0], Quaternion)

    def test_getitem_int_slice(self):
        from quat.collections import QuatTensor, QuatMatrix, QuatVector
        T = QuatTensor(np.random.randn(3, 4, 5, 4))
        self.assertIsInstance(T[0], QuatMatrix)
        self.assertIsInstance(T[0, 0], QuatVector)


class TestModuleFunctions(QuatTestCase):
    def test_dict_to_quat_matrix(self):
        from quat.collections import dict_to_quat_matrix, QuatMatrix
        X_dict = {
            'real': np.ones((2, 3)),
            'i': np.ones((2, 3)) * 2,
            'j': np.ones((2, 3)) * 3,
            'k': np.ones((2, 3)) * 4,
        }
        M = dict_to_quat_matrix(X_dict)
        self.assertIsInstance(M, QuatMatrix)
        self.assertEqual(M.shape, (2, 3))

    def test_dict_to_quat_tensor(self):
        from quat.collections import dict_to_quat_tensor, QuatTensor
        X_dict = {
            'real': np.ones((5, 2, 3)),
            'i': np.ones((5, 2, 3)) * 2,
            'j': np.ones((5, 2, 3)) * 3,
            'k': np.ones((5, 2, 3)) * 4,
        }
        T = dict_to_quat_tensor(X_dict)
        self.assertIsInstance(T, QuatTensor)
        self.assertEqual(T.shape, (5, 2, 3))

    def test_labels_to_quat_vector(self):
        from quat.collections import labels_to_quat_vector, QuatVector
        y = np.array([0, 1, 0, 1])
        v = labels_to_quat_vector(y)
        self.assertIsInstance(v, QuatVector)
        self.assertEqual(v.shape, (4,))

    def test_labels_to_quat_vector_binary(self):
        from quat.collections import labels_to_quat_vector
        y = np.array([0, 1])
        v = labels_to_quat_vector(y, binary=True)
        self.assertTrue(np.allclose(v.real, [-1., 1.]))
