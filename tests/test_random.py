"""Tests for quat.random — random quaternion generators."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestRandomQuat(QuatTestCase):
    def test_random_quat(self):
        from quat.random import random_quat
        q = random_quat(42)
        self.assertGreater(q.norm(), 0)

    def test_random_quat_reproducible(self):
        from quat.random import random_quat
        q1 = random_quat(42)
        q2 = random_quat(42)
        self.assertEqual(q1, q2)

    def test_random_quat_differs(self):
        from quat.random import random_quat
        q1 = random_quat(1)
        q2 = random_quat(2)
        self.assertNotEqual(q1, q2)

    def test_random_quat_rng(self):
        from quat.random import random_quat
        rng = np.random.default_rng(42)
        q1 = random_quat(rng)
        q2 = random_quat(42)
        self.assertEqual(q1, q2)


class TestRandomUnitQuat(QuatTestCase):
    def test_random_unit_quat_normalized(self):
        from quat.random import random_unit_quat
        for _ in range(20):
            q = random_unit_quat()
            self.assertAlmostEqual(q.norm(), 1.0, places=10)

    def test_random_unit_quat_reproducible(self):
        from quat.random import random_unit_quat
        q1 = random_unit_quat(42)
        q2 = random_unit_quat(42)
        self.assertEqual(q1, q2)


class TestRandomQuatVector(QuatTestCase):
    def test_random_quat_vector_shape(self):
        from quat.random import random_quat_vector
        v = random_quat_vector(5, 42)
        self.assertEqual(v.shape, (5,))

    def test_random_quat_vector_reproducible(self):
        from quat.random import random_quat_vector
        v1 = random_quat_vector(3, 42)
        v2 = random_quat_vector(3, 42)
        self.assertTrue(np.allclose(v1.to_array(), v2.to_array()))


class TestRandomQuatMatrix(QuatTestCase):
    def test_random_quat_matrix_shape(self):
        from quat.random import random_quat_matrix
        M = random_quat_matrix(3, 4, 42)
        self.assertEqual(M.shape, (3, 4))

    def test_random_quat_matrix_reproducible(self):
        from quat.random import random_quat_matrix
        M1 = random_quat_matrix(2, 2, 42)
        M2 = random_quat_matrix(2, 2, 42)
        self.assertTrue(np.allclose(M1.to_array(), M2.to_array()))


class TestRandomQuatTensor(QuatTestCase):
    def test_random_quat_tensor_shape(self):
        from quat.random import random_quat_tensor
        T = random_quat_tensor(2, 3, 4, 42)
        self.assertEqual(T.shape, (2, 3, 4))
