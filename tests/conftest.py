"""Shared test fixtures for quat tests."""
import unittest
import numpy as np


class QuatTestCase(unittest.TestCase):
    """Base test case with common quaternion generators."""

    @staticmethod
    def random_quaternion():
        return np.random.randn(4)

    @staticmethod
    def random_unit_quaternion():
        q = np.random.randn(4)
        return q / np.linalg.norm(q)

    @staticmethod
    def random_vector(n):
        return np.random.randn(n, 4)

    @staticmethod
    def random_matrix(m, n):
        return np.random.randn(m, n, 4)

    @staticmethod
    def random_tensor(p, q, r):
        return np.random.randn(p, q, r, 4)
