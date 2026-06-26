import numpy as np
from quat.core import Quaternion
from quat.collections import QuatVector, QuatMatrix
from quat.algebra import _hamilton
from quat.signal import qfft, qconv
from quat.optimized import hamilton_einsum, quat_matmul
from quat.linalg import svd, pseudo_inverse


def test_hamilton_small(benchmark):
    p = np.random.randn(100, 4)
    q = np.random.randn(100, 4)
    benchmark(_hamilton, p, q)


def test_hamilton_large(benchmark):
    p = np.random.randn(10000, 4)
    q = np.random.randn(10000, 4)
    benchmark(_hamilton, p, q)


def test_quaternion_mul(benchmark):
    a = Quaternion(*np.random.randn(4))
    b = Quaternion(*np.random.randn(4))
    benchmark(lambda: a * b)


def test_quat_matrix_mul_small(benchmark):
    A = QuatMatrix(np.random.randn(10, 10, 4))
    B = QuatMatrix(np.random.randn(10, 10, 4))
    benchmark(lambda: A * B)


def test_quat_matrix_mul_medium(benchmark):
    A = QuatMatrix(np.random.randn(32, 32, 4))
    B = QuatMatrix(np.random.randn(32, 32, 4))
    benchmark(lambda: A * B)


def test_qfft_small(benchmark):
    x = np.random.randn(256, 4)
    benchmark(qfft, x)


def test_qfft_large(benchmark):
    x = np.random.randn(4096, 4)
    benchmark(qfft, x)


def test_qconv_medium(benchmark):
    x = np.random.randn(256, 4)
    k = np.random.randn(16, 4)
    benchmark(qconv, x, k)


def test_svd_small(benchmark):
    A = QuatMatrix(np.random.randn(16, 16, 4))
    benchmark(svd, A)


def test_svd_medium(benchmark):
    A = QuatMatrix(np.random.randn(64, 64, 4))
    benchmark(svd, A)


def test_hamilton_einsum_small(benchmark):
    p = np.random.randn(100, 4)
    q = np.random.randn(100, 4)
    benchmark(hamilton_einsum, p, q)


def test_quat_matmul_small(benchmark):
    A = QuatMatrix(np.random.randn(10, 10, 4))
    B = QuatMatrix(np.random.randn(10, 10, 4))
    benchmark(quat_matmul, A, B)
