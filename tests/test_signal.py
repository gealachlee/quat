import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestQFFT1D(QuatTestCase):
    def test_qfft_shape(self):
        from quat.signal import qfft
        x = np.array([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
        result = qfft(x)
        self.assertEqual(result.shape, (2, 4))

    def test_iqfft_roundtrip(self):
        from quat.signal import qfft, iqfft
        x = np.random.randn(16, 4)
        X = qfft(x)
        y = iqfft(X)
        self.assertTrue(np.allclose(x, y))

    def test_qfft_linearity(self):
        from quat.signal import qfft
        p = np.random.randn(8, 4)
        q = np.random.randn(8, 4)
        a, b = 2.0, 3.0
        L = qfft(a * p + b * q)
        R = a * qfft(p) + b * qfft(q)
        self.assertTrue(np.allclose(L, R))

    def test_qfft_side_left_vs_right(self):
        from quat.signal import qfft
        x = np.random.randn(8, 4)
        L = qfft(x, side='left')
        R = qfft(x, side='right')
        self.assertFalse(np.allclose(L, R))

    def test_qfft_invalid_side(self):
        from quat.signal import qfft
        x = np.random.randn(8, 4)
        with self.assertRaises(ValueError):
            qfft(x, side='both')

    def test_qfft_axis(self):
        from quat.signal import qfft
        x = np.random.randn(3, 8, 4)
        r0 = qfft(x, axis=0)
        r1 = qfft(x, axis=1)
        self.assertEqual(r0.shape, (3, 8, 4))
        self.assertEqual(r1.shape, (3, 8, 4))
        self.assertFalse(np.allclose(r0, r1))
