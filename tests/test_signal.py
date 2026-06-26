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


class TestQFFT2D(QuatTestCase):
    def test_qfft2_roundtrip(self):
        from quat.signal import qfft2, iqfft2
        x = np.random.randn(8, 8, 4)
        X = qfft2(x)
        self.assertEqual(X.shape, (8, 8, 4))
        y = iqfft2(X)
        self.assertTrue(np.allclose(x, y))

    def test_qfft2_axes(self):
        from quat.signal import qfft2
        x = np.random.randn(4, 6, 4)
        result = qfft2(x, axes=(0, 1))
        self.assertEqual(result.shape, (4, 6, 4))

    def test_qfft2_side(self):
        from quat.signal import qfft2
        x = np.random.randn(4, 4, 4)
        L = qfft2(x, side='left')
        R = qfft2(x, side='right')
        self.assertFalse(np.allclose(L, R))


class TestQConv1D(QuatTestCase):
    def test_qconv_real_equivalent(self):
        from quat.signal import qconv
        x_real = np.arange(1.0, 17.0)
        k_real = np.array([1.0, 2.0, 1.0])
        x_quat = np.zeros((16, 4))
        x_quat[:, 0] = x_real
        k_quat = np.zeros((3, 4))
        k_quat[:, 0] = k_real
        result = qconv(x_quat, k_quat)
        self.assertEqual(result.shape, (18, 4))
        self.assertTrue(np.allclose(result[:, 0], np.convolve(x_real, k_real, mode='full')))

    def test_qconv_modes(self):
        from quat.signal import qconv
        x = np.random.randn(16, 4)
        k = np.random.randn(4, 4)
        self.assertEqual(qconv(x, k, mode='full').shape, (19, 4))
        self.assertEqual(qconv(x, k, mode='same').shape, (16, 4))
        self.assertEqual(qconv(x, k, mode='valid').shape, (13, 4))

    def test_qconv_invalid_shape(self):
        from quat.signal import qconv
        x = np.random.randn(16, 3)
        k = np.random.randn(3, 4)
        with self.assertRaises(ValueError):
            qconv(x, k)


class TestQConv2D(QuatTestCase):
    def test_qconv2_identity(self):
        from quat.signal import qconv2
        x = np.random.randn(8, 8, 4)
        delta = np.zeros((1, 1, 4))
        delta[:, :, 0] = 1.0
        result = qconv2(x, delta, mode='same')
        self.assertEqual(result.shape, (8, 8, 4))
        self.assertTrue(np.allclose(result, x))

    def test_qconv2_modes(self):
        from quat.signal import qconv2
        x = np.random.randn(8, 8, 4)
        k = np.random.randn(3, 3, 4)
        self.assertEqual(qconv2(x, k, mode='full').shape, (10, 10, 4))
        self.assertEqual(qconv2(x, k, mode='same').shape, (8, 8, 4))
        self.assertEqual(qconv2(x, k, mode='valid').shape, (6, 6, 4))


class TestFilters(QuatTestCase):
    def test_lowpass_shape(self):
        from quat.signal import lowpass
        from quat.collections import QuatVector
        f = lowpass(32, 0.25)
        self.assertEqual(f.shape, (32,))
        self.assertIsInstance(f, QuatVector)

    def test_highpass_shape(self):
        from quat.signal import highpass
        f = highpass(32, 0.25)
        self.assertEqual(f.shape, (32,))

    def test_bandpass_shape(self):
        from quat.signal import bandpass
        f = bandpass(32, 0.1, 0.3)
        self.assertEqual(f.shape, (32,))

    def test_bandstop_shape(self):
        from quat.signal import bandstop
        f = bandstop(32, 0.1, 0.3)
        self.assertEqual(f.shape, (32,))

    def test_filter_real_pure(self):
        from quat.signal import lowpass
        f = lowpass(16, 0.25)
        arr = f.to_array()
        self.assertTrue(np.allclose(arr[:, 1:], 0.0))

    def test_lowpass_attenuation(self):
        from quat.signal import lowpass, qfft
        f = lowpass(64, 0.2)
        F = qfft(f.to_array())
        mag = np.sqrt((F * F).sum(axis=-1))
        self.assertGreater(float(mag[0]), float(mag[32]))

    def test_bandpass_range(self):
        from quat.signal import bandpass, qfft
        bp = bandpass(64, 0.1, 0.3)
        F = qfft(bp.to_array())
        mag = np.sqrt((F * F).sum(axis=-1))
        self.assertGreater(float(mag[12]), float(mag[0]))
