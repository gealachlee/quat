"""03_signal_filtering — QFFT, quaternion convolution, FIR filter design."""
import numpy as np
from quat import QuatVector
from quat.signal import qfft, iqfft, qconv, lowpass, highpass, bandpass

print("=" * 60)
print("1D QFFT round-trip (pure real signal)")
print("=" * 60)

signal = np.zeros((64, 4))
signal[:, 0] = np.sin(2 * np.pi * np.arange(64) / 8)  # pure-real
X = qfft(signal)
reconstructed = iqfft(X)
err = np.max(np.abs(signal - reconstructed))
print("round-trip max error:", err)

print()
print("=" * 60)
print("FIR lowpass filter (cutoff=0.2, length=16)")
print("=" * 60)

k_lp = lowpass(16, cutoff=0.2)
print("kernel shape:", k_lp.shape)
print("kernel (real part):", np.round(k_lp.real, 4))

print()
print("=" * 60)
print("Quaternion convolution")
print("=" * 60)

noisy = np.random.randn(128, 4)
filtered = qconv(noisy, k_lp._data, mode="same")
print("input shape:", noisy.shape)
print("output shape:", filtered.shape)
print("noise reduction ratio:", np.std(filtered) / np.std(noisy))

print()
print("=" * 60)
print("Bandpass filter (0.1–0.3)")
print("=" * 60)

k_bp = bandpass(32, low=0.1, high=0.3)
print("kernel shape:", k_bp.shape)
print("kernel (real part):", np.round(k_bp.real, 4))
