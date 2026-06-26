# Spec: Quaternion Signal Processing Module

**Date:** 2026-06-26
**Status:** approved

## Motivation

Add a `quat/signal.py` module providing quaternion Fourier transform (QFFT), convolution, and filter design. Quaternion-valued signals arise in color image processing, vector sensor arrays, and orientation data analysis.

## Module Structure

```
quat/signal.py
├── qfft(x, axis=-1, side='left')       — 1D QFFT (forward)
├── iqfft(X, axis=-1, side='left')      — 1D QFFT (inverse)
├── qfft2(x, axes=(-2,-1), side='left') — 2D QFFT (forward)
├── iqfft2(X, axes=(-2,-1), side='left')— 2D QFFT (inverse)
├── qconv(x, kernel, mode='full')       — 1D quaternion convolution
├── qconv2(x, kernel, mode='full')      — 2D quaternion convolution
├── lowpass(n, cutoff)                   — low-pass filter kernel
├── highpass(n, cutoff)                  — high-pass filter kernel
├── bandpass(n, low, high)               — band-pass filter kernel
└── bandstop(n, low, high)               — band-stop filter kernel
```

## Algorithm Design

### QFFT — Complex Pair Decomposition

Quaternion `q = a + bi + cj + dk` is decomposed as:

```
q = (a + bi) + (c + di) · j = s₁ + s₂ · j
```

where `s₁, s₂ ∈ ℂ`. The FFT is applied component-wise:

- **Left QFFT**: kernel multiplies from left → `FFT(s₁) + FFT(s₂)·j`
- **Right QFFT**: kernel multiplies from right → `FFT(s₁) + j·FFT(s₂)`

Implementation uses `np.fft.fft` / `np.fft.ifft` on the complex slices.

### Convolution

Implemented via convolution theorem: `qconv(x, k) = iqfft(qfft(x) · qfft(k))`. Supports `mode='full'`/`'same'`/`'valid'`.

### Filters

Standard FIR filter designs using window method (Hamming window):

| Function | Parameters |
|----------|-----------|
| `lowpass(n, cutoff)` | cutoff = normalized freq in (0, 0.5] |
| `highpass(n, cutoff)` | cutoff = normalized freq in (0, 0.5] |
| `bandpass(n, low, high)` | low, high = normalized bounds |
| `bandstop(n, low, high)` | low, high = normalized bounds |

Returns 1D QuatVector kernel of length `n`. Filters are pure-real (identity quaternion in the real component), so they commute with any quaternion signal.

### Input/Output Convention

All functions accept `(..., 4)` ndarrays (real,i,j,k in last axis). Output is always an ndarray — no Quaternion wrapping. If a function returns a filter kernel, it wraps in `QuatVector` for usability.

## Dependencies

- `numpy >= 1.21` (existing)
- `quat.core` (Quaternion, for filter output wrapping)
- `quat.collections` (QuatVector, for filter output)
- No new external packages

## Files

| File | Action |
|------|--------|
| `quat/signal.py` | Create — all signal functions |
| `tests/test_signal.py` | Create — all tests |
| `quat/__init__.py` | Modify — export signal functions |

## Test Plan

| Test Case | What It Verifies |
|-----------|-----------------|
| `test_qfft_identity` | Pure-real quaternion FFT equals scalar FFT |
| `test_qfft_linearity` | `QFFT(a·p + b·q) = a·QFFT(p) + b·QFFT(q)` |
| `test_qfft_shift` | Shift property (time shift ↔ phase rotation) |
| `test_iqfft_roundtrip` | `iqfft(qfft(x)) ≈ x` to 1e-10 |
| `test_qfft2_roundtrip` | 2D forward-inverse roundtrip |
| `test_qfft_side` | Left vs right QFFT produce different results |
| `test_qconv_impulse` | Convolution with unit impulse returns kernel |
| `test_qconv_modes` | full/same/valid mode shapes |
| `test_qconv2_identity` | 2D convolution with delta kernel |
| `test_lowpass_cutoff` | High frequencies attenuated after lowpass |
| `test_highpass_cutoff` | Low frequencies attenuated after highpass |
| `test_bandpass_range` | Only in-band frequencies pass |
| `test_filter_real_equivalent` | Real signal matches standard FIR filter |
