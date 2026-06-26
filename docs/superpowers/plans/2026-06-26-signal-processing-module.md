# Quaternion Signal Processing Module — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `quat/signal.py` with QFFT, convolution, and FIR filter design for quaternion-valued signals.

**Architecture:** Complex-pair decomposition reduces QFFT to two standard complex FFTs via numpy. Convolution uses the convolution theorem (FFT-based). Filters are pure-real FIR kernels via window method, wrapped in QuatVector.

**Tech Stack:** Python, NumPy, quat (core + collections)

---

### Task 1: Create test file skeleton + implement 1D QFFT/iqFFT

**Files:**
- Create: `tests/test_signal.py`
- Create: `quat/signal.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_signal.py`:

```python
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestQFFT1D(QuatTestCase):
    def test_qfft_identity_real(self):
        from quat.signal import qfft
        x = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(2, 4)
        result = qfft(x)
        self.assertEqual(result.shape, (2, 4))

    def test_qfft_linearity(self):
        from quat.signal import qfft
        p = np.random.randn(8, 4)
        q = np.random.randn(8, 4)
        a, b = 2.0, 3.0
        L = qfft(a * p + b * q)
        R = a * qfft(p) + b * qfft(q)
        self.assertTrue(np.allclose(L, R))

    def test_iqfft_roundtrip(self):
        from quat.signal import qfft, iqfft
        x = np.random.randn(16, 4)
        X = qfft(x)
        y = iqfft(X)
        self.assertTrue(np.allclose(x, y))

    def test_qfft_side_left_vs_right(self):
        from quat.signal import qfft
        x = np.random.randn(8, 4)
        L = qfft(x, side='left')
        R = qfft(x, side='right')
        self.assertFalse(np.allclose(L, R))

    def test_qfft_invalid_side_raises(self):
        from quat.signal import qfft
        x = np.random.randn(8, 4)
        with self.assertRaises(ValueError):
            qfft(x, side='both')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_signal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quat.signal'`

- [ ] **Step 3: Create `quat/signal.py` with qfft and iqfft**

Write `quat/signal.py`:

```python
"""Quaternion signal processing — QFFT, convolution, and filter design."""
from __future__ import annotations
import numpy as np
from quat.collections import QuatVector


def qfft(x: np.ndarray, axis: int = -1, side: str = 'left') -> np.ndarray:
    if side not in ('left', 'right'):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    x = np.asarray(x, dtype=float)
    a, b, c, d = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    s1 = a + 1j * b
    s2 = c + 1j * d
    S1 = np.fft.fft(s1, axis=axis)
    S2 = np.fft.fft(s2, axis=axis)
    result = np.empty_like(x)
    result[..., 0] = S1.real
    result[..., 1] = S1.imag
    if side == 'right':
        S2 = S2.conjugate()
    result[..., 2] = S2.real
    result[..., 3] = S2.imag
    return result


def iqfft(X: np.ndarray, axis: int = -1, side: str = 'left') -> np.ndarray:
    if side not in ('left', 'right'):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    X = np.asarray(X, dtype=float)
    a, b, c, d = X[..., 0], X[..., 1], X[..., 2], X[..., 3]
    s1 = a + 1j * b
    s2 = c + 1j * d
    S1 = np.fft.ifft(s1, axis=axis)
    S2 = np.fft.ifft(s2, axis=axis)
    result = np.empty_like(X)
    result[..., 0] = S1.real
    result[..., 1] = S1.imag
    if side == 'right':
        S2 = S2.conjugate()
    result[..., 2] = S2.real
    result[..., 3] = S2.imag
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_signal.py::TestQFFT1D -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add quat/signal.py tests/test_signal.py
git commit -m "feat: add 1D QFFT (qfft/iqfft) to quat.signal"
```

---

### Task 2: Add 2D QFFT (qfft2/iqfft2)

**Files:**
- Modify: `quat/signal.py` — add qfft2, iqfft2
- Modify: `tests/test_signal.py` — add test class

- [ ] **Step 1: Write the failing test**

Append to `tests/test_signal.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_signal.py::TestQFFT2D -v`
Expected: FAIL with `ImportError` / `AttributeError`

- [ ] **Step 3: Implement qfft2 and iqfft2**

Append to `quat/signal.py`:

```python
def qfft2(x: np.ndarray, axes=(-2, -1), side: str = 'left') -> np.ndarray:
    X = qfft(x, axis=axes[0], side=side)
    return qfft(X, axis=axes[1], side=side)


def iqfft2(X: np.ndarray, axes=(-2, -1), side: str = 'left') -> np.ndarray:
    x = iqfft(X, axis=axes[0], side=side)
    return iqfft(x, axis=axes[1], side=side)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_signal.py::TestQFFT2D -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add quat/signal.py tests/test_signal.py
git commit -m "feat: add 2D QFFT (qfft2/iqfft2)"
```

---

### Task 3: Add 1D quaternion convolution (qconv)

**Files:**
- Modify: `quat/signal.py` — add qconv
- Modify: `tests/test_signal.py` — add test class

- [ ] **Step 1: Write the failing test**

Append to `tests/test_signal.py`:

```python
class TestQConv1D(QuatTestCase):
    def test_qconv_impulse(self):
        from quat.signal import qconv
        x = np.array([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0],
                       [3.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]])
        kernel = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        result = qconv(x, kernel)
        self.assertEqual(result.shape, (7, 4))

    def test_qconv_real_equivalent(self):
        from quat.signal import qconv
        x_real = np.random.randn(16)
        k_real = np.random.randn(4)
        x_quat = np.zeros((16, 4))
        x_quat[:, 0] = x_real
        k_quat = np.zeros((4, 4))
        k_quat[:, 0] = k_real
        result = qconv(x_quat, k_quat)
        expected = np.convolve(x_real, k_real)
        self.assertTrue(np.allclose(result[:, 0], expected))

    def test_qconv_modes(self):
        from quat.signal import qconv
        x = np.random.randn(16, 4)
        k = np.random.randn(4, 4)
        self.assertEqual(qconv(x, k, mode='full').shape, (19, 4))
        self.assertEqual(qconv(x, k, mode='same').shape, (16, 4))
        self.assertEqual(qconv(x, k, mode='valid').shape, (13, 4))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_signal.py::TestQConv1D -v`
Expected: FAIL with `ImportError: cannot import name 'qconv'`

- [ ] **Step 3: Implement qconv**

Append to `quat/signal.py`:

```python
def qconv(x: np.ndarray, kernel: np.ndarray, mode: str = 'full') -> np.ndarray:
    x = np.asarray(x, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if x.shape[-1] != 4 or kernel.shape[-1] != 4:
        raise ValueError("Last axis must have size 4 (r,i,j,k)")
    n, k = x.shape[-2], kernel.shape[-2]
    N = n + k - 1
    x_pad = np.zeros((N, 4))
    k_pad = np.zeros((N, 4))
    x_pad[:n] = x
    k_pad[:k] = kernel
    X = qfft(x_pad, axis=0)
    K = qfft(k_pad, axis=0)
    Y = _hamilton_product_batch(X, K)
    result_full = iqfft(Y, axis=0).real
    if mode == 'full':
        return result_full
    elif mode == 'same':
        start = (k - 1) // 2
        return result_full[start:start + n]
    elif mode == 'valid':
        return result_full[k - 1:n]
    else:
        raise ValueError(f"mode must be 'full', 'same', or 'valid', got {mode!r}")
```

Wait — the convolution theorem for quaternions requires the Hamilton product in the frequency domain. Let me think about this more carefully.

For quaternion convolution, the frequency-domain multiplication is the Hamilton product. So we need `_hamilton_product_batch` which is essentially calling `_hamilton` from `quat.algebra`.

Actually, for the quaternion convolution theorem, `QFFT(x * k) = QFFT(x) ⊗ QFFT(k)` where ⊗ is the quaternion (Hamilton) product performed element-wise in the frequency domain.

But wait — the standard convolution theorem approach needs the FFT to be at least N+L-1 in length. Let me use the standard approach: pad to N+L-1, QFFT both, Hamilton-multiply, iQFFT.

For `_hamilton_product_batch`, I'll use `quat.algebra._hamilton` directly.

Let me rewrite step 3 more carefully. Actually, let me use `np.fft.fft` on component-level since QFFT with the left side is what we want for convolution.

Hmm, actually the quaternion convolution theorem depends on which QFFT you use. For the left-side QFFT, the convolution theorem holds with left-multiplication in the frequency domain:

`QFFT_left(x * k)[m] = QFFT_left(x)[m] * QFFT_left(k)[m]` (Hamilton product)

where * is convolution and the right * is Hamilton product.

So my implementation should:
1. Pad both to N+L-1
2. QFFT both (left side)
3. Hamilton-multiply element-wise in frequency domain
4. iQFFT the result
5. Slice for mode

Let me use `quat.algebra._hamilton` for the element-wise Hamilton product.

Actually, for the plan, let me keep it simpler. Let me implement `qconv` using the overlap-save approach that calls np.convolve on each component and then adjusts for the cross terms. 

Wait, that's even more complex. Let me just do the FFT-based approach properly.

```python
from quat.algebra import _hamilton

def qconv(x, kernel, mode='full'):
    x = np.asarray(x, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    n, k = x.shape[-2], kernel.shape[-2]
    N = n + k - 1
    # Zero-pad
    x_pad = np.zeros((N, 4))
    k_pad = np.zeros((N, 4))
    x_pad[:n] = x
    k_pad[:k] = kernel
    # QFFT
    X = qfft(x_pad, axis=0)
    K = qfft(k_pad, axis=0)
    # Hamilton product element-wise
    Y = _hamilton(X, K)
    # IQFFT
    y_full = iqfft(Y, axis=0)
    # Slice
    if mode == 'full':
        return y_full
    elif mode == 'same':
        start = (k - 1) // 2
        return y_full[start:start + n]
    elif mode == 'valid':
        return y_full[k - 1:n]
```

This should work! Let me update the plan.

- [ ] **Step 3: Implement qconv**

Append to `quat/signal.py`:

```python
from quat.algebra import _hamilton


def qconv(x: np.ndarray, kernel: np.ndarray, mode: str = 'full') -> np.ndarray:
    x = np.asarray(x, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if x.ndim != 2 or x.shape[-1] != 4:
        raise ValueError(f"x must be (n, 4), got {x.shape}")
    if kernel.ndim != 2 or kernel.shape[-1] != 4:
        raise ValueError(f"kernel must be (k, 4), got {kernel.shape}")
    n, k = x.shape[0], kernel.shape[0]
    N = n + k - 1
    x_pad = np.zeros((N, 4))
    k_pad = np.zeros((N, 4))
    x_pad[:n] = x
    k_pad[:k] = kernel
    X = qfft(x_pad, axis=0)
    K = qfft(k_pad, axis=0)
    Y = _hamilton(X, K)
    y_full = iqfft(Y, axis=0)
    if mode == 'full':
        return y_full
    elif mode == 'same':
        start = (k - 1) if k > 1 else 0
        return y_full[start:start + n]
    elif mode == 'valid':
        return y_full[k - 1:n]
    else:
        raise ValueError(f"mode must be 'full', 'same', or 'valid', got {mode!r}")
```

Wait, I also need to make sure the '_hamilton' import is at the top of the file. Actually, `_hamilton` is already imported from quat.algebra. Let me structure the imports properly.

Let me refine the plan. I should put the _hamilton import at the top of signal.py. For the plan step, let me just show what to add.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_signal.py::TestQConv1D -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add quat/signal.py tests/test_signal.py
git commit -m "feat: add 1D quaternion convolution (qconv)"
```

OK let me just write the whole plan now and make sure it's all correct. I'll write the tasks sequentially.

Actually, I realize I need to be careful about the `same` mode offset. Let me check: for standard numpy/scipy convolution with `mode='same'`, the output length is max(n, k) and it's centered. For numpy.convolve, `same` starts at `(k-1)//2`. Let me use that.

Now let me write the complete plan.<｜end▁of▁thinking｜>

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="write">
<｜｜DSML｜｜parameter name="content" string="true"># Quaternion Signal Processing Module — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `quat/signal.py` with QFFT, convolution, and FIR filter design for quaternion-valued signals.

**Architecture:** Complex-pair decomposition reduces QFFT to two standard complex FFTs via numpy. Convolution uses the convolution theorem with the Hamilton product in frequency domain. Filters are pure-real FIR kernels via window method, wrapped in QuatVector.

**Tech Stack:** Python, NumPy, quat (algebra + collections)

---

### Task 1: Create `quat/signal.py` + implement 1D QFFT/iqFFT

**Files:**
- Create: `tests/test_signal.py`
- Create: `quat/signal.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_signal.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_signal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quat.signal'`

- [ ] **Step 3: Create `quat/signal.py` with qfft and iqfft**

Write `quat/signal.py`:

```python
"""Quaternion signal processing — QFFT, convolution, and filter design."""
from __future__ import annotations
import numpy as np


def qfft(x: np.ndarray, axis: int = -1, side: str = 'left') -> np.ndarray:
    if side not in ('left', 'right'):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    x = np.asarray(x, dtype=float)
    a, b, c, d = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    s1 = a + 1j * b
    s2 = c + 1j * d
    S1 = np.fft.fft(s1, axis=axis)
    S2 = np.fft.fft(s2, axis=axis)
    if side == 'right':
        S2 = S2.conjugate()
    result = np.empty_like(x)
    result[..., 0] = S1.real
    result[..., 1] = S1.imag
    result[..., 2] = S2.real
    result[..., 3] = S2.imag
    return result


def iqfft(X: np.ndarray, axis: int = -1, side: str = 'left') -> np.ndarray:
    if side not in ('left', 'right'):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    X = np.asarray(X, dtype=float)
    a, b, c, d = X[..., 0], X[..., 1], X[..., 2], X[..., 3]
    s1 = a + 1j * b
    s2 = c + 1j * d
    S1 = np.fft.ifft(s1, axis=axis)
    S2 = np.fft.ifft(s2, axis=axis)
    if side == 'right':
        S2 = S2.conjugate()
    result = np.empty_like(X)
    result[..., 0] = S1.real
    result[..., 1] = S1.imag
    result[..., 2] = S2.real
    result[..., 3] = S2.imag
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_signal.py::TestQFFT1D -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add quat/signal.py tests/test_signal.py
git commit -m "feat: add 1D QFFT (qfft/iqfft) to quat.signal"
```

---

### Task 2: Add 2D QFFT (qfft2/iqfft2)

**Files:**
- Modify: `quat/signal.py`
- Modify: `tests/test_signal.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_signal.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_signal.py::TestQFFT2D -v`
Expected: FAIL with `ImportError: cannot import name 'qfft2'`

- [ ] **Step 3: Implement qfft2 and iqfft2**

Append to `quat/signal.py` (after `iqfft`):

```python
def qfft2(x: np.ndarray, axes=(-2, -1), side: str = 'left') -> np.ndarray:
    X = qfft(x, axis=axes[0], side=side)
    return qfft(X, axis=axes[1], side=side)


def iqfft2(X: np.ndarray, axes=(-2, -1), side: str = 'left') -> np.ndarray:
    x = iqfft(X, axis=axes[0], side=side)
    return iqfft(x, axis=axes[1], side=side)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_signal.py::TestQFFT2D -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add quat/signal.py tests/test_signal.py
git commit -m "feat: add 2D QFFT (qfft2/iqfft2)"
```

---

### Task 3: Add 1D quaternion convolution (qconv)

**Files:**
- Modify: `quat/signal.py`
- Modify: `tests/test_signal.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_signal.py`:

```python
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
        expected = np.convolve(x_real, k_real, mode='full')
        self.assertEqual(result.shape, (18, 4))
        self.assertTrue(np.allclose(result[:, 0], expected))

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_signal.py::TestQConv1D -v`
Expected: FAIL with `ImportError: cannot import name 'qconv'`

- [ ] **Step 3: Add import and implement qconv**

Add import at top of `quat/signal.py` (after `import numpy as np`):

```python
from quat.algebra import _hamilton
```

Append `qconv` after `iqfft2`:

```python
def qconv(x: np.ndarray, kernel: np.ndarray, mode: str = 'full') -> np.ndarray:
    x = np.asarray(x, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if x.ndim < 2 or x.shape[-1] != 4:
        raise ValueError(f"x last axis must have size 4 (r,i,j,k), got shape {x.shape}")
    if kernel.ndim < 2 or kernel.shape[-1] != 4:
        raise ValueError(f"kernel last axis must have size 4, got shape {kernel.shape}")
    if mode not in ('full', 'same', 'valid'):
        raise ValueError(f"mode must be 'full', 'same', or 'valid', got {mode!r}")
    n = x.shape[-2]
    k = kernel.shape[-2]
    N = n + k - 1
    x_pad = np.zeros(x.shape[:-2] + (N, 4))
    k_pad = np.zeros(kernel.shape[:-2] + (N, 4))
    x_pad[..., :n, :] = x
    k_pad[..., :k, :] = kernel
    X = qfft(x_pad, axis=-2)
    K = qfft(k_pad, axis=-2)
    Y = _hamilton(X, K)
    y_full = iqfft(Y, axis=-2)
    if mode == 'full':
        return y_full
    elif mode == 'same':
        start = (k - 1) // 2
        return y_full[..., start:start + n, :]
    else:
        return y_full[..., k - 1:n, :]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_signal.py::TestQConv1D -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add quat/signal.py tests/test_signal.py
git commit -m "feat: add 1D quaternion convolution (qconv)"
```

---

### Task 4: Add 2D quaternion convolution (qconv2)

**Files:**
- Modify: `quat/signal.py`
- Modify: `tests/test_signal.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_signal.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_signal.py::TestQConv2D -v`
Expected: FAIL with `ImportError: cannot import name 'qconv2'`

- [ ] **Step 3: Implement qconv2**

Append after `qconv`:

```python
def qconv2(x: np.ndarray, kernel: np.ndarray, mode: str = 'full') -> np.ndarray:
    x = np.asarray(x, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if x.ndim < 3 or x.shape[-1] != 4:
        raise ValueError(f"x last axis must have size 4, got shape {x.shape}")
    if kernel.ndim < 3 or kernel.shape[-1] != 4:
        raise ValueError(f"kernel last axis must have size 4, got shape {kernel.shape}")
    if mode not in ('full', 'same', 'valid'):
        raise ValueError(f"mode must be 'full', 'same', or 'valid', got {mode!r}")
    nh, nw = x.shape[-3], x.shape[-2]
    kh, kw = kernel.shape[-3], kernel.shape[-2]
    Nh, Nw = nh + kh - 1, nw + kw - 1
    x_pad = np.zeros(x.shape[:-3] + (Nh, Nw, 4))
    k_pad = np.zeros(kernel.shape[:-3] + (Nh, Nw, 4))
    x_pad[..., :nh, :nw, :] = x
    k_pad[..., :kh, :kw, :] = kernel
    X = qfft2(x_pad)
    K = qfft2(k_pad)
    Y = _hamilton(X, K)
    y_full = iqfft2(Y)
    if mode == 'full':
        return y_full
    elif mode == 'same':
        sh = (kh - 1) // 2
        sw = (kw - 1) // 2
        return y_full[..., sh:sh + nh, sw:sw + nw, :]
    else:
        return y_full[..., kh - 1:nh, kw - 1:nw, :]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_signal.py::TestQConv2D -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add quat/signal.py tests/test_signal.py
git commit -m "feat: add 2D quaternion convolution (qconv2)"
```

---

### Task 5: Add FIR filter design functions

**Files:**
- Modify: `quat/signal.py`
- Modify: `tests/test_signal.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_signal.py`:

```python
class TestFilters(QuatTestCase):
    def test_lowpass_shape(self):
        from quat.signal import lowpass
        f = lowpass(32, 0.25)
        self.assertEqual(f.shape, (32,))
        from quat.collections import QuatVector
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

    def test_lowpass_attenuation(self):
        from quat.signal import lowpass
        from quat.signal import qfft
        f = lowpass(64, 0.2)
        F = qfft(f.to_array())
        mag = np.sqrt((F * F).sum(axis=-1))
        self.assertGreater(mag[0], mag[32])
        self.assertGreater(mag[0] / max(mag[32], 1e-10), 10.0)

    def test_bandpass_range(self):
        from quat.signal import bandpass, lowpass, highpass
        from quat.signal import qfft
        bp = bandpass(64, 0.1, 0.3)
        F = qfft(bp.to_array())
        mag = np.sqrt((F * F).sum(axis=-1))
        self.assertGreater(mag[12], mag[0])
        self.assertGreater(mag[12], mag[32])

    def test_filter_real_pure(self):
        from quat.signal import lowpass
        f = lowpass(16, 0.25)
        arr = f.to_array()
        self.assertTrue(np.allclose(arr[:, 1:], 0.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_signal.py::TestFilters -v`
Expected: FAIL with `ImportError: cannot import name 'lowpass'`

- [ ] **Step 3: Implement filter functions**

Append after `qconv2`:

```python
def _fir_ideal(cutoff: float, n: int, ftype: str) -> np.ndarray:
    m = (n - 1) / 2
    t = np.arange(n) - m
    t = np.where(t == 0, 1e-12, t)
    if ftype == 'lowpass':
        return np.sinc(2 * cutoff * t) * (2 * cutoff)
    elif ftype == 'highpass':
        return -np.sinc(2 * cutoff * t) * (2 * cutoff)
    elif ftype == 'bandpass':
        return None
    return None


def _hamming_window(n: int) -> np.ndarray:
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def lowpass(n: int, cutoff: float):
    from quat.collections import QuatVector
    if not 0 < cutoff <= 0.5:
        raise ValueError(f"cutoff must be in (0, 0.5], got {cutoff}")
    ideal = _fir_ideal(cutoff, n, 'lowpass')
    window = _hamming_window(n)
    kernel = ideal * window
    data = np.zeros((n, 4))
    data[:, 0] = kernel
    return QuatVector(data)


def highpass(n: int, cutoff: float):
    from quat.collections import QuatVector
    if not 0 < cutoff <= 0.5:
        raise ValueError(f"cutoff must be in (0, 0.5], got {cutoff}")
    ideal = _fir_ideal(cutoff, n, 'highpass')
    window = _hamming_window(n)
    kernel = ideal * window
    kernel[n // 2] += 1.0
    data = np.zeros((n, 4))
    data[:, 0] = kernel
    return QuatVector(data)


def bandpass(n: int, low: float, high: float):
    from quat.collections import QuatVector
    if not 0 < low < high <= 0.5:
        raise ValueError(f"require 0 < low < high <= 0.5, got {low}, {high}")
    low_kernel = lowpass(n, low)._data[:, 0]
    high_kernel = lowpass(n, high)._data[:, 0]
    kernel = high_kernel - low_kernel
    data = np.zeros((n, 4))
    data[:, 0] = kernel
    return QuatVector(data)


def bandstop(n: int, low: float, high: float):
    from quat.collections import QuatVector
    if not 0 < low < high <= 0.5:
        raise ValueError(f"require 0 < low < high <= 0.5, got {low}, {high}")
    bp = bandpass(n, low, high)._data[:, 0]
    kernel = np.zeros(n)
    kernel[n // 2] = 1.0
    kernel = kernel - bp
    data = np.zeros((n, 4))
    data[:, 0] = kernel
    return QuatVector(data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_signal.py::TestFilters -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add quat/signal.py tests/test_signal.py
git commit -m "feat: add FIR filter design (lowpass/highpass/bandpass/bandstop)"
```

---

### Task 6: Update `__init__.py` + full test suite

**Files:**
- Modify: `quat/__init__.py`

- [ ] **Step 1: Add signal exports to `__init__.py`**

In `quat/__init__.py`, add import block after existing imports:

```python
from quat.signal import (
    qfft, iqfft, qfft2, iqfft2,
    qconv, qconv2,
    lowpass, highpass, bandpass, bandstop,
)
```

And add to `__all__`:

```python
    'qfft', 'iqfft', 'qfft2', 'iqfft2',
    'qconv', 'qconv2',
    'lowpass', 'highpass', 'bandpass', 'bandstop',
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS (6 new test classes, all existing tests still pass)

- [ ] **Step 3: Commit**

```bash
git add quat/__init__.py
git commit -m "feat: export signal processing functions from quat"
```
