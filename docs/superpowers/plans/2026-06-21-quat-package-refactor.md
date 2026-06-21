# quat Package Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor flat `.py` files into a `quat` Python package with full test coverage (unittest), serialization utils, numpy-optimized operations, and data conversion helpers.

**Architecture:** Split `quaternion.py` into `algebra.py` (low-level ops), `core.py` (Quaternion class), `collections.py` (Vector/Matrix/Tensor). New modules: `utils.py`, `serialization.py`, `optimized.py`. Algorithms go to `quat.algorithms/`, data to `quat/data/`. All existing APIs preserved via `__init__.py` re-exports.

**Tech Stack:** Python 3, numpy, scipy (optional), Pillow (optional), scikit-learn (optional), unittest (stdlib)

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `quat/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "quat"
version = "0.1.0"
description = "Quaternion algebra library with kernel methods"
requires-python = ">=3.9"
dependencies = ["numpy>=1.21"]

[project.optional-dependencies]
data = ["scipy>=1.7", "Pillow>=9.0", "scikit-learn>=1.0"]

[tool.setuptools.packages.find]
where = ["."]
include = ["quat", "quat.*"]
```

- [ ] **Step 2: Create quat/__init__.py (stub)**

```python
"""Quaternion algebra library."""
```

- [ ] **Step 3: Create quat/algorithms/__init__.py (stub)**

```python
"""Quaternion learning algorithms."""
```

- [ ] **Step 4: Create tests/__init__.py (empty)**

```python
```

- [ ] **Step 5: Create tests/conftest.py (shared unittest fixtures)**

```python
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
```

- [ ] **Step 6: Create directory structure**

```powershell
New-Item -ItemType Directory -Path "quat\algorithms" -Force
New-Item -ItemType Directory -Path "quat\data" -Force
New-Item -ItemType Directory -Path "tests" -Force
```

- [ ] **Step 7: Verify scaffolding**

```powershell
python -c "import quat; print('quat package OK')"
```

Expected: `quat package OK`

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml quat/ quat/algorithms/ quat/data/ tests/
git commit -m "chore: project scaffolding for quat package"
```

---

### Task 2: algebra.py — Low-level Operations

**Files:**
- Create: `quat/algebra.py`
- Create: `tests/test_algebra.py`
- Modify: `quat/__init__.py`

- [ ] **Step 1: Write test_algebra.py**

```python
"""Tests for quat.algebra module."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestHamiltonProduct(QuatTestCase):
    def test_two_vectors(self):
        from quat.algebra import _hamilton
        p = np.array([[1., 2., 3., 4.]])
        q = np.array([[5., 6., 7., 8.]])
        r = _hamilton(p, q)
        self.assertEqual(r.shape, (1, 4))
        expected = np.array([[-60., 12., 30., 24.]])
        self.assertTrue(np.allclose(r, expected))

    def test_batch_square(self):
        from quat.algebra import _hamilton
        p = np.random.randn(5, 4)
        q = np.random.randn(1, 4)
        r = _hamilton(p, q)
        self.assertEqual(r.shape, (5, 4))

    def test_broadcasting(self):
        from quat.algebra import _hamilton
        p = np.random.randn(3, 1, 4)
        q = np.random.randn(1, 5, 4)
        r = _hamilton(p, q)
        self.assertEqual(r.shape, (3, 5, 4))

    def test_identity(self):
        from quat.algebra import _hamilton
        one = np.array([1., 0., 0., 0.])
        q = np.random.randn(10, 4)
        r1 = _hamilton(one[None, :], q)
        r2 = _hamilton(q, one[None, :])
        self.assertTrue(np.allclose(r1, q))
        self.assertTrue(np.allclose(r2, q))

    def test_basis_multiplication(self):
        from quat.algebra import _hamilton
        i = np.array([0., 1., 0., 0.])
        j = np.array([0., 0., 1., 0.])
        k = np.array([0., 0., 0., 1.])
        one = np.array([1., 0., 0., 0.])
        zero = np.array([0., 0., 0., 0.])
        self.assertTrue(np.allclose(_hamilton(i, i), -one))
        self.assertTrue(np.allclose(_hamilton(i, j), k))
        self.assertTrue(np.allclose(_hamilton(j, k), i))
        self.assertTrue(np.allclose(_hamilton(k, i), j))
        self.assertTrue(np.allclose(_hamilton(i, j), -_hamilton(j, i)))


class TestConstants(QuatTestCase):
    def test_conj_mask(self):
        from quat.algebra import _CONJ
        self.assertTrue(np.allclose(_CONJ, [1., -1., -1., -1.]))

    def test_RW_tensor(self):
        from quat.algebra import _RW
        self.assertEqual(_RW.shape, (4, 4, 4))
        q = np.array([1., 2., 3., 4.])
        R = np.einsum('rck,k->rc', _RW, q)
        expected = np.array([
            [1., -2., -3., -4.],
            [2.,  1., -4.,  3.],
            [3.,  4.,  1., -2.],
            [4., -3.,  2.,  1.]
        ])
        self.assertTrue(np.allclose(R, expected))
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
python -m pytest tests/test_algebra.py -v
```

Expected: FAIL (module `quat.algebra` not found / no `_hamilton`)

- [ ] **Step 3: Create quat/algebra.py**

```python
"""Low-level quaternion algebra operations — constants, Hamilton product, real-matrix tensor."""
import numpy as np

_CONJ  = np.array([1., -1., -1., -1.])
_REAL  = np.array([1.,  0.,  0.,  0.])
_ZERO4 = np.array([0.,  0.,  0.,  0.])

_RW = np.zeros((4, 4, 4))
_RW[0, 0, 0] = 1;   _RW[0, 1, 1] = -1;  _RW[0, 2, 2] = -1;  _RW[0, 3, 3] = -1
_RW[1, 0, 1] = 1;   _RW[1, 1, 0] = 1;   _RW[1, 2, 3] = -1;  _RW[1, 3, 2] = 1
_RW[2, 0, 2] = 1;   _RW[2, 1, 3] = 1;   _RW[2, 2, 0] = 1;   _RW[2, 3, 1] = -1
_RW[3, 0, 3] = 1;   _RW[3, 1, 2] = -1;  _RW[3, 2, 1] = 1;   _RW[3, 3, 0] = 1


def _hamilton(p, q):
    """Vectorized Hamilton product. p, q: broadcastable arrays, last dim == 4."""
    a1, b1, c1, d1 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    a2, b2, c2, d2 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    shp = np.broadcast_shapes(p.shape[:-1], q.shape[:-1]) + (4,)
    out = np.empty(shp)
    out[..., 0] = a1*a2 - b1*b2 - c1*c2 - d1*d2
    out[..., 1] = a1*b2 + b1*a2 + c1*d2 - d1*c2
    out[..., 2] = a1*c2 - b1*d2 + c1*a2 + d1*b2
    out[..., 3] = a1*d2 + b1*c2 - c1*b2 + d1*a2
    return out



```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
python -m pytest tests/test_algebra.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add quat/algebra.py tests/test_algebra.py
git commit -m "feat: add quat.algebra — Hamilton product, constants, RW tensor"
```

---

### Task 3: core.py — Quaternion Class

**Files:**
- Create: `quat/core.py`
- Create: `tests/test_core.py`

- [ ] **Step 1: Write tests/test_core.py**

```python
"""Tests for Quaternion class."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestQuaternionConstruction(QuatTestCase):
    def test_four_args(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertAlmostEqual(q.r, 1.0)
        self.assertAlmostEqual(q.i, 2.0)
        self.assertAlmostEqual(q.j, 3.0)
        self.assertAlmostEqual(q.k, 4.0)

    def test_scalar_only(self):
        from quat.core import Quaternion
        q = Quaternion(5.0)
        self.assertAlmostEqual(q.r, 5.0)
        self.assertAlmostEqual(q.i, 0.0)
        self.assertAlmostEqual(q.j, 0.0)
        self.assertAlmostEqual(q.k, 0.0)

    def test_two_args(self):
        from quat.core import Quaternion
        q = Quaternion(2., -1.)
        self.assertAlmostEqual(q.r, 2.0)
        self.assertAlmostEqual(q.i, -1.0)
        self.assertAlmostEqual(q.j, 0.0)

    def test_copy_constructor(self):
        from quat.core import Quaternion
        q1 = Quaternion(1, 2, 3, 4)
        q2 = Quaternion(q1)
        self.assertEqual(q1, q2)
        self.assertIsNot(q1, q2)

    def test_from_ndarray(self):
        from quat.core import Quaternion
        q = Quaternion(np.array([1., 2., 3., 4.]))
        self.assertAlmostEqual(q.r, 1.0)

    def test_from_list(self):
        from quat.core import Quaternion
        q = Quaternion([1., 2., 3., 4.])
        self.assertAlmostEqual(q.r, 1.0)

    def test_from_complex(self):
        from quat.core import Quaternion
        q = Quaternion(2. + 3.j)
        self.assertAlmostEqual(q.r, 2.0)
        self.assertAlmostEqual(q.i, 3.0)
        self.assertAlmostEqual(q.j, 0.0)

    def test_zero_factory(self):
        from quat.core import Quaternion
        q = Quaternion.zero()
        self.assertAlmostEqual(q.r, 0.0)
        self.assertAlmostEqual(q.i, 0.0)

    def test_from_axis_angle(self):
        from quat.core import Quaternion
        q = Quaternion.from_axis_angle((0, 0, 1), np.pi / 2)
        expected = np.cos(np.pi / 4)
        self.assertAlmostEqual(q.r, expected, places=10)
        self.assertAlmostEqual(q.k, np.sin(np.pi / 4), places=10)

    def test_invalid_ndarray_size(self):
        from quat.core import Quaternion
        with self.assertRaises(ValueError):
            Quaternion(np.array([1., 2., 3.]))


class TestQuaternionAccessors(QuatTestCase):
    def test_components(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertEqual(q.components, (1.0, 2.0, 3.0, 4.0))

    def test_scalar_vector(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertAlmostEqual(q.scalar, 1.0)
        self.assertTrue(np.allclose(q.vector, [2., 3., 4.]))

    def test_real_imag(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertAlmostEqual(q.real, 1.0)
        self.assertTrue(np.allclose(q.imag, [2., 3., 4.]))

    def test_getitem(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertEqual(q[0], 1.0)
        self.assertEqual(q[1], 2.0)
        self.assertEqual(q[2], 3.0)
        self.assertEqual(q[3], 4.0)

    def test_iter(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertEqual(list(q), [1.0, 2.0, 3.0, 4.0])


class TestQuaternionArithmetic(QuatTestCase):
    def test_add(self):
        from quat.core import Quaternion
        a = Quaternion(1, 2, 3, 4)
        b = Quaternion(5, 6, 7, 8)
        c = a + b
        self.assertTrue(np.allclose(c._data, [6., 8., 10., 12.]))

    def test_add_scalar(self):
        from quat.core import Quaternion
        a = Quaternion(1, 2, 3, 4)
        c = a + 3.0
        self.assertTrue(np.allclose(c._data, [4., 2., 3., 4.]))

    def test_radd_scalar(self):
        from quat.core import Quaternion
        a = Quaternion(1, 2, 3, 4)
        c = 3.0 + a
        self.assertTrue(np.allclose(c._data, [4., 2., 3., 4.]))

    def test_sub(self):
        from quat.core import Quaternion
        a = Quaternion(5, 6, 7, 8)
        b = Quaternion(1, 2, 3, 4)
        c = a - b
        self.assertTrue(np.allclose(c._data, [4., 4., 4., 4.]))

    def test_neg(self):
        from quat.core import Quaternion
        q = Quaternion(1, -2, 3, -4)
        n = -q
        self.assertTrue(np.allclose(n._data, [-1., 2., -3., 4.]))

    def test_mul_quaternion(self):
        from quat.core import Quaternion
        i = Quaternion(0, 1, 0, 0)
        j = Quaternion(0, 0, 1, 0)
        k = Quaternion(0, 0, 0, 1)
        self.assertEqual(i * i, Quaternion(-1, 0, 0, 0))
        self.assertEqual(i * j, k)
        self.assertEqual(j * k, i)
        self.assertEqual(k * i, j)

    def test_mul_scalar(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        r = q * 2.0
        self.assertTrue(np.allclose(r._data, [2., 4., 6., 8.]))

    def test_rmul_scalar(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        r = 3.0 * q
        self.assertTrue(np.allclose(r._data, [3., 6., 9., 12.]))

    def test_rmul_quaternion(self):
        from quat.core import Quaternion
        r = Quaternion(2, 0, 0, 0)
        q = Quaternion(1, 2, 3, 4)
        l = r * q  # 2 * q
        self.assertTrue(np.allclose(l._data, [2., 4., 6., 8.]))

    def test_div_scalar(self):
        from quat.core import Quaternion
        q = Quaternion(2, 4, 6, 8)
        r = q / 2.0
        self.assertTrue(np.allclose(r._data, [1., 2., 3., 4.]))

    def test_div_quaternion(self):
        from quat.core import Quaternion
        a = Quaternion(2, 0, 0, 0)
        b = Quaternion(1, 0, 0, 0)
        self.assertEqual(a / b, Quaternion(2, 0, 0, 0))

    def test_commutator(self):
        from quat.core import Quaternion
        i = Quaternion(0, 1, 0, 0)
        j = Quaternion(0, 0, 1, 0)
        k = Quaternion(0, 0, 0, 1)
        self.assertEqual(i.commutator(j), Quaternion(0, 0, 0, 2))


class TestQuaternionAlgebra(QuatTestCase):
    def test_conjugate(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        c = q.conjugate()
        self.assertTrue(np.allclose(c._data, [1., -2., -3., -4.]))

    def test_norm(self):
        from quat.core import Quaternion
        q = Quaternion(3, 4, 0, 0)
        self.assertAlmostEqual(q.norm(), 5.0)

    def test_norm_squared(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 2, 0)
        self.assertAlmostEqual(q.norm_squared(), 9.0)

    def test_normalize(self):
        from quat.core import Quaternion
        q = Quaternion(3, 4, 0, 0)
        n = q.normalize()
        self.assertAlmostEqual(n.norm(), 1.0)

    def test_normalize_zero_raises(self):
        from quat.core import Quaternion
        q = Quaternion(0, 0, 0, 0)
        with self.assertRaises(ZeroDivisionError):
            q.normalize()

    def test_inverse(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        inv = q.inverse()
        self.assertAlmostEqual((q * inv).r, 1.0, places=10)
        self.assertAlmostEqual((inv * q).r, 1.0, places=10)

    def test_inverse_zero_raises(self):
        from quat.core import Quaternion
        with self.assertRaises(ZeroDivisionError):
            Quaternion.zero().inverse()

    def test_exp_log_roundtrip(self):
        from quat.core import Quaternion
        q = Quaternion(0, 0.3, 0.4, 0)
        roundtrip = q.exp().log()
        self.assertAlmostEqual(q.r, roundtrip.r, places=10)
        self.assertAlmostEqual(q.i, roundtrip.i, places=10)

    def test_pow_integer(self):
        from quat.core import Quaternion
        q = Quaternion(1, 0, 0, 0)
        self.assertEqual(q.pow(3), Quaternion(1, 0, 0, 0))

    def test_dot(self):
        from quat.core import Quaternion
        a = Quaternion(1, 2, 3, 4)
        b = Quaternion(4, 3, 2, 1)
        self.assertAlmostEqual(a.dot(b), 20.0)


class TestQuaternionRotation(QuatTestCase):
    def test_rotate_vector(self):
        from quat.core import Quaternion
        q_rot = Quaternion.from_axis_angle((0, 0, 1), np.pi / 2)
        v = (1.0, 0.0, 0.0)
        vr = q_rot.rotate_vector(v)
        self.assertAlmostEqual(vr[0], 0.0, places=10)
        self.assertAlmostEqual(vr[1], 1.0, places=10)
        self.assertAlmostEqual(vr[2], 0.0, places=10)


class TestQuaternionMatrixRepresentations(QuatTestCase):
    def test_complex_matrix(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        M = q.to_complex_matrix()
        self.assertEqual(M.shape, (2, 2))

    def test_complex_roundtrip(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertEqual(q, Quaternion.from_complex_matrix(q.to_complex_matrix()))

    def test_real_matrix(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        M = q.to_real_matrix()
        self.assertEqual(M.shape, (4, 4))
        self.assertTrue(np.allclose(M.T @ M, q.norm_squared() * np.eye(4)))

    def test_real_roundtrip(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        self.assertEqual(q, Quaternion.from_real_matrix(q.to_real_matrix()))

    def test_to_array(self):
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        arr = q.to_array()
        self.assertTrue(np.allclose(arr, [1., 2., 3., 4.]))


class TestQuaternionTypeConversions(QuatTestCase):
    def test_float(self):
        from quat.core import Quaternion
        self.assertEqual(float(Quaternion(3.5)), 3.5)

    def test_float_nonreal_raises(self):
        from quat.core import Quaternion
        with self.assertRaises(ValueError):
            float(Quaternion(1, 2, 3, 4))

    def test_int(self):
        from quat.core import Quaternion
        self.assertEqual(int(Quaternion(3)), 3)

    def test_complex(self):
        from quat.core import Quaternion
        c = complex(Quaternion(2, 3))
        self.assertEqual(c, 2 + 3j)

    def test_bool(self):
        from quat.core import Quaternion
        self.assertTrue(bool(Quaternion(1, 2, 3, 4)))
        self.assertFalse(bool(Quaternion(0, 0, 0, 0)))

    def test_abs(self):
        from quat.core import Quaternion
        q = Quaternion(3, 4, 0, 0)
        self.assertAlmostEqual(abs(q), 5.0)


class TestQuaternionHashEquality(QuatTestCase):
    def test_equal(self):
        from quat.core import Quaternion
        q1 = Quaternion(1, 2, 3, 4)
        q2 = Quaternion(1, 2, 3, 4)
        self.assertEqual(q1, q2)

    def test_not_equal(self):
        from quat.core import Quaternion
        self.assertNotEqual(Quaternion(1, 0, 0, 0), Quaternion(2, 0, 0, 0))

    def test_hash(self):
        from quat.core import Quaternion
        q1 = Quaternion(1.5, 2.5, 3.5, 4.5)
        q2 = Quaternion(1.5, 2.5, 3.5, 4.5)
        self.assertEqual(hash(q1), hash(q2))

    def test_representation(self):
        from quat.core import Quaternion
        q = Quaternion(1, -2, 3, -4)
        r = repr(q)
        self.assertIn("1", r)
        self.assertIn("-2", r)


class TestQuatFactory(QuatTestCase):
    def test_quat_function(self):
        from quat.core import quat, Quaternion
        q = quat(1, 2, 3, 4)
        self.assertIsInstance(q, Quaternion)
        self.assertEqual(q, Quaternion(1, 2, 3, 4))
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
python -m pytest tests/test_core.py -v
```

Expected: FAIL (module `quat.core` not found)

- [ ] **Step 3: Create quat/core.py** — Migrate the entire `Quaternion` class (lines 37-355) and `quat()` factory (line 995) from `quaternion.py`. Add import at top:

```python
"""Quaternion class — single quaternion value type."""
import numpy as np
from typing import Tuple, List
from numbers import Real, Complex
from quat.algebra import _hamilton, _CONJ
```

Content: copy lines 37-355 (Quaternion class) from `quaternion.py` verbatim. Then add at end of file:

```python
def quat(*args):
    """Convenience constructor for Quaternion."""
    return Quaternion(*args)


_I    = Quaternion(0, 1, 0, 0)
_J    = Quaternion(0, 0, 1, 0)
_K    = Quaternion(0, 0, 0, 1)
_ZERO = Quaternion(0, 0, 0, 0)
_R    = Quaternion(1, 0, 0, 0)
_ONE  = Quaternion(1, 1, 1, 1)
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
python -m pytest tests/test_core.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add quat/core.py tests/test_core.py
git commit -m "feat: add quat.core — Quaternion class with full arithmetic, algebra, rotation"
```

---

### Task 4: collections.py — QuatVector, QuatMatrix, QuatTensor

**Files:**
- Create: `quat/collections.py`
- Create: `tests/test_collections.py`

- [ ] **Step 1: Write tests/test_collections.py**

```python
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
        self.assertTrue(np.allclose(v.to_array(), 0.))

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
        v = QuatVector(np.eye(4))  # i, j, k, (1,1,1,1)
        q = Quaternion(2, 0, 0, 0)
        r = v * q
        self.assertTrue(np.allclose(r.to_array(), np.eye(4) * 2))

    def test_inner(self):
        from quat.collections import QuatVector
        a = QuatVector(np.eye(4)[:1])  # [1,0,0,0]
        b = QuatVector(np.eye(4)[:1])
        result = a.inner(b)
        # inner(A,B) = sum conj(a_i) * b_i = [1,0,0,0] * [1,0,0,0] = 1
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
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
python -m pytest tests/test_collections.py -v
```

Expected: FAIL

- [ ] **Step 3: Create quat/collections.py**

Migrate all three classes from `quaternion.py`:
- `QuatVector` class (lines 358-534)
- `QuatMatrix` class (lines 537-760)
- `QuatTensor` class (lines 762-990)
- Module-level functions: `dict_to_quat_matrix`, `dict_to_quat_tensor`, `labels_to_quat_vector` (lines 999-1038)

Add imports at top:
```python
"""Quaternion collection types — QuatVector, QuatMatrix, QuatTensor."""
import numpy as np
from typing import Tuple, List
from numbers import Real, Complex
from quat.algebra import _hamilton, _CONJ, _RW
from quat.core import Quaternion
```

Copy the class bodies and functions verbatim from `quaternion.py`.

- [ ] **Step 4: Run tests to verify they pass**

```powershell
python -m pytest tests/test_collections.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add quat/collections.py tests/test_collections.py
git commit -m "feat: add quat.collections — QuatVector, QuatMatrix, QuatTensor"
```

---

### Task 5: utils.py — Data Conversion Helpers

**Files:**
- Create: `quat/utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write tests/test_utils.py**

```python
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
        self.assertTrue(np.allclose(arr, [[1., 2., 3., 4.], [5., 6., 7., 8.]]))

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
        self.assertEqual(q.i, 2.0)

    def test_vector(self):
        from quat.utils import from_ndarray
        from quat.collections import QuatVector
        v = from_ndarray(np.random.randn(5, 4))
        self.assertIsInstance(v, QuatVector)
        self.assertEqual(v.shape, (5,))

    def test_matrix(self):
        from quat.utils import from_ndarray
        from quat.collections import QuatMatrix
        M = from_ndarray(np.random.randn(3, 4, 4))
        self.assertIsInstance(M, QuatMatrix)
        self.assertEqual(M.shape, (3, 4))

    def test_tensor(self):
        from quat.utils import from_ndarray
        from quat.collections import QuatTensor
        T = from_ndarray(np.random.randn(2, 3, 4, 4))
        self.assertIsInstance(T, QuatTensor)
        self.assertEqual(T.shape, (2, 3, 4))

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
        self.assertEqual(v.shape, (3,))
        self.assertTrue(np.allclose(v.real, r))
        self.assertTrue(np.allclose(v.i, i))

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


class TestBroadcastHelpers(QuatTestCase):
    def test_broadcast_shapes(self):
        from quat.utils import broadcast_quat_shapes
        shapes = broadcast_quat_shapes((3, 1), (1, 4))
        self.assertEqual(shapes, (3, 4))

    def test_stack_vectors(self):
        from quat.utils import stack_quat
        from quat.collections import QuatVector, QuatMatrix
        from quat.core import Quaternion
        q1 = Quaternion(1, 0, 0, 0)
        q2 = Quaternion(0, 1, 0, 0)
        q3 = Quaternion(0, 0, 1, 0)
        v = stack_quat([q1, q2, q3])
        self.assertIsInstance(v, QuatVector)
        self.assertEqual(v.shape, (3,))
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
python -m pytest tests/test_utils.py -v
```

Expected: FAIL

- [ ] **Step 3: Create quat/utils.py**

```python
"""Data conversion helpers for quaternion types."""
import numpy as np
from quat.core import Quaternion
from quat.collections import QuatVector, QuatMatrix, QuatTensor


def to_ndarray(q):
    """Convert any quaternion type to a plain numpy ndarray.

    Args:
        q: Quaternion, QuatVector, QuatMatrix, or QuatTensor

    Returns:
        ndarray with last dimension == 4
    """
    if isinstance(q, Quaternion):
        return q._data.copy()
    if isinstance(q, (QuatVector, QuatMatrix, QuatTensor)):
        return q._data.copy()
    raise TypeError(f"Cannot convert {type(q)} to ndarray")


def from_ndarray(arr):
    """Convert an ndarray to the appropriate quaternion type.

    Args:
        arr: ndarray with last dimension == 4,
             or 1-d array (will be expanded)

    Returns:
        Quaternion, QuatVector, QuatMatrix, or QuatTensor
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        if arr.size == 4:
            return Quaternion(arr)
        if arr.size == 1:
            return Quaternion(arr[0])
        raise ValueError(f"Cannot infer type from 1-d array of size {arr.size}")
    if arr.ndim == 2:
        return QuatVector(arr)
    if arr.ndim == 3:
        return QuatMatrix(arr)
    if arr.ndim == 4:
        return QuatTensor(arr)
    raise ValueError(f"Cannot infer quaternion type from {arr.ndim}-d array")


def batch_quat(r, i, j, k):
    """Construct quaternion collection from separate component arrays.

    Args:
        r, i, j, k: component arrays of the same shape (or broadcastable)

    Returns:
        QuatVector (if components are 1-d),
        QuatMatrix (if components are 2-d),
        QuatTensor (if components are 3-d)
    """
    r = np.asarray(r, dtype=float)
    i = np.asarray(i, dtype=float)
    j = np.asarray(j, dtype=float)
    k = np.asarray(k, dtype=float)
    data = np.stack([r, i, j, k], axis=-1)
    return from_ndarray(data)


def broadcast_quat_shapes(*shapes):
    """Compute the broadcast shape for quaternion collections."""
    return np.broadcast_shapes(*shapes)


def stack_quat(quaternions, axis=0):
    """Stack a sequence of quaternion objects into a collection.

    Args:
        quaternions: list of Quaternion, QuatVector, QuatMatrix, or QuatTensor
        axis: axis along which to stack (for collection types)

    Returns:
        QuatVector, QuatMatrix, or QuatTensor
    """
    if len(quaternions) == 0:
        raise ValueError("Cannot stack empty sequence")
    first = quaternions[0]
    if isinstance(first, Quaternion):
        data = np.stack([q._data for q in quaternions], axis=axis)
        return from_ndarray(data)
    if isinstance(first, QuatVector):
        data = np.stack([q._data for q in quaternions], axis=axis)
        return from_ndarray(data)
    raise TypeError(f"Cannot stack type {type(first)}")
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
python -m pytest tests/test_utils.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add quat/utils.py tests/test_utils.py
git commit -m "feat: add quat.utils — to/from_ndarray, batch_quat, stack_quat"
```

---

### Task 6: serialization.py — JSON, Bytes, scipy Interop

**Files:**
- Create: `quat/serialization.py`
- Create: `tests/test_serialization.py`

- [ ] **Step 1: Write tests/test_serialization.py**

```python
"""Tests for quat.serialization — JSON, binary, scipy interop."""
import unittest
import json
import numpy as np
from tests.conftest import QuatTestCase


class TestJSONSerialization(QuatTestCase):
    def test_quaternion_roundtrip(self):
        from quat.serialization import to_json, from_json
        from quat.core import Quaternion
        q = Quaternion(1.5, -2.3, 3.7, -4.1)
        s = to_json(q)
        q2 = from_json(s)
        self.assertEqual(q, q2)

    def test_quatvector_roundtrip(self):
        from quat.serialization import to_json, from_json
        from quat.collections import QuatVector
        v = QuatVector(np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]))
        s = to_json(v)
        v2 = from_json(s)
        self.assertTrue(np.allclose(v.to_array(), v2.to_array()))

    def test_quatmatrix_roundtrip(self):
        from quat.serialization import to_json, from_json
        from quat.collections import QuatMatrix
        M = QuatMatrix(np.random.randn(3, 4, 4))
        s = to_json(M)
        M2 = from_json(s)
        self.assertTrue(np.allclose(M.to_array(), M2.to_array()))

    def test_quattensor_roundtrip(self):
        from quat.serialization import to_json, from_json
        from quat.collections import QuatTensor
        T = QuatTensor(np.random.randn(2, 3, 4, 4))
        s = to_json(T)
        T2 = from_json(s)
        self.assertTrue(np.allclose(T.to_array(), T2.to_array()))

    def test_json_format(self):
        from quat.serialization import to_json, from_json
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        s = to_json(q)
        d = json.loads(s)
        self.assertIn("type", d)
        self.assertIn("data", d)


class TestBinarySerialization(QuatTestCase):
    def test_quaternion_roundtrip(self):
        from quat.serialization import to_bytes, from_bytes
        from quat.core import Quaternion
        q = Quaternion(1.5, -2.3, 3.7, -4.1)
        b = to_bytes(q)
        q2 = from_bytes(b)
        self.assertEqual(q, q2)

    def test_matrix_roundtrip(self):
        from quat.serialization import to_bytes, from_bytes
        from quat.collections import QuatMatrix
        M = QuatMatrix(np.random.randn(3, 4, 4))
        b = to_bytes(M)
        M2 = from_bytes(b)
        self.assertTrue(np.allclose(M.to_array(), M2.to_array()))


class TestNumpyInterop(QuatTestCase):
    def test_ndarray_quaternion(self):
        from quat.serialization import as_ndarray
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        arr = as_ndarray(q)
        self.assertEqual(arr.shape, (1, 4))
        self.assertTrue(np.allclose(arr, [[1., 2., 3., 4.]]))

    def test_ndarray_vector(self):
        from quat.serialization import as_ndarray
        from quat.collections import QuatVector
        v = QuatVector(np.ones((5, 4)))
        arr = as_ndarray(v)
        self.assertEqual(arr.shape, (5, 4))
        self.assertTrue(arr.flags['C_CONTIGUOUS'] or arr.flags['F_CONTIGUOUS'])

    def test_ndarray_matrix(self):
        from quat.serialization import as_ndarray
        from quat.collections import QuatMatrix
        M = QuatMatrix(np.ones((3, 4, 4)))
        arr = as_ndarray(M)
        self.assertEqual(arr.shape, (3, 4, 4))


class TestScipyRotationInterop(QuatTestCase):
    def test_quaternion_to_scipy(self):
        try:
            from scipy.spatial.transform import Rotation
        except ImportError:
            self.skipTest("scipy not available")
        from quat.serialization import to_scipy_rotation, from_scipy_rotation
        from quat.core import Quaternion
        q = Quaternion.from_axis_angle((0, 0, 1), np.pi / 2)
        rot = to_scipy_rotation(q)
        self.assertIsInstance(rot, Rotation)

    def test_roundtrip_scipy(self):
        try:
            from scipy.spatial.transform import Rotation
        except ImportError:
            self.skipTest("scipy not available")
        from quat.serialization import to_scipy_rotation, from_scipy_rotation
        from quat.core import Quaternion
        q = Quaternion.from_axis_angle((1, 0, 0), np.pi / 4)
        rot = to_scipy_rotation(q)
        q2 = from_scipy_rotation(rot)
        self.assertAlmostEqual(q.normalize().r, q2.normalize().r, places=10)

    def test_rotation_vector_to_scipy(self):
        try:
            from scipy.spatial.transform import Rotation
        except ImportError:
            self.skipTest("scipy not available")
        from quat.serialization import to_scipy_rotation, from_scipy_rotation
        from quat.collections import QuatVector
        from quat.core import Quaternion
        q1 = Quaternion.from_axis_angle((1, 0, 0), 0.5)
        q2 = Quaternion.from_axis_angle((0, 1, 0), 1.0)
        v = QuatVector([q1, q2])
        rot = to_scipy_rotation(v)
        v2 = from_scipy_rotation(rot)
        self.assertEqual(v2.shape, (2,))
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
python -m pytest tests/test_serialization.py -v
```

Expected: FAIL

- [ ] **Step 3: Create quat/serialization.py**

```python
"""Serialization and interop: JSON, binary bytes, scipy Rotation, numpy views."""
import json
import struct
import numpy as np
from quat.core import Quaternion
from quat.collections import QuatVector, QuatMatrix, QuatTensor
from quat.utils import to_ndarray, from_ndarray


def to_json(q):
    """Serialize quaternion object to JSON string.

    Returns:
        JSON string with {"type": "<classname>", "data": <list>}
    """
    data = to_ndarray(q)
    result = {
        "type": type(q).__name__,
        "data": data.tolist()
    }
    return json.dumps(result)


def from_json(s):
    """Deserialize JSON string to quaternion object.

    Args:
        s: JSON string produced by to_json()

    Returns:
        Quaternion, QuatVector, QuatMatrix, or QuatTensor
    """
    d = json.loads(s)
    arr = np.array(d["data"], dtype=float)
    return from_ndarray(arr)


def to_bytes(q):
    """Serialize quaternion object to bytes.

    Format: 4-byte header (type_id, ndim, reserved, reserved) + float64 data.
      type_id: 0=Quaternion, 1=QuatVector, 2=QuatMatrix, 3=QuatTensor

    Returns:
        bytes
    """
    from quat.collections import QuatVector, QuatMatrix, QuatTensor
    if isinstance(q, Quaternion):
        type_id = 0
    elif isinstance(q, QuatVector):
        type_id = 1
    elif isinstance(q, QuatMatrix):
        type_id = 2
    elif isinstance(q, QuatTensor):
        type_id = 3
    else:
        raise TypeError(f"Cannot serialize {type(q)} to bytes")
    data = to_ndarray(q)
    shape = np.array(data.shape, dtype=np.int32)
    header = struct.pack('<Ii', type_id, len(shape)) + shape.tobytes()
    return header + data.astype(np.float64).tobytes()


def from_bytes(b):
    """Deserialize bytes back to quaternion object.

    Args:
        b: bytes produced by to_bytes()

    Returns:
        Quaternion, QuatVector, QuatMatrix, or QuatTensor
    """
    type_id, ndim = struct.unpack_from('<Ii', b, 0)
    offset = 8
    shape = np.frombuffer(b[offset:offset + ndim * 4], dtype=np.int32)
    offset += ndim * 4
    size = int(np.prod(shape))
    data = np.frombuffer(b[offset:offset + size * 8], dtype=np.float64)
    data = data.reshape(shape)
    return from_ndarray(data)


def as_ndarray(q):
    """Return a numpy ndarray view/copy optimized for numpy interop.

    For Quaternion: returns (1, 4) array.
    For collections: returns the underlying _data array as contiguous copy.
    """
    if isinstance(q, Quaternion):
        return q._data.copy().reshape(1, 4)
    return np.ascontiguousarray(q._data)


def to_scipy_rotation(q):
    """Convert quaternion to scipy.spatial.transform.Rotation.

    Only works for unit quaternions (rotation quaternions).
    For QuatVector of unit quaternions, returns a single Rotation object.

    Args:
        q: Quaternion or QuatVector (must be unit quaternions)

    Returns:
        scipy.spatial.transform.Rotation
    """
    from scipy.spatial.transform import Rotation
    data = as_ndarray(q)
    if data.ndim == 2:  # (n, 4)
        return Rotation.from_quat(data[:, [1, 2, 3, 0]])  # scipy: x,y,z,w
    return Rotation.from_quat(data[0, [1, 2, 3, 0]])


def from_scipy_rotation(rot):
    """Convert scipy.spatial.transform.Rotation to QuatVector.

    Args:
        rot: scipy.spatial.transform.Rotation

    Returns:
        QuatVector of unit quaternions (w,x,y,z format internally)
    """
    scipy_quat = rot.as_quat()  # (n, 4) in x,y,z,w order
    if scipy_quat.ndim == 1:
        scipy_quat = scipy_quat.reshape(1, -1)
    data = np.zeros((scipy_quat.shape[0], 4))
    data[:, 0] = scipy_quat[:, 3]  # w -> r
    data[:, 1] = scipy_quat[:, 0]  # x -> i
    data[:, 2] = scipy_quat[:, 1]  # y -> j
    data[:, 3] = scipy_quat[:, 2]  # z -> k
    return QuatVector(data)
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
python -m pytest tests/test_serialization.py -v
```

Expected: all PASS (scipy tests skip if scipy unavailable)

- [ ] **Step 5: Commit**

```bash
git add quat/serialization.py tests/test_serialization.py
git commit -m "feat: add quat.serialization — JSON, binary, scipy Rotation interop"
```

---

### Task 7: optimized.py — Numpy-Vectorized Optimizations

**Files:**
- Create: `quat/optimized.py`
- Create: `tests/test_optimized.py`

- [ ] **Step 1: Write tests/test_optimized.py**

```python
"""Tests for quat.optimized — correctness vs standard implementations."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestOptimizedHamilton(QuatTestCase):
    def test_correctness_vs_standard(self):
        from quat.algebra import _hamilton
        from quat.optimized import hamilton_einsum
        p = np.random.randn(10, 4)
        q = np.random.randn(10, 4)
        r1 = _hamilton(p, q)
        r2 = hamilton_einsum(p, q)
        self.assertTrue(np.allclose(r1, r2))

    def test_broadcasting(self):
        from quat.algebra import _hamilton
        from quat.optimized import hamilton_einsum
        p = np.random.randn(3, 1, 4)
        q = np.random.randn(1, 5, 4)
        r1 = _hamilton(p, q)
        r2 = hamilton_einsum(p, q)
        self.assertTrue(np.allclose(r1, r2))


class TestOptimizedMatrixMultiply(QuatTestCase):
    def test_correctness(self):
        from quat.optimized import quat_matmul
        from quat.collections import QuatMatrix
        A = QuatMatrix(np.random.randn(3, 4, 4))
        B = QuatMatrix(np.random.randn(4, 5, 4))
        C1 = (A * B).to_array()
        C2 = quat_matmul(A._data, B._data)
        self.assertTrue(np.allclose(C1, C2))


class TestOptimizedOperations(QuatTestCase):
    def test_conjugate_batch(self):
        from quat.optimized import conjugate_batch
        data = np.random.randn(100, 4)
        expected = data * np.array([1., -1., -1., -1.])
        result = conjugate_batch(data)
        self.assertTrue(np.allclose(result, expected))

    def test_norm_squared_batch(self):
        from quat.optimized import norm_squared_batch
        data = np.random.randn(50, 4)
        result = norm_squared_batch(data)
        expected = (data * data).sum(axis=-1)
        self.assertTrue(np.allclose(result, expected))
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
python -m pytest tests/test_optimized.py -v
```

Expected: FAIL

- [ ] **Step 3: Create quat/optimized.py**

```python
"""Numpy-vectorized optimizations for quaternion operations.

All functions produce identical results to the standard implementations
but are optimized using numpy einsum, broadcasting, and stride tricks.
"""
import numpy as np


def hamilton_einsum(p, q):
    """Hamilton product using einsum for arbitrary leading dimensions.

    Equivalent to _hamilton(p, q) but uses a single einsum call.
    """
    # The Hamilton product constants as a (4,4,4) tensor:
    # H[r,c,k] where out[...,r] = sum_{c,k} H[r,c,k] * p[...,c] * q[...,k]
    H = np.zeros((4, 4, 4))
    H[0, 0, 0] = 1;    H[0, 1, 1] = -1;   H[0, 2, 2] = -1;   H[0, 3, 3] = -1
    H[1, 0, 1] = 1;    H[1, 1, 0] = 1;    H[1, 2, 3] = -1;   H[1, 3, 2] = 1
    H[2, 0, 2] = 1;    H[2, 1, 3] = 1;    H[2, 2, 0] = 1;    H[2, 3, 1] = -1
    H[3, 0, 3] = 1;    H[3, 1, 2] = -1;   H[3, 2, 1] = 1;    H[3, 3, 0] = 1
    # Assume p, q have matching leading dims or are broadcastable
    return np.einsum('rck,...c,...k->...r', H, p, q, optimize=True)


def quat_matmul(A_data, B_data):
    """Vectorized quaternion matrix multiplication.

    Args:
        A_data: (m, k, 4) ndarray
        B_data: (k, n, 4) ndarray

    Returns:
        (m, n, 4) ndarray
    """
    return np.einsum('rck,mkc,knc->mnr', _H_TENSOR, A_data, B_data, optimize=True)


_H_TENSOR = np.zeros((4, 4, 4))
_H_TENSOR[0, 0, 0] = 1;    _H_TENSOR[0, 1, 1] = -1;   _H_TENSOR[0, 2, 2] = -1;   _H_TENSOR[0, 3, 3] = -1
_H_TENSOR[1, 0, 1] = 1;    _H_TENSOR[1, 1, 0] = 1;    _H_TENSOR[1, 2, 3] = -1;   _H_TENSOR[1, 3, 2] = 1
_H_TENSOR[2, 0, 2] = 1;    _H_TENSOR[2, 1, 3] = 1;    _H_TENSOR[2, 2, 0] = 1;    _H_TENSOR[2, 3, 1] = -1
_H_TENSOR[3, 0, 3] = 1;    _H_TENSOR[3, 1, 2] = -1;   _H_TENSOR[3, 2, 1] = 1;    _H_TENSOR[3, 3, 0] = 1


def conjugate_batch(data):
    """Element-wise quaternion conjugate for (..., 4) data."""
    mask = np.array([1., -1., -1., -1.])
    return data * mask


def norm_squared_batch(data):
    """Element-wise quaternion norm squared for (..., 4) data."""
    return (data * data).sum(axis=-1)


def normalize_batch(data):
    """Batch normalize quaternion vectors. data: (..., 4)."""
    norms = np.sqrt((data * data).sum(axis=-1, keepdims=True))
    norms = np.where(norms == 0, 1., norms)
    return data / norms
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
python -m pytest tests/test_optimized.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add quat/optimized.py tests/test_optimized.py
git commit -m "feat: add quat.optimized — einsum-based vectorized operations"
```

---

### Task 8: algorithms/ — Kernels & Solver Migration

**Files:**
- Create: `quat/algorithms/__init__.py`
- Create: `quat/algorithms/kernels.py` (migrate from `kernels.py`)
- Create: `quat/algorithms/solver.py` (migrate from `qsmm.py`)
- Create: `tests/test_kernels.py`
- Create: `tests/test_solver.py`

- [ ] **Step 1: Update quat/algorithms/__init__.py**

```python
"""Quaternion learning algorithms — kernel methods and ADMM solver."""
from quat.algorithms.kernels import (
    cubic_kernel_matrix,
    cubic_kernel_cross,
    gaussian_kernel_matrix,
    gaussian_kernel_cross,
    normalize_kernel,
    compute_kernel_matrix,
    compute_kernel_cross,
)
from quat.algorithms.solver import (
    solve_ksqmm,
    predict_ksqmm,
)
```

- [ ] **Step 2: Copy kernels.py to quat/algorithms/kernels.py**

Copy entire content of `kernels.py`, update imports:
```python
from quat.algebra import _hamilton, _CONJ
from quat.collections import QuatTensor, QuatMatrix
```
All other code unchanged.

- [ ] **Step 3: Copy qsmm.py to quat/algorithms/solver.py**

Copy entire content of `qsmm.py`, update imports:
```python
from quat.core import Quaternion
from quat.collections import QuatMatrix
from quat.algebra import _hamilton, _CONJ, _RW
```
All other code unchanged.

- [ ] **Step 4: Write tests/test_kernels.py**

```python
"""Tests for quaternion kernel computations."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestCubicKernel(QuatTestCase):
    def test_shape(self):
        from quat.algorithms.kernels import cubic_kernel_matrix
        from quat.collections import QuatTensor
        X = QuatTensor(np.random.randn(5, 3, 4, 4))
        K = cubic_kernel_matrix(X)
        self.assertEqual(K.shape, (5, 5, 4))

    def test_identity_property(self):
        from quat.algorithms.kernels import cubic_kernel_matrix
        from quat.collections import QuatTensor
        X = QuatTensor(np.zeros((3, 1, 1, 4)))
        K = cubic_kernel_matrix(X)
        self.assertTrue(np.allclose(K, np.ones((3, 3, 4))))


class TestGaussianKernel(QuatTestCase):
    def test_shape(self):
        from quat.algorithms.kernels import gaussian_kernel_matrix
        from quat.collections import QuatTensor
        X = QuatTensor(np.random.randn(4, 2, 2, 4))
        K = gaussian_kernel_matrix(X)
        self.assertEqual(K.shape, (4, 4, 4))

    def test_self_kernel(self):
        from quat.algorithms.kernels import gaussian_kernel_matrix
        from quat.collections import QuatTensor
        X = QuatTensor(np.zeros((2, 1, 1, 4)))
        K = gaussian_kernel_matrix(X)
        self.assertTrue(np.allclose(K, np.ones((2, 2, 4))))


class TestNormalizeKernel(QuatTestCase):
    def test_range(self):
        from quat.algorithms.kernels import normalize_kernel
        K = np.random.randn(10, 10, 4) * 5
        K_norm = normalize_kernel(K)
        self.assertLessEqual(np.abs(K_norm).max(), 1.0)
```

- [ ] **Step 5: Write tests/test_solver.py**

```python
"""Tests for Q-ADMM solver."""
import unittest
import numpy as np
from tests.conftest import QuatTestCase


class TestProxRamp(QuatTestCase):
    def test_prox_ramp_basic(self):
        from quat.algorithms.solver import _prox_ramp
        t = np.array([[2.0, 0.5, -0.1, 3.0]])
        beta = np.array([0.5, 0.5, 0.5, 0.5])
        result = _prox_ramp(t, beta)
        self.assertEqual(result.shape, (1, 4))


class TestPredict(QuatTestCase):
    def test_predict_shape(self):
        from quat.algorithms.solver import predict_ksqmm
        K_test = np.ones((3, 5, 4))
        alpha = np.ones((5, 4))
        b = np.zeros(4)
        y_pred = predict_ksqmm(K_test, alpha, b)
        self.assertEqual(y_pred.shape, (3, 4))


class TestSolveKSQMM(QuatTestCase):
    def test_solve_smoke(self):
        from quat.algorithms.solver import solve_ksqmm
        from quat.algorithms.kernels import cubic_kernel_matrix
        from quat.collections import QuatTensor
        N = 10
        X = QuatTensor(np.random.randn(N, 1, 1, 4))
        K = cubic_kernel_matrix(X)
        y = np.ones((N, 4))
        y[5:, 0] = -1
        alpha, b, info = solve_ksqmm(K, y, max_iter=50, tol=1e-3, verbose=False)
        self.assertEqual(alpha.shape, (N, 4))
        self.assertEqual(b.shape, (4,))
        self.assertIn('n_iter', info)
```

- [ ] **Step 6: Run tests**

```powershell
python -m pytest tests/test_kernels.py tests/test_solver.py -v
```

Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add quat/algorithms/ quat/algorithms/kernels.py quat/algorithms/solver.py tests/test_kernels.py tests/test_solver.py
git commit -m "feat: migrate algorithms — kernels + ADMM solver to quat.algorithms"
```

---

### Task 9: data/ — Data Module Migration

**Files:**
- Create: `quat/data/__init__.py` (migrate from `data/__init__.py`)
- Create: `quat/data/converter.py` (migrate from `data/converter.py`)
- Create: `quat/data/loader.py` (migrate from `data/loader.py`)

- [ ] **Step 1: Copy data/__init__.py to quat/data/__init__.py**

Update import paths:
```python
"""Quaternion SMM Data Module"""
from quat.data.loader import (
    load_cifar10_samples,
    load_stl10_samples,
    load_flower17_samples,
    load_imagenette_samples,
    load_svhn_samples,
)
```

- [ ] **Step 2: Copy data/converter.py to quat/data/converter.py**

No imports reference `quaternion` module, so no import changes needed (uses only numpy).

- [ ] **Step 3: Copy data/loader.py to quat/data/loader.py**

Update lazy imports on lines 163, 237, 317, 422, 513 — change `from quaternion import ...` to:
```python
from quat.core import Quaternion
from quat.collections import dict_to_quat_tensor, labels_to_quat_vector
```

- [ ] **Step 4: Verify imports work**

```powershell
python -c "from quat.data import load_cifar10_samples; print('data module OK')"
```

Expected: `data module OK`

- [ ] **Step 5: Commit**

```bash
git add quat/data/
git commit -m "feat: migrate data module to quat.data"
```

---

### Task 10: Integration — __init__.py, Demo Updates, Final Verification

**Files:**
- Modify: `quat/__init__.py`
- Modify: `demo_quaternion.py`
- Modify: `demo_real_matrix.py`

- [ ] **Step 1: Write full quat/__init__.py**

```python
"""Quaternion Algebra Library — quat package.

Provides:
  Quaternion  — single quaternion value
  QuatVector  — 1-d collection of quaternions
  QuatMatrix  — 2-d quaternion matrix
  QuatTensor  — 3-d quaternion tensor

  quat()      — convenience constructor
  dict_to_quat_matrix, dict_to_quat_tensor, labels_to_quat_vector

Algebra primitives (from quat.algebra):
  _hamilton, _CONJ, _RW, _I, _J, _K, _ZERO, _R

Utilities (from quat.utils):
  to_ndarray, from_ndarray, batch_quat, stack_quat

Serialization (from quat.serialization):
  to_json, from_json, to_bytes, from_bytes,
  as_ndarray, to_scipy_rotation, from_scipy_rotation

Optimized (from quat.optimized):
  hamilton_einsum, quat_matmul, conjugate_batch,
  norm_squared_batch, normalize_batch

Algorithms (from quat.algorithms):
  cubic_kernel_matrix, gaussian_kernel_matrix,
  compute_kernel_matrix, compute_kernel_cross, normalize_kernel,
  solve_ksqmm, predict_ksqmm
"""
from quat.algebra import _hamilton, _CONJ, _RW
from quat.core import Quaternion, quat, _I, _J, _K, _ZERO, _R, _ONE
from quat.collections import (
    QuatVector, QuatMatrix, QuatTensor,
    dict_to_quat_matrix, dict_to_quat_tensor, labels_to_quat_vector,
)
from quat.utils import to_ndarray, from_ndarray, batch_quat, stack_quat
from quat.serialization import (
    to_json, from_json, to_bytes, from_bytes,
    as_ndarray, to_scipy_rotation, from_scipy_rotation,
)
from quat.optimized import (
    hamilton_einsum, quat_matmul,
    conjugate_batch, norm_squared_batch, normalize_batch,
)
from quat.algorithms import (
    cubic_kernel_matrix, cubic_kernel_cross,
    gaussian_kernel_matrix, gaussian_kernel_cross,
    normalize_kernel, compute_kernel_matrix, compute_kernel_cross,
    solve_ksqmm, predict_ksqmm,
)

__all__ = [
    'Quaternion', 'QuatVector', 'QuatMatrix', 'QuatTensor',
    'quat', 'dict_to_quat_matrix', 'dict_to_quat_tensor', 'labels_to_quat_vector',
    '_hamilton', '_CONJ', '_RW', '_I', '_J', '_K', '_ZERO', '_R',
    'to_ndarray', 'from_ndarray', 'batch_quat', 'stack_quat',
    'to_json', 'from_json', 'to_bytes', 'from_bytes',
    'as_ndarray', 'to_scipy_rotation', 'from_scipy_rotation',
    'hamilton_einsum', 'quat_matmul',
    'conjugate_batch', 'norm_squared_batch', 'normalize_batch',
    'cubic_kernel_matrix', 'cubic_kernel_cross',
    'gaussian_kernel_matrix', 'gaussian_kernel_cross',
    'normalize_kernel', 'compute_kernel_matrix', 'compute_kernel_cross',
    'solve_ksqmm', 'predict_ksqmm',
]
```

- [ ] **Step 2: Update demo_quaternion.py imports**

Change line 5 from:
```python
from quaternion import (
    Quaternion, QuatVector, QuatMatrix, QuatTensor,
    quat, _I, _J, _K, _ZERO, _ONE, _R,
)
```
to:
```python
from quat import (
    Quaternion, QuatVector, QuatMatrix, QuatTensor,
    quat, _I, _J, _K, _ZERO, _R,
)
```

Note: `_ONE` is removed (it was `Quaternion(1,1,1,1)` defined in quaternion.py line 1046). Add `_ONE = Quaternion(1, 1, 1, 1)` locally in demo, or replace references.

- [ ] **Step 3: Update demo_real_matrix.py imports**

Change line 5 from:
```python
from quaternion import Quaternion, QuatMatrix, _RW
```
to:
```python
from quat import Quaternion, QuatMatrix, _RW
```

- [ ] **Step 4: Run all tests**

```powershell
python -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 5: Run demo scripts to verify end-to-end**

```powershell
python demo_quaternion.py
python demo_real_matrix.py
```

Expected: both run without errors, output matches expected quaternion algebra behavior.

- [ ] **Step 6: Commit**

```bash
git add quat/__init__.py demo_quaternion.py demo_real_matrix.py
git commit -m "feat: complete integration — __init__.py exports, demo updates"
```

---

### Task 11: Final Cleanup & Verification

- [ ] **Step 1: Run full test suite one final time**

```powershell
python -m pytest tests/ -v --tb=short
```

Expected: all tests PASS, no warnings

- [ ] **Step 2: Verify import compatibility**

```powershell
python -c "from quat import Quaternion, QuatVector, QuatMatrix, QuatTensor; print('All imports OK')"
python -c "from quat.algorithms import cubic_kernel_matrix, solve_ksqmm; print('Algorithms OK')"
python -c "from quat.data import load_cifar10_samples; print('Data module OK')"
python -c "from quat import to_json, from_json, to_bytes, from_bytes; print('Serialization OK')"
python -c "from quat import to_ndarray, from_ndarray, batch_quat, stack_quat; print('Utils OK')"
python -c "from quat import hamilton_einsum, quat_matmul; print('Optimized OK')"
```

Expected: all print "OK"

- [ ] **Step 3: Commit final state**

```bash
git add -A
git commit -m "chore: final verification — all imports and tests pass"
```
