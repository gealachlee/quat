# Add `to_numpy()` Instance Method — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `to_numpy(copy=True, dtype=None)` instance method to `Quaternion`, `QuatVector`, `QuatMatrix`, `QuatTensor`.

**Architecture:** Add an identical 4-line method to all four classes. Internally returns `_data` directly when `copy=False and dtype is None`, otherwise a new array via `np.array()`.

**Tech Stack:** Python, NumPy, unittest

---

### Task 1: Add `to_numpy()` to `Quaternion`

**Files:**
- Modify: `quat/core.py` — add method after `to_array()`
- Modify: `tests/test_core.py` — add test class

- [ ] **Step 1: Write the failing test**

Append to `tests/test_core.py`:

```python
class TestQuaternionToNumpy(QuatTestCase):
    def test_to_numpy_default_copy(self):
        from quat.core import Quaternion
        import numpy as np
        q = Quaternion(1, 2, 3, 4)
        arr = q.to_numpy()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (4,))
        self.assertTrue(np.allclose(arr, [1., 2., 3., 4.]))
        arr[0] = 99.0
        self.assertNotEqual(q.r, 99.0)

    def test_to_numpy_no_copy(self):
        from quat.core import Quaternion
        import numpy as np
        q = Quaternion(1, 2, 3, 4)
        arr = q.to_numpy(copy=False)
        self.assertTrue(arr is q._data)
        arr[0] = 99.0
        self.assertEqual(q.r, 99.0)

    def test_to_numpy_dtype(self):
        from quat.core import Quaternion
        import numpy as np
        q = Quaternion(1, 2, 3, 4)
        arr = q.to_numpy(dtype=np.float32)
        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(np.allclose(arr, [1., 2., 3., 4.]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_core.py::TestQuaternionToNumpy -v`
Expected: FAIL with `'Quaternion' object has no attribute 'to_numpy'`

- [ ] **Step 3: Add `to_numpy()` to `Quaternion`**

In `quat/core.py`, after the `to_array()` method (line 610), add:

```python
    def to_numpy(self, copy: bool = True, dtype=None) -> np.ndarray:
        if copy is False and dtype is None:
            return self._data
        return np.array(self._data, dtype=dtype, copy=copy)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_core.py::TestQuaternionToNumpy -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add quat/core.py tests/test_core.py
git commit -m "feat: add to_numpy() to Quaternion"
```

---

### Task 2: Add `to_numpy()` to `QuatVector`

**Files:**
- Modify: `quat/collections.py` — add method after `to_array()`
- Modify: `tests/test_collections.py` — add test class

- [ ] **Step 1: Write the failing test**

Append to `tests/test_collections.py`:

```python
class TestQuatVectorToNumpy(QuatTestCase):
    def test_to_numpy_default_copy(self):
        from quat.collections import QuatVector
        import numpy as np
        v = QuatVector(np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]))
        arr = v.to_numpy()
        self.assertEqual(arr.shape, (2, 4))
        self.assertTrue(np.allclose(arr, [[1., 2., 3., 4.], [5., 6., 7., 8.]]))
        arr[0, 0] = 99.0
        self.assertNotEqual(v.real[0], 99.0)

    def test_to_numpy_no_copy(self):
        from quat.collections import QuatVector
        import numpy as np
        v = QuatVector(np.array([[1., 2., 3., 4.]]))
        arr = v.to_numpy(copy=False)
        self.assertTrue(arr is v._data)
        arr[0, 0] = 99.0
        self.assertEqual(v.real[0], 99.0)

    def test_to_numpy_dtype(self):
        from quat.collections import QuatVector
        import numpy as np
        v = QuatVector(np.array([[1., 2., 3., 4.]]))
        arr = v.to_numpy(dtype=np.float32)
        self.assertEqual(arr.dtype, np.float32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_collections.py::TestQuatVectorToNumpy -v`
Expected: FAIL with `'QuatVector' object has no attribute 'to_numpy'`

- [ ] **Step 3: Add `to_numpy()` to `QuatVector`**

In `quat/collections.py`, after `QuatVector.to_array()` (line 80), add:

```python
    def to_numpy(self, copy: bool = True, dtype=None) -> np.ndarray:
        if copy is False and dtype is None:
            return self._data
        return np.array(self._data, dtype=dtype, copy=copy)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_collections.py::TestQuatVectorToNumpy -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add quat/collections.py tests/test_collections.py
git commit -m "feat: add to_numpy() to QuatVector"
```

---

### Task 3: Add `to_numpy()` to `QuatMatrix`

**Files:**
- Modify: `quat/collections.py` — add method after `to_array()`
- Modify: `tests/test_collections.py` — add test class

- [ ] **Step 1: Write the failing test**

Append to `tests/test_collections.py`:

```python
class TestQuatMatrixToNumpy(QuatTestCase):
    def test_to_numpy_default_copy(self):
        from quat.collections import QuatMatrix
        import numpy as np
        M = QuatMatrix(np.ones((2, 3, 4)))
        arr = M.to_numpy()
        self.assertEqual(arr.shape, (2, 3, 4))
        arr[0, 0, 0] = 99.0
        self.assertNotEqual(M.real[0, 0], 99.0)

    def test_to_numpy_no_copy(self):
        from quat.collections import QuatMatrix
        import numpy as np
        M = QuatMatrix(np.ones((2, 3, 4)))
        arr = M.to_numpy(copy=False)
        self.assertTrue(arr is M._data)
        arr[0, 0, 0] = 99.0
        self.assertEqual(M.real[0, 0], 99.0)

    def test_to_numpy_dtype(self):
        from quat.collections import QuatMatrix
        import numpy as np
        M = QuatMatrix(np.ones((2, 2, 4)))
        arr = M.to_numpy(dtype=np.float32)
        self.assertEqual(arr.dtype, np.float32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_collections.py::TestQuatMatrixToNumpy -v`
Expected: FAIL with `'QuatMatrix' object has no attribute 'to_numpy'`

- [ ] **Step 3: Add `to_numpy()` to `QuatMatrix`**

In `quat/collections.py`, after `QuatMatrix.to_array()` (line 384), add:

```python
    def to_numpy(self, copy: bool = True, dtype=None) -> np.ndarray:
        if copy is False and dtype is None:
            return self._data
        return np.array(self._data, dtype=dtype, copy=copy)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_collections.py::TestQuatMatrixToNumpy -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add quat/collections.py tests/test_collections.py
git commit -m "feat: add to_numpy() to QuatMatrix"
```

---

### Task 4: Add `to_numpy()` to `QuatTensor`

**Files:**
- Modify: `quat/collections.py` — add method after `to_array()`
- Modify: `tests/test_collections.py` — add test class

- [ ] **Step 1: Write the failing test**

Append to `tests/test_collections.py`:

```python
class TestQuatTensorToNumpy(QuatTestCase):
    def test_to_numpy_default_copy(self):
        from quat.collections import QuatTensor
        import numpy as np
        T = QuatTensor(np.ones((2, 3, 4, 4)))
        arr = T.to_numpy()
        self.assertEqual(arr.shape, (2, 3, 4, 4))
        arr[0, 0, 0, 0] = 99.0
        self.assertNotEqual(T.real[0, 0, 0], 99.0)

    def test_to_numpy_no_copy(self):
        from quat.collections import QuatTensor
        import numpy as np
        T = QuatTensor(np.ones((2, 3, 4, 4)))
        arr = T.to_numpy(copy=False)
        self.assertTrue(arr is T._data)
        arr[0, 0, 0, 0] = 99.0
        self.assertEqual(T.real[0, 0, 0], 99.0)

    def test_to_numpy_dtype(self):
        from quat.collections import QuatTensor
        import numpy as np
        T = QuatTensor(np.ones((2, 2, 2, 4)))
        arr = T.to_numpy(dtype=np.float32)
        self.assertEqual(arr.dtype, np.float32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_collections.py::TestQuatTensorToNumpy -v`
Expected: FAIL with `'QuatTensor' object has no attribute 'to_numpy'`

- [ ] **Step 3: Add `to_numpy()` to `QuatTensor`**

In `quat/collections.py`, after `QuatTensor.to_array()` (line 709), add:

```python
    def to_numpy(self, copy: bool = True, dtype=None) -> np.ndarray:
        if copy is False and dtype is None:
            return self._data
        return np.array(self._data, dtype=dtype, copy=copy)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_collections.py::TestQuatTensorToNumpy -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add quat/collections.py tests/test_collections.py
git commit -m "feat: add to_numpy() to QuatTensor"
```

---

### Task 5: Full test suite verification

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Commit if any cleanup needed, otherwise done**
