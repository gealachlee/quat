# Design Patterns Refactoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate duplicated methods across QuatVector/QuatMatrix/QuatTensor, unify serialization, and add a three-tier Hamilton product dispatch.

**Architecture:** Three new private modules (`_checks.py`, `_arrayops.py`, `_serialize.py`) hold shared helpers; collection classes delegate to them. `_hamilton` gains a `tensordot`-based middle tier. Public API unchanged — all imports, signatures, and return types identical.

**Tech Stack:** Python 3.9+, numpy 1.21+

## Global Constraints

- Public API unchanged — `from quat import ...` must work identically
- All existing tests must pass without modification (no behavioral change)
- No new external dependencies
- No abstract base class hierarchy
- Quaternion's binary format (simple, no shape prefix) stays unchanged

---

### Task 1: Shared validation helpers (`_checks.py`)

**Files:**
- Create: `quat/_checks.py`
- Modify: `quat/collections.py`

**Interfaces:**
- Produces: `_vec_isnan(data)`, `_vec_isinf(data)`, `_vec_isfinite(data)`, `_vec_isclose(data, other_data, rtol, atol)` — all accept `(..., 4)` ndarray

- [ ] **Step 1: Create `quat/_checks.py`**

```python
"""Validation helpers for quaternion collection types."""
from __future__ import annotations
import numpy as np


def _vec_isnan(data: np.ndarray) -> np.ndarray:
    return np.any(np.isnan(data), axis=-1)


def _vec_isinf(data: np.ndarray) -> np.ndarray:
    return np.any(np.isinf(data), axis=-1)


def _vec_isfinite(data: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(data), axis=-1)


def _vec_isclose(data: np.ndarray, other_data: np.ndarray,
                 rtol: float, atol: float) -> np.ndarray:
    return np.isclose(data, other_data, rtol=rtol, atol=atol).all(axis=-1)
```

- [ ] **Step 2: Add import to `quat/collections.py`**

Add this line after the existing imports:

```python
from quat._checks import _vec_isnan, _vec_isinf, _vec_isfinite, _vec_isclose
```

- [ ] **Step 3: Replace QuatVector `isnan`/`isinf`/`isfinite`/`isclose`**

Replace lines currently containing `def isnan(self)` ... `def isclose(self)` under `QuatVector` (approximately lines 218-231):

```python
    def isnan(self) -> np.ndarray:
        return _vec_isnan(self._data)

    def isinf(self) -> np.ndarray:
        return _vec_isinf(self._data)

    def isfinite(self) -> np.ndarray:
        return _vec_isfinite(self._data)

    def isclose(self, other: QuatVector, rtol: float = 1e-05, atol: float = 1e-08) -> np.ndarray:
        if len(self) != len(other):
            raise ValueError("Size mismatch")
        return _vec_isclose(self._data, other._data, rtol, atol)
```

- [ ] **Step 4: Replace QuatMatrix `isnan`/`isinf`/`isfinite`/`isclose`**

Replace lines currently containing `def isnan(self)` ... `def isclose(self)` under `QuatMatrix`:

```python
    def isnan(self) -> np.ndarray:
        return _vec_isnan(self._data)

    def isinf(self) -> np.ndarray:
        return _vec_isinf(self._data)

    def isfinite(self) -> np.ndarray:
        return _vec_isfinite(self._data)

    def isclose(self, other: QuatMatrix, rtol: float = 1e-05, atol: float = 1e-08) -> np.ndarray:
        if self.shape != other.shape:
            raise ValueError("Shape mismatch")
        return _vec_isclose(self._data, other._data, rtol, atol)
```

- [ ] **Step 5: Replace QuatTensor `isnan`/`isinf`/`isfinite`/`isclose`**

Replace lines currently containing `def isnan(self)` ... `def isclose(self)` under `QuatTensor`:

```python
    def isnan(self) -> np.ndarray:
        return _vec_isnan(self._data)

    def isinf(self) -> np.ndarray:
        return _vec_isinf(self._data)

    def isfinite(self) -> np.ndarray:
        return _vec_isfinite(self._data)

    def isclose(self, other: QuatTensor, rtol: float = 1e-05, atol: float = 1e-08) -> np.ndarray:
        if self.shape != other.shape:
            raise ValueError("Shape mismatch")
        return _vec_isclose(self._data, other._data, rtol, atol)
```

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_collections.py -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add quat/_checks.py quat/collections.py
git commit -m "refactor: extract validation helpers to quat._checks"
```

---

### Task 2: Shared NumPy interop helpers (`_arrayops.py`)

**Files:**
- Create: `quat/_arrayops.py`
- Modify: `quat/collections.py`

**Interfaces:**
- Produces: `_to_array(data)`, `_to_numpy(data, copy, dtype)`, `_dispatch_collection_ufunc(self, ufunc, method, *inputs, **kwargs)`

- [ ] **Step 1: Create `quat/_arrayops.py`**

```python
"""NumPy interop helpers for quaternion collection types."""
from __future__ import annotations
import numpy as np


def _data_copy(data: np.ndarray) -> np.ndarray:
    return data.copy()


def _to_numpy(data: np.ndarray, copy: bool = True,
              dtype: np.dtype | None = None) -> np.ndarray:
    if copy is False and dtype is None:
        return data
    return np.array(data, dtype=dtype, copy=copy)


def _to_array(data: np.ndarray) -> np.ndarray:
    return data.copy()


def _dispatch_collection_ufunc(self, ufunc, method, *inputs, **kwargs):
    """Shared __array_ufunc__ dispatch for QuatVector/QuatMatrix/QuatTensor."""
    if method != '__call__' or kwargs.get('out') is not None:
        return NotImplemented
    a, b = (inputs[0], inputs[1]) if len(inputs) == 2 else (inputs[0], None)
    if ufunc is np.add:
        return a + b
    if ufunc is np.subtract:
        return a - b
    if ufunc is np.multiply:
        return a * b
    if ufunc is np.true_divide or ufunc is np.floor_divide:
        return a / b
    if ufunc is np.negative:
        return -a
    return NotImplemented
```

- [ ] **Step 2: Add import to `quat/collections.py`**

Add after existing imports:

```python
from quat._arrayops import _data_copy, _to_array, _to_numpy, _dispatch_collection_ufunc
```

- [ ] **Step 3: Replace QuatVector's interop methods**

Replace QuatVector's `data` property, `to_array`, `to_numpy`, `__array__`, and `__array_ufunc__` methods:

```python
    @property
    def data(self) -> np.ndarray:
        return _data_copy(self._data)

    def to_array(self) -> np.ndarray:
        return _to_array(self._data)

    def to_numpy(self, copy: bool = True, dtype=None) -> np.ndarray:
        return _to_numpy(self._data, copy=copy, dtype=dtype)

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        if copy is False and dtype is None:
            return self._data
        return np.array(self._data, dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _dispatch_collection_ufunc(self, ufunc, method, *inputs, **kwargs)
```

- [ ] **Step 4: Replace QuatMatrix's interop methods**

Replace QuatMatrix's `data` property, `to_array`, `to_numpy`, `__array__`, and `__array_ufunc__` methods:

```python
    @property
    def data(self) -> np.ndarray:
        return _data_copy(self._data)

    def to_array(self) -> np.ndarray:
        return _to_array(self._data)

    def to_numpy(self, copy: bool = True, dtype=None) -> np.ndarray:
        return _to_numpy(self._data, copy=copy, dtype=dtype)

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        if copy is False and dtype is None:
            return self._data
        return np.array(self._data, dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _dispatch_collection_ufunc(self, ufunc, method, *inputs, **kwargs)
```

- [ ] **Step 5: Replace QuatTensor's interop methods**

Replace QuatTensor's `data` property, `to_array`, `to_numpy`, `__array__`, and `__array_ufunc__` methods:

```python
    @property
    def data(self) -> np.ndarray:
        return _data_copy(self._data)

    def to_array(self) -> np.ndarray:
        return _to_array(self._data)

    def to_numpy(self, copy: bool = True, dtype=None) -> np.ndarray:
        return _to_numpy(self._data, copy=copy, dtype=dtype)

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        if copy is False and dtype is None:
            return self._data
        return np.array(self._data, dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _dispatch_collection_ufunc(self, ufunc, method, *inputs, **kwargs)
```

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add quat/_arrayops.py quat/collections.py
git commit -m "refactor: extract numpy interop helpers to quat._arrayops"
```

---

### Task 3: Serialization unification (`_serialize.py`)

**Files:**
- Create: `quat/_serialize.py`
- Modify: `quat/collections.py`
- Modify: `quat/serialization.py`

**Note:** Quaternion's `to_json`/`from_json`/`to_bytes`/`from_bytes` in `core.py` are left unchanged (already minimal). The standalone `serialization.py` module-level functions are updated.

**Interfaces:**
- Produces: `_serialize_to_json(type_name, data)`, `_deserialize_from_json(s, cls_map)`, `_serialize_bytes_shaped(type_id, data)`, `_deserialize_bytes_shaped(b, cls_map)`

- [ ] **Step 1: Create `quat/_serialize.py`**

```python
"""Shared serialization primitives for quaternion types."""
from __future__ import annotations
import json as _json
import struct as _struct
import numpy as np

_TYPE_IDS = {0: "Quaternion", 1: "QuatVector", 2: "QuatMatrix", 3: "QuatTensor"}
_TYPE_NAMES = {v: k for k, v in _TYPE_IDS.items()}


def _serialize_to_json(type_name: str, data: np.ndarray) -> str:
    return _json.dumps({"type": type_name, "data": data.tolist()})


def _deserialize_from_json(s: str, cls_map: dict):
    d = _json.loads(s)
    return cls_map[d["type"]](np.array(d["data"], dtype=float))


def _serialize_bytes_shaped(type_id: int, data: np.ndarray) -> bytes:
    """Binary format for collection types: type_id + ndim + shape + float64."""
    data64 = data.astype(np.float64)
    shape = np.array(data64.shape, dtype=np.int32)
    return _struct.pack('<ii', type_id, len(shape)) + shape.tobytes() + data64.tobytes()


def _deserialize_bytes_shaped(b: bytes, cls_map: dict):
    """Deserialize binary for collection types."""
    type_id, ndim = _struct.unpack_from('<ii', b, 0)
    offset = 8
    shape = np.frombuffer(b[offset:offset + ndim * 4], dtype=np.int32)
    offset += ndim * 4
    size = int(np.prod(shape))
    data = np.frombuffer(b[offset:offset + size * 8], dtype=np.float64).reshape(shape)
    return cls_map[type_id](data)
```

- [ ] **Step 2: Update QuatVector serialization in `quat/collections.py`**

Add import after existing imports:

```python
from quat._serialize import _serialize_to_json, _serialize_bytes_shaped
```

Replace QuatVector's `to_json` and `to_bytes` methods:

```python
    def to_json(self) -> str:
        return _serialize_to_json("QuatVector", self._data)

    def to_bytes(self) -> bytes:
        return _serialize_bytes_shaped(1, self._data)

    @classmethod
    def from_json(cls, s: str) -> "QuatVector":
        from quat._serialize import _deserialize_from_json
        return _deserialize_from_json(s, {"QuatVector": cls})

    @classmethod
    def from_bytes(cls, b: bytes) -> "QuatVector":
        from quat._serialize import _deserialize_bytes_shaped
        return _deserialize_bytes_shaped(b, {1: cls})
```

- [ ] **Step 3: Update QuatMatrix serialization in `quat/collections.py`**

Replace QuatMatrix's `to_json`, `to_bytes`, `from_json`, `from_bytes` methods:

```python
    def to_json(self) -> str:
        return _serialize_to_json("QuatMatrix", self._data)

    def to_bytes(self) -> bytes:
        return _serialize_bytes_shaped(2, self._data)

    @classmethod
    def from_json(cls, s: str) -> "QuatMatrix":
        from quat._serialize import _deserialize_from_json
        return _deserialize_from_json(s, {"QuatMatrix": cls})

    @classmethod
    def from_bytes(cls, b: bytes) -> "QuatMatrix":
        from quat._serialize import _deserialize_bytes_shaped
        return _deserialize_bytes_shaped(b, {2: cls})
```

- [ ] **Step 4: Update QuatTensor serialization in `quat/collections.py`**

Replace QuatTensor's `to_json`, `to_bytes`, `from_json`, `from_bytes` methods:

```python
    def to_json(self) -> str:
        return _serialize_to_json("QuatTensor", self._data)

    def to_bytes(self) -> bytes:
        return _serialize_bytes_shaped(3, self._data)

    @classmethod
    def from_json(cls, s: str) -> "QuatTensor":
        from quat._serialize import _deserialize_from_json
        return _deserialize_from_json(s, {"QuatTensor": cls})

    @classmethod
    def from_bytes(cls, b: bytes) -> "QuatTensor":
        from quat._serialize import _deserialize_bytes_shaped
        return _deserialize_bytes_shaped(b, {3: cls})
```

- [ ] **Step 5: Update standalone `quat/serialization.py`**

Replace the `to_json`, `from_json`, `to_bytes`, `from_bytes` module-level functions:

```python
"""Serialization and interop — thin wrappers over quaternion object methods.

The primary API is via instance and class methods on the quaternion types:

    q = Quaternion(1,2,3,4)
    s = q.to_json()            # instance → JSON string
    q = Quaternion.from_json(s)  # class method ← JSON string
    b = q.to_bytes()           # instance → bytes
    q = Quaternion.from_bytes(b) # class method ← bytes

Standalone functions below delegate to these methods and accept
Quaternion, QuatVector, QuatMatrix, or QuatTensor.
"""
from __future__ import annotations
import json as _json
import struct as _struct
import numpy as np
from quat.core import Quaternion
from quat.collections import QuatVector, QuatMatrix, QuatTensor


def to_json(q: Quaternion | QuatVector | QuatMatrix | QuatTensor) -> str:
    return q.to_json()


def from_json(s: str) -> Quaternion | QuatVector | QuatMatrix | QuatTensor:
    d = _json.loads(s)
    cls_map = {
        "Quaternion": Quaternion,
        "QuatVector": QuatVector,
        "QuatMatrix": QuatMatrix,
        "QuatTensor": QuatTensor,
    }
    return cls_map[d["type"]](np.array(d["data"], dtype=float))


def to_bytes(q: Quaternion | QuatVector | QuatMatrix | QuatTensor) -> bytes:
    return q.to_bytes()


def from_bytes(b: bytes) -> Quaternion | QuatVector | QuatMatrix | QuatTensor:
    type_id = _struct.unpack_from('<i', b, 0)[0]
    cls_map = {
        0: Quaternion,
        1: QuatVector,
        2: QuatMatrix,
        3: QuatTensor,
    }
    return cls_map[type_id].from_bytes(b)


def to_scipy_rotation(q: Quaternion | QuatVector):
    """Convert to ``scipy.spatial.transform.Rotation``.

    scipy uses ``(x, y, z, w)``; our order is ``(w, x, y, z)`` = ``(r, i, j, k)``.

    Example:
        >>> from quat import Quaternion, to_scipy_rotation
        >>> import numpy as np
        >>> q = Quaternion.from_axis_angle((0, 0, 1), np.pi / 2)
        >>> rot = to_scipy_rotation(q)
        >>> rot.as_quat()
        array([0.        , 0.        , 0.70710678, 0.70710678])
    """
    from scipy.spatial.transform import Rotation
    from quat.utils import to_ndarray
    data = to_ndarray(q)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return Rotation.from_quat(data[:, [1, 2, 3, 0]])


def from_scipy_rotation(rot) -> Quaternion | QuatVector:
    """Convert ``scipy.spatial.transform.Rotation`` back to quaternion(s)."""
    scipy_quat = rot.as_quat()
    if scipy_quat.ndim == 1:
        scipy_quat = scipy_quat.reshape(1, -1)
    data = np.zeros((scipy_quat.shape[0], 4))
    data[:, 0] = scipy_quat[:, 3]
    data[:, 1] = scipy_quat[:, 0]
    data[:, 2] = scipy_quat[:, 1]
    data[:, 3] = scipy_quat[:, 2]
    if data.shape[0] == 1:
        return Quaternion(data[0])
    return QuatVector(data)
```

- [ ] **Step 6: Verify Quaternion's own methods still work**

Run: `python -c "from quat.core import Quaternion; q=Quaternion(1,2,3,4); s=q.to_json(); q2=Quaternion.from_json(s); print(q==q2)"`
Expected: `True`

- [ ] **Step 7: Run all serialization tests**

```bash
python -m pytest tests/test_serialization.py -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add quat/_serialize.py quat/collections.py quat/serialization.py
git commit -m "refactor: extract serialization helpers to quat._serialize"
```

---

### Task 4: Three-tier Hamilton product dispatch

**Files:**
- Modify: `quat/algebra.py`

**Interfaces:**
- Produces: `_hamilton(p, q)` — unchanged signature, now dispatches to 3 internal kernels
- Internal: `_hamilton_component(p, q)`, `_hamilton_tensordot(p, q)` (einsum optimize=False), `_hamilton_einsum(p, q)` (einsum optimize=True)

- [ ] **Step 1: Replace `_hamilton` and constants in `quat/algebra.py`**

The affected section starts at line 31 (`_EINSUM_THRESHOLD = 2000`) through the end of the file (line 68). Replace the entire block from line 31 onward:

```python
_SMALL_THRESHOLD = 500
_LARGE_THRESHOLD = 5000


def _hamilton(p: npt.NDArray, q: npt.NDArray) -> npt.NDArray:
    """Vectorized Hamilton (quaternion) product.

    Dispatches to the optimal kernel based on data size:
      - small (<=500 elements): component-wise arithmetic
      - medium (500-5000): einsum without contraction-path optimisation
      - large (>5000): einsum with full contraction-path optimisation

    Supports arbitrary leading-dimension broadcasting.
    """
    total_elements = p.size + q.size
    if total_elements <= _SMALL_THRESHOLD:
        return _hamilton_component(p, q)
    if total_elements <= _LARGE_THRESHOLD:
        return _hamilton_tensordot(p, q)
    return _hamilton_einsum(p, q)


def _hamilton_component(p: npt.NDArray, q: npt.NDArray) -> npt.NDArray:
    """Component-wise Hamilton product — fastest for small batches."""
    a1, b1, c1, d1 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    a2, b2, c2, d2 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    shp = np.broadcast_shapes(p.shape[:-1], q.shape[:-1]) + (4,)
    out = np.empty(shp)
    out[..., 0] = a1*a2 - b1*b2 - c1*c2 - d1*d2
    out[..., 1] = a1*b2 + b1*a2 + c1*d2 - d1*c2
    out[..., 2] = a1*c2 - b1*d2 + c1*a2 + d1*b2
    out[..., 3] = a1*d2 + b1*c2 - c1*b2 + d1*a2
    return out


def _hamilton_tensordot(p: npt.NDArray, q: npt.NDArray) -> npt.NDArray:
    """Einsum without contraction-path optimisation — faster for medium batches
    where the optimisation overhead outweighs its benefit."""
    return np.einsum('rck,...c,...k->...r', _HAMILTON_TENSOR, p, q, optimize=False)


def _hamilton_einsum(p: npt.NDArray, q: npt.NDArray) -> npt.NDArray:
    """Einsum with full contraction-path optimisation — best throughput for large
    batches where the optimisation cost is amortised."""
    return np.einsum('rck,...c,...k->...r', _HAMILTON_TENSOR, p, q, optimize=True)
```

- [ ] **Step 2: Run core algebra tests**

```bash
python -m pytest tests/test_algebra.py -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run the full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS (including any test that uses Hamilton product via `*` operator).

- [ ] **Step 4: Run benchmarks to verify performance**

```bash
python -m pytest tests/test_benchmark.py -v
```

Expected: benchmarks complete (pytest-benchmark required). Performance should be at least as good as before for all size tiers.

- [ ] **Step 5: Commit**

```bash
git add quat/algebra.py
git commit -m "perf: add three-tier Hamilton product dispatch (component/tensordot/einsum)"
```

---

### Task 5: Final verification

- [ ] **Step 1: Run full test suite one final time**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all tests PASS.

- [ ] **Step 2: Verify imports still work**

```bash
python -c "from quat import Quaternion, QuatVector, QuatMatrix, QuatTensor, quat, _I, _J, _K, _R, _ZERO, _ONE_Q; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit any remaining cleanup**

```bash
git status
```

If clean, done.
