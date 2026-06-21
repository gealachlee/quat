# Design Spec: quat Package Refactor

Date: 2026-06-21

## Goal

Refactor the Quaternion project from a flat collection of `.py` files into a proper Python package `quat`, with comprehensive unit tests, serialization, optimized numpy vectorization, and data conversion utilities.

## Decisions

| Decision | Choice |
|----------|--------|
| Approach | Full package restructuring (方案B) |
| Package name | `quat` |
| Test framework | `unittest` (stdlib) |
| Performance | Pure numpy vectorization (no numba/Cython) |
| Utils scope | Data conversion helpers only (no SLERP/Euler) |
| Serialization | JSON, binary bytes, scipy Rotation interop, numpy interop |
| Algorithms | Separate `quat.algorithms` sub-package |

## Package Structure

```
quat/
├── __init__.py          # Re-export all public API
├── algebra.py           # _hamilton, _RW, _CONJ, _I/_J/_K, _ZERO/_R/_ONE
├── core.py              # Quaternion class + quat() factory
├── collections.py       # QuatVector, QuatMatrix, QuatTensor
├── utils.py             # [NEW] Data conversion helpers
├── serialization.py     # [NEW] JSON/bytes serialization, scipy Rotation, numpy interop
├── optimized.py         # [NEW] Pure numpy vectorized optimizations
├── algorithms/          # [NEW] Algorithm sub-package
│   ├── __init__.py
│   ├── kernels.py       # Migrated from kernels.py
│   └── solver.py        # Migrated from qsmm.py
└── data/
    ├── __init__.py
    ├── converter.py     # Migrated from data/converter.py
    └── loader.py        # Migrated from data/loader.py
```

## Module Responsibilities

### algebra.py
- `_hamilton(p, q)` — vectorized Hamilton product
- `_RW` — (4,4,4) weight tensor for real matrix representation
- `_CONJ` — conjugate mask
- `_I`, `_J`, `_K`, `_ZERO`, `_R`, `_ONE` — basis constants

### core.py
- `Quaternion` class (__slots__, full arithmetic, algebra, rotation, matrix representations)
- `quat()` — convenience constructor

### collections.py
- `QuatVector` — 1D array of quaternions
- `QuatMatrix` — 2D matrix of quaternions
- `QuatTensor` — 3D tensor of quaternions

### utils.py [NEW]
- `to_ndarray(q)` — quaternion object(s) → (..., 4) ndarray
- `from_ndarray(arr)` — (..., 4) ndarray → quaternion object(s)
- `batch_quat(*args)` — batch construction from component arrays
- `broadcast_shapes(*shapes)` — broadcast shape computation for quaternion collections

### serialization.py [NEW]
- `to_json(q)` / `from_json(s)` — JSON serialization
- `to_bytes(q)` / `from_bytes(b)` — binary serialization
- `to_scipy_rotation(q)` / `from_scipy_rotation(r)` — scipy.spatial.transform.Rotation interop
- `as_ndarray(q)` — optimized ndarray view/conversion (numpy interop)

### optimized.py [NEW]
- Pure numpy vectorized alternatives for core operations using einsum/broadcasting
- Same API signatures as the standard versions
- Primary targets: Hamilton product, matrix multiplication, kernel computation

### algorithms/kernels.py
- Migrated from `kernels.py`
- Q-cubic and Q-Gaussian kernel functions
- public API unchanged

### algorithms/solver.py
- Migrated from `qsmm.py`
- Q-ADMM solver with ramp loss
- public API unchanged

## Test Structure

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures (random quaternion generators)
├── test_core.py         # Quaternion: constructors, arithmetic, algebra, matrix reps, edge cases
├── test_collections.py  # QuatVector, QuatMatrix, QuatTensor: all operations
├── test_algebra.py      # _hamilton (shapes, broadcasting), _RW correctness, constants
├── test_utils.py        # to_ndarray, from_ndarray, batch_quat, broadcast_shapes
├── test_serialization.py # Round-trip tests for JSON, bytes, scipy Rotation
├── test_optimized.py    # Correctness comparison: optimized vs standard
├── test_kernels.py      # Kernel matrix computation, cross-kernel, edge cases
└── test_solver.py       # ADMM convergence, prediction accuracy
```

## Key Constraints

1. **Zero new dependencies** beyond existing (numpy, scipy, Pillow, scikit-learn for data module)
2. **API stability** — all existing public method signatures unchanged
3. **Import compatibility** — `from quat import Quaternion` works; demo scripts updated via path changes only
4. **No SLERP/Euler** — utils.py is data conversion only
5. **No numba/Cython** — optimized.py uses pure numpy (einsum, broadcasting, stride tricks)

## Migration Notes

- `quaternion.py` → split into `algebra.py`, `core.py`, `collections.py`
- `qsmm.py` → `quat/algorithms/solver.py`
- `kernels.py` → `quat/algorithms/kernels.py`
- `data/` → `quat/data/`
- Demo scripts: update imports from `from quaternion import X` to `from quat import X`
