# Design Patterns Refactoring — Spec

**Date:** 2026-06-27
**Status:** approved

## Motivation

The quat library (~2,200 lines) has significant code duplication across
`QuatVector`, `QuatMatrix`, and `QuatTensor` (collectively ~40% overlapping methods)
and a hardcoded multiplication-strategy threshold. The goal is to reduce
duplication without introducing over-abstraction.

## Design

### 1. Shared helper modules (two new private modules)

**`quat/_checks.py`** — numeric validation helpers, used by collection classes.

```python
def _vec_isnan(data):   return np.any(np.isnan(data), axis=-1)
def _vec_isinf(data):   return np.any(np.isinf(data), axis=-1)
def _vec_isfinite(data): return np.all(np.isfinite(data), axis=-1)
def _vec_isclose(data, other_data, rtol, atol):
    return np.isclose(data, other_data, rtol=rtol, atol=atol).all(axis=-1)
```

Each collection class delegates:

```python
def isnan(self):
    return _vec_isnan(self._data)
```

**`quat/_arrayops.py`** — NumPy interop helpers.

- `_to_array(data)` / `_to_numpy(data, copy, dtype)` — shared data extraction
- `_dispatch_collection_ufunc(self, ufunc, method, *inputs, **kwargs)` — the
  entire `__array_ufunc__` dispatch logic, identical for all three collection types

Each class's `__array_ufunc__` becomes a one-liner.

Quaternion **does not** use these — its 1-d data and different ufunc set
(e.g. `np.conjugate`, `np.absolute`) are intentionally kept separate.

### 2. Serialization unification

**`quat/_serialize.py`** — shared serialization primitives.

Four functions handle the common JSON/binary logic:

| function | purpose |
|---|---|
| `_serialize_to_json(type_name, data)` | data → `'{"type":"...","data":[...]}'` |
| `_deserialize_from_json(s, cls_map)` | JSON → class instance |
| `_serialize_to_bytes(type_id, data)` | data → type_id header + shape + float64 |
| `_deserialize_from_bytes(b, cls_map)` | bytes → class instance |

Type-id/name registry lives in the same module:

```python
_TYPE_ID_MAP = {0: "Quaternion", 1: "QuatVector", 2: "QuatMatrix", 3: "QuatTensor"}
```

Each class's `to_json` / `to_bytes` become single-line calls. Standalone module
functions (`serialization.py`) delegate through the same helpers.

### 3. Multiplication kernel strategies

Insert a **three-tier dispatch** into `_hamilton` in `algebra.py`:

| tier | condition | kernel |
|---|---|---|
| small | total elements ≤ 500 | `_hamilton_component` (existing) |
| medium | 500 < total ≤ 5000 | `_hamilton_tensordot` (new) |
| large | total > 5000 | `_hamilton_einsum` (existing) |

The new medium tier uses `np.tensordot` with `_HAMILTON_TENSOR` to avoid
einsum's optimisation overhead while staying fully vectorised.

Thresholds are module-level constants; future work could auto-calibrate
at import time with a micro-benchmark.

### Impact summary

| metric | before | after | delta |
|---|---|---|---|
| total lines (quat/) | ~2,200 | ~2,100 | -100 |
| `__array_ufunc__` duplication | 3 copies | 1 shared | -2 copies |
| `isnan/isinf/isfinite/isclose` | 3 copies each | 1 shared + delegation | -8 copies |
| serialization code paths | 4 independent | 1 shared + 4 thin wrappers | -3 paths |
| Hamilton product strategies | 2 | 3 | +1 strategy |

### Non-goals

- No public API changes (all current imports remain valid).
- No abstract base class — the four types have meaningfully different shapes,
  arithmetic rules, and indexing semantics; forcing them into one hierarchy
  would create more complexity than it removes.
- No external dependency additions.

### Files changed

| file | change |
|---|---|
| `quat/_checks.py` | **new** — ~15 lines |
| `quat/_arrayops.py` | **new** — ~40 lines |
| `quat/_serialize.py` | **new** — ~45 lines |
| `quat/algebra.py` | add `_hamilton_tensordot`, three-tier dispatch — +20 / -5 |
| `quat/collections.py` | delegate to helpers, remove duplicated methods — -120 lines |
| `quat/serialization.py` | delegate to `_serialize` helpers — -30 lines |
| `tests/` | no changes (behaviour is identical) |

### Tests

All existing tests must pass without modification. The refactoring is
purely structural — no behavioural change.
