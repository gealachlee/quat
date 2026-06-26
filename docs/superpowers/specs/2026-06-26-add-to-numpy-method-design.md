# Spec: Add `to_numpy()` Instance Method

**Date:** 2026-06-26
**Status:** approved

## Motivation

Add a `to_numpy()` instance method on all four quaternion types (`Quaternion`, `QuatVector`, `QuatMatrix`, `QuatTensor`) for idiomatic numpy conversion with zero-copy support.

## Design

Add identical method to all four types in `core.py` and `collections.py`:

```python
def to_numpy(self, copy: bool = True, dtype=None) -> np.ndarray:
    if copy is False and dtype is None:
        return self._data
    return np.array(self._data, dtype=dtype, copy=copy)
```

### Behavior

| Call | Returns |
|------|---------|
| `q.to_numpy()` | copy of `_data` |
| `q.to_numpy(copy=False)` | internal `_data` reference (zero-copy) |
| `q.to_numpy(dtype=np.float32)` | new array with specified dtype |

### Files to modify

1. `quat/core.py` — add `to_numpy()` to `Quaternion`
2. `quat/collections.py` — add `to_numpy()` to `QuatVector`, `QuatMatrix`, `QuatTensor`
3. `tests/test_core.py` — add tests for `Quaternion.to_numpy()`
4. `tests/test_collections.py` — add tests for collection `to_numpy()`

No changes to `__init__.py` needed (instance methods don't require export).

## Test plan

- `to_numpy()` default (copy=True): returns correct shape and values, verify independence from original
- `to_numpy(copy=False)`: returns same object as `_data` (identity check), verify mutation affects original
- `to_numpy(dtype=...)`: returns correct dtype, verify independent copy
- Test on all four types: `Quaternion`, `QuatVector`, `QuatMatrix`, `QuatTensor`
