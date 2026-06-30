# Changelog

## [1.0.0b1] - 2026-07

### Added
- **Distance metrics**: `rotation.intrinsic`, `rotation.chordal`, `rotor.intrinsic`, `rotor.chordal` with batch variants (`quat.stats`)
- **Kinematics**: `angular_velocity()`, `integrate_angular_velocity()`, `rotate_frame()` for quaternion trajectory estimation (`quat.interpolate`)
- **Mean rotation**: `mean_rotation()` (SVD-based closed-form) and `approximate_karcher_mean()` (iterative Riemannian) (`quat.stats`)
- **Component-space statistics**: `quaternion_mean()`, `quaternion_cov()`, `quaternion_pca()` — exclusive module (`quat.stats`)
- **Quaternion**: `.w` property, `.normalized` property alias, `.angle` and `.axis` rotation properties, `.minimal()` canonical representation; immutable `_data`
- **QuatVector**: element-wise Hamilton product (`v * w` Hadamard)
- **linalg**: `svd_values()` public fast path via complex representation (~5-10x faster than full SVD)

### Changed
- **Breaking:** `QuatVector.__mul__` now handles `QuatVector * QuatVector` as element-wise Hamilton product
- **Architecture:** Extracted `_BaseCollection` base class; merged `_arrayops/_checks/_serialize` into `collections.py`, `optimized` → `algebra.py`, `kinematics` → `interpolate.py`, `distance + mean` → `stats.py`; 16 files → 11 files
- `karcher_mean` renamed to `approximate_karcher_mean`

### Fixed
- `det()` mathematical bug (double-negation)
- `condition_number` crash on zero matrix
- Angular velocity spike for identical consecutive quaternions
- `floor_divide` incorrectly mapped to true division
- Global constant arrays now read-only (`writeable=False`)
- Stale docstring references to deleted modules

### Performance
- Einsum contraction-path caching (`lru_cache` on `_hamilton_einsum`)
- SVD singular-values-only fast path exposed as public API

### Version
- `0.2.0` → `1.0.0b1`, Development Status: `3 - Alpha` → `4 - Beta`

## [0.2.0] - 2026-06-29

Initial public release.
