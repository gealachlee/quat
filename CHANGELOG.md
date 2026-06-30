# Changelog

## [1.0.0b1] - 2026-07

### Added
- **quat.distance**: 4 distance metrics (`rotation.intrinsic`, `rotation.chordal`, `rotor.intrinsic`, `rotor.chordal`) with batch variants
- **quat.kinematics**: `angular_velocity()`, `integrate_angular_velocity()`, `rotate_frame()` for quaternion trajectory estimation
- **quat.mean**: `mean_rotation()` (SVD-based closed-form) and `karcher_mean()` (iterative Riemannian)
- **quat.stats**: `quaternion_mean()`, `quaternion_cov()`, `quaternion_pca()` — exclusive module for quaternion component-space statistics
- **Quaternion**: `.w` property, `.normalized` property alias, `.angle` and `.axis` rotation properties, `.minimal()` canonical representation
- **QuatVector**: element-wise Hamilton product (`v * w` Hadamard)
- **linalg**: `svd_values()` public fast path via complex representation (~5-10x faster than full SVD)

### Performance
- Einsum contraction-path caching (`lru_cache` on `_hamilton_einsum`) — reduces repeated path discovery overhead
- SVD singular-values-only fast path exposed as public API

### Changed
- **Breaking:** `QuatVector.__mul__` now handles `QuatVector * QuatVector` as element-wise Hamilton product (previously only scalar/Quaternion multiplication)
- Package version bumped to `1.0.0b1`
- Development Status: `3 - Alpha` → `4 - Beta`

## [0.2.0] - 2026-06-29

Initial public release.
