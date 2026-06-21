# quat — Quaternion Algebra Library

Pure-Python quaternion algebra library built on numpy.  Provides scalar
quaternions, vector/matrix/tensor collections, linear algebra (SVD, pseudo-inverse),
serialization, and optimized einsum kernels — all respecting quaternion arithmetic
(Hamilton product, non-commutativity).

## Install

```bash
pip install -e .        # editable install for development
```

Requires **Python ≥ 3.9** and **numpy ≥ 1.21**.  Optional: `scipy` for
`Rotation` interop.

## Quickstart

```python
import numpy as np
from quat import Quaternion, QuatVector, QuatMatrix

# -- construction --------------------------------------------------
q = Quaternion(1, 2, 3, 4)               # a + bi + cj + dk
r = Quaternion(5.0)                      # pure real
u = Quaternion.from_axis_angle((0,0,1), np.pi/2)   # 90° around z

print(q.r, q.i, q.j, q.k)               # 1.0  2.0  3.0  4.0
print(q.components)                      # (1.0, 2.0, 3.0, 4.0)

# -- arithmetic (Hamilton product) --------------------------------
i, j, k = Quaternion(0,1,0,0), Quaternion(0,0,1,0), Quaternion(0,0,0,1)
print(i * j)                             # Quaternion(0,0,0,1) — i*j = k
print(i * i)                             # Quaternion(-1,0,0,0) — i² = -1

# -- algebra ------------------------------------------------------
print(q.conjugate())                     # Quaternion(1,-2,-3,-4)
print(q.norm())                          # 5.477...
print(q.inverse() * q)                   # Quaternion(1,0,0,0) — identity
print(q.exp())                           # quaternion exponential

# -- 3D rotation --------------------------------------------------
v = (1.0, 0.0, 0.0)
vr = u.rotate_vector(v)                  # rotate (1,0,0) 90° around z
print(vr)                                # ≈ (0, 1, 0)

# -- collections --------------------------------------------------
v = QuatVector([Quaternion(1,0,0,0), Quaternion(0,1,0,0), Quaternion(0,0,1,0)])
print(v.real, v.i, v.j, v.k)             # component arrays

A = QuatMatrix.eye(3)
B = QuatMatrix(np.random.randn(3, 4, 4))
C = A * B                                # quaternion matrix multiplication
print(C.shape)                           # (3, 4)
print(A.H)                               # conjugate-transpose

# -- linear algebra -----------------------------------------------
from quat.linalg import svd, pseudo_inverse, solve

U, s, Vh = svd(A)
print(s)                                 # singular values
A_pinv = pseudo_inverse(A)
x = solve(A, QuatVector(np.ones((3, 4))))

# -- serialization ------------------------------------------------
s = q.to_json()                          # q → '{"type":"Quaternion","data":[...]}'
b = q.to_bytes()                         # q → compact binary
q2 = Quaternion.from_json(s)             # JSON → Quaternion
q3 = Quaternion.from_bytes(b)            # binary → Quaternion

# -- basis constants ----------------------------------------------
from quat import _I, _J, _K, _R, _ZERO, _ONE_Q
print(_I * _J * _K)                      # -1 (Hamilton's ijk = -1)
```

## API Overview

### `Quaternion` — scalar quaternion

| constructors | `zero()`, `one_q()`, `from_axis_angle()`, `from_complex_matrix()`, `from_real_matrix_left()` |
|---|---|
| **components** | `.r`, `.i`, `.j`, `.k`, `.real`, `.imag`, `.components` |
| **arithmetic** | `+`, `-`, `*`, `/`, `-q` |
| **algebra** | `.conjugate()`, `.norm()`, `.normalize()`, `.inverse()`, `.exp()`, `.log()`, `.pow(t)`, `.re_inner(q)`, `.commutator(q)` |
| **validation** | `.isnan()`, `.isinf()`, `.isfinite()`, `.isclose(q)` |
| **rotation** | `.rotate_vector(v)` |
| **serialization** | `.to_json()`, `.from_json(s)`, `.to_bytes()`, `.from_bytes(b)` |
| **matrix reps** | `.to_complex_matrix()`, `.to_real_matrix_left()`, `.to_real_matrix_right()` |

### `QuatVector` / `QuatMatrix` / `QuatTensor` — collections

|       | QuatVector | QuatMatrix | QuatTensor |
|-------|------------|------------|------------|
| dim   | 1          | 2          | 3          |
| shape | `(n,)`     | `(m,n)`    | `(p,q,r)`  |
| construct | `zeros`, `ones` | `zeros`, `eye` | `zeros` |
| algebra | `.inner(v)`, `.norm()` | `.norm()`, `.T`, `.H`, `.conjugate()` | `.inner(T)`, `.norm()`, `.unfold(mode)`, `.mode_n_product(A)` |

### `quat.linalg` — linear algebra

`svd`, `rank`, `condition_number`, `pseudo_inverse`, `trace`, `det`, `norm`, `solve`

### `quat.serialization`

`to_json`, `from_json`, `to_bytes`, `from_bytes`, `to_scipy_rotation`, `from_scipy_rotation`

### `quat.optimized`

`hamilton_einsum`, `quat_matmul`, `conjugate_batch`, `norm_squared_batch`, `normalize_batch`

### Internals (`quat.algebra`)

`_hamilton(p, q)` — vectorized Hamilton product (the core multiplication kernel).
`_CONJ` — conjugate mask `[1, -1, -1, -1]`.
`_REAL_LEFT` — `(4,4,4)` tensor mapping a quaternion to its 4×4 left-regular
real representation.
