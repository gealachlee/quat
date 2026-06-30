<p align="center">
  <h1 align="center">quat</h1>
  <p align="center"><b>Quaternion Algebra for NumPy</b></p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python">
  <img src="https://img.shields.io/badge/numpy-%E2%89%A51.21-4d77cf" alt="NumPy">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
</p>

---

**quat** brings first-class quaternion support to NumPy.  Scalar quaternions,
vector/matrix/tensor collections, SVD, SLERP interpolation, QFFT signal
processing, serialization — with a three-tier-einsum Hamilton kernel that
picks the fastest strategy at every data size.  All in pure Python, zero C
extensions.

```
pip install quat-numpy
```

## Why quat?

Most quaternion libraries stop at rotating 3D vectors.  quat gives you the
full toolbox:

| What | How |
|---|---|
| **Scalar quaternions** | Full algebra — `+`, `*`, `/`, `exp`, `log`, `pow`, Hamilton product |
| **Collections** | `QuatVector` (1D), `QuatMatrix` (2D), `QuatTensor` (3D) — broadcast-aware |
| **Linear algebra** | SVD, rank, condition number, pseudo-inverse, determinant — on quaternion matrices |
| **Interpolation** | SLERP (shortest arc on S³), squad cubic spline, batch SLERP, angular velocity estimation |
| **Distance & statistics** | 4 geodesic/chordal distance metrics, quaternion vector mean, Karcher mean, PCA |
| **Signal processing** | 1D/2D QFFT, quaternion convolution, FIR filter design |
| **Random generation** | Reproducible generators for all four types |
| **Serialization** | JSON + compact binary roundtrip for all types; SciPy `Rotation` interop |
| **Performance** | Three-tier Hamilton kernel (component-wise → einsum), `~20x` faster SVD fast-path |

## Quickstart

```python
import numpy as np
from quat import Quaternion, QuatVector, QuatMatrix
from quat import slerp, squad, random_unit_quat

# ---- Construction -------------------------------------------------
q = Quaternion(1, 2, 3, 4)                         # a + bi + cj + dk
r = Quaternion(5.0)                                 # pure real
u = Quaternion.from_axis_angle((0,0,1), np.pi/2)    # 90 deg around z
e = Quaternion.from_euler((0.1, 0.2, 0.3))          # intrinsic ZYX

# ---- Hamilton product ---------------------------------------------
i, j, k = Quaternion(0,1,0,0), Quaternion(0,0,1,0), Quaternion(0,0,0,1)
print(i * j)           # Quaternion(0, 0, 0, 1)   — i*j = k
print(i * j * k)       # Quaternion(-1, 0, 0, 0)  — i² = j² = k² = ijk = -1

# ---- Algebra ------------------------------------------------------
q.conjugate()          # Quaternion(1, -2, -3, -4)
q.norm()               # Euclidean norm
q.inverse() * q        # ≈ identity
q.exp()                # quaternion exponential
q.log()                # quaternion logarithm

# ---- 3D rotation --------------------------------------------------
u = Quaternion.from_axis_angle((0, 0, 1), np.pi / 2)
u.rotate_vector((1, 0, 0))           # ≈ (0, 1, 0) — 90° rotation
axis, angle = u.to_axis_angle()      # roundtrip (axis, angle)

# ---- Smooth interpolation -----------------------------------------
a = Quaternion(1, 0, 0, 0)
b = Quaternion(0, 1, 0, 0)
mid = slerp(a, b, 0.5)               # shortest arc on the 3-sphere

q0, q1, q2, q3 = [random_unit_quat() for _ in range(4)]
curve = squad(q0, q1, q2, q3, 0.75)  # cubic spline (Shoemake 1987)

# ---- Collections --------------------------------------------------
v = QuatVector([
    Quaternion(1,0,0,0), Quaternion(0,1,0,0), Quaternion(0,0,1,0)
])
v.real          # array([1., 0., 0.])
v.data.shape    # (3, 4)

A = QuatMatrix.eye(3)
B = QuatMatrix(np.random.randn(3, 4, 4))
C = A * B       # quaternion matrix multiply
C.shape         # (3, 4)

# ---- Linear algebra -----------------------------------------------
from quat.linalg import svd, pseudo_inverse, solve

U, s, Vh = svd(C)
A_pinv = pseudo_inverse(C)
x = solve(A_pinv, QuatVector(np.ones((3, 4))))

# ---- Signal processing --------------------------------------------
from quat.signal import qfft, qconv, lowpass

X = qfft(np.random.randn(256, 4))            # 1D quaternion FFT
k = lowpass(16, cutoff=0.2)                  # FIR lowpass filter
y = qconv(np.random.randn(128, 4), k._data)  # quaternion convolution

# ---- Serialization ------------------------------------------------
s = q.to_json()         # → '{"type":"Quaternion","data":[...]}'
b = q.to_bytes()        # → compact binary
q2 = Quaternion.from_json(s)
q3 = Quaternion.from_bytes(b)

# ---- Basis constants ----------------------------------------------
from quat import _I, _J, _K, _R, _ZERO
print(_I * _J * _K)     # -1 (Hamilton's fundamental identity)
```

## Multiplication

All `*` operations in quat use the **Hamilton product** (not component-wise
multiplication).  The only exception is scalar multiplication (`* s`), which
multiplies each of the 4 components by a real/complex number.

### Multiplication is always Hamilton — never component-wise

```
q1 * q2  →  Hamilton product  q₁·q₂
```

There is **no** `*` operator that multiplies real parts together and imaginary
parts separately.  Every non-scalar `*` invokes the full quaternion Hamilton
product in some form.

### Two families

| family | forms | what it does |
|---|---|---|
| **Scalar multiply** | `q * s`, `v * s`, `A * s`, `T * s` | Each component × scalar (real × real, i × i, j × j, k × k) |
| **Hamilton multiply** | everything else below | Full quaternion product |

### Four semantic patterns

#### ① `Quat × Quat` — Hamilton product (one result)
```
q1 * q2 → Quaternion
```
Classic quaternion multiplication `(a+bi+cj+dk)(e+fi+gj+hk)`.  Non-commutative.

#### ② `Collection × Quaternion` — element-wise Hamilton
```
v * q   →  v[i]·q          (right-multiply each element)
q * v   →  q·v[i]          (left-multiply each element, via __rmul__)
A * q   →  A[i,j]·q        (right-multiply each element)
q * A   →  q·A[i,j]        (left-multiply each element)
T * q   →  T[i,j,k]·q      (right-multiply each element)
```
Every slot in the collection is multiplied individually by the same quaternion.

#### ③ `Collection × Collection (same shape)` — element-wise Hamilton
```
v1 * v2  →  v1[i]·v2[i]    — QuatVector of same length
```
Each pair `(v1[i], v2[i])` is multiplied via Hamilton product independently.
Requires matching shapes.

#### ④ `Matrix × Matrix / Vector` — quaternion matrix multiplication
```
A * B  →  (A·B)[i,j] = Σₖ A[i,k] · B[k,j]  — QuatMatrix (m×p)
A * v  →  y[i]       = Σⱼ A[i,j] · v[j]     — QuatVector (m)
```
Standard matrix multiplication, but with Hamilton product replacing scalar
multiplication.  **Not** element-wise — equivalent to `C[i,j] = Σₖ A[i,k]·B[k,j]`.

### Cross-type reference

| expression | semantics | result |
|---|---|---|
| `q1 * q2` | Hamilton product | `Quaternion` |
| `q * 2.5` | component-wise scalar × | `Quaternion` |
| `3.0 * q` | scalar × (via `__rmul__`) | `Quaternion` |
| `v1 * v2` | element-wise Hamilton | `QuatVector` |
| `v * q` | element-wise Hamilton, right-multiply | `QuatVector` |
| `q * v` | element-wise Hamilton, left-multiply | `QuatVector` |
| `v * s` | scalar × each element (component-wise) | `QuatVector` |
| `A * B` | **matrix** multiplication (Σ Hamilton) | `QuatMatrix` |
| `A * v` | **matrix-vector** multiplication (Σ Hamilton) | `QuatVector` |
| `A * q` | element-wise Hamilton, right-multiply | `QuatMatrix` |
| `q * A` | element-wise Hamilton, left-multiply | `QuatMatrix` |
| `A * s` | scalar × each element (component-wise) | `QuatMatrix` |
| `T * q` | element-wise Hamilton, right-multiply | `QuatTensor` |
| `T * s` | scalar × each element (component-wise) | `QuatTensor` |
| `T.mode_k_product(A)` | tensor mode-`k` product (`A·ₖ T`) | `QuatTensor` |
| `v1.inner(v2)` | inner product `∑ v[i]⁻ · v[i]` | `Quaternion` |
| `q1.component_wise_mul(q2)` | component-wise `a·e, b·f, c·g, d·h` (no cross-terms) | `Quaternion` |
| `v1.component_wise_mul(v2)` | component-wise, same shape required | `QuatVector` |
| `A.component_wise_mul(B)` | component-wise, same shape required | `QuatMatrix` |
| `component_wise_mul(p, q)` | raw ndarray batch, shape `(..., 4)` | `ndarray` |

### Examples

```python
# ---- ① Scalar × Scalar (Hamilton product) --------------------------
p = Quaternion(2, 3, -1, 4)
q = Quaternion(1, -2, 0, 5)
p * q               # Hamilton product
p.commutator(q)     # [p, q] = pq − qp

# ---- ② Element-wise Hamilton (collection × quaternion) -------------
v = QuatVector([_I, _J, _K])
A = QuatMatrix([[_I, _J], [_K, _R]])
q = Quaternion(0, 1, 0, 0)

v * q               # QuatVector([-1, -k,  j])   — v[i]·q  right-multiply
q * v               # QuatVector([-1,  k, -j])   — q·v[i]  left-multiply
A * q               # each A[i,j]·q

# ---- ③ Element-wise Hamilton (vector × vector) ---------------------
v1 = QuatVector([_I, _J, _K])
v2 = QuatVector([_J, _K, _I])
v1 * v2             # QuatVector([ k,  i,  j])   — element-wise Hamilton

# ---- ④ Matrix multiplication (NOT element-wise) --------------------
A = QuatMatrix([[_I, _J], [_K, _R]])
B = QuatMatrix([[_R, _K], [_J, _I]])
A * B               # shape (2,2)  — C[i,j] = Σ_k A[i,k]·B[k,j]

# ---- ④ Matrix-vector multiplication --------------------------------
A = QuatMatrix(2, 3)
x = QuatVector(3)
A * x               # length-2 vector  — y[i] = Σ_j A[i,j]·x[j]

# ---- Tensor mode-k product -----------------------------------------
T = QuatTensor(2, 3, 4)
T.mode_1_product(A)  # unfold(1) → A * unfolded → fold back
T.mode_2_product(A)
T.mode_3_product(A)

# ---- Inner products -------------------------------------------------
v1.inner(v2)        # Quaternion — sum of v₁[i]⁻ · v₂[i]
T.inner(T)          # Quaternion — Frobenius norm squared

# ---- Component-wise multiply (no cross-terms) -----------------------
q1 = Quaternion(1, 2, 3, 4)
q2 = Quaternion(5, 6, 7, 8)
q1.component_wise_mul(q2)  # Quaternion(5, 12, 21, 32)  — a·e, b·f, c·g, d·h

v1 = QuatVector([_I, _J, _K])
v2 = QuatVector([_J, _K, _I])
v1.component_wise_mul(v2)  # QuatVector([0, 0, 0])  — no cross-terms → all zero
```

### Key points

- **All non-scalar `*` is Hamilton.**  There is no `*` operator for
  component-wise multiply (`a·e, b·f, c·g, d·h`).  Use `.component_wise_mul()`
  instead.
- **`A * B` is matrix multiplication**, not element-wise.  The matrix
  dimensions must be compatible (`m×n` · `n×p`).
- **`A * v` is matrix-vector**, not element-wise.
- **`v1 * v2` is element-wise** — each `v1[i]` is Hamilton-multiplied by
  `v2[i]` independently.
- **Left vs right matters.**  `v * q` (right-multiply) ≠ `q * v` (left-multiply)
  because Hamilton product is non-commutative.
- **`QuatTensor * QuatTensor` is not supported.**  Use mode-`k` products or
  `.inner()` instead.

## API Reference

### `Quaternion` — scalar quaternion

| category | methods |
|---|---|
| **constructors** | `zero()`, `one_q()`, `from_axis_angle()`, `from_euler()`, `from_complex_matrix()`, `from_real_matrix_left()` |
| **components** | `.w`, `.r`, `.i`, `.j`, `.k`, `.real`, `.imag`, `.components`, `.data` |
| **arithmetic** | `+`, `-`, `*`, `/`, `-q` (Hamilton product via `*`; `@` not overridden) |
| **algebra** | `.conjugate()`, `.norm()`, `.normalize()`, `.normalized`, `.inverse()`, `.exp()`, `.log()`, `.pow(t)`, `.re_inner(q)`, `.commutator(q)`, `.minimal()`, `.component_wise_mul(q)` |
| **rotation** | `.rotate_vector(v)`, `.to_axis_angle()`, `.to_euler(seq='zyx')`, `.angle`, `.axis` |
| **validation** | `.isnan()`, `.isinf()`, `.isfinite()`, `.isclose(q)` |
| **serialization** | `.to_json()`, `.from_json(s)`, `.to_bytes()`, `.from_bytes(b)` |
| **matrices** | `.to_complex_matrix()`, `.to_real_matrix_left()`, `.to_real_matrix_right()` |
| **conversion** | `float(q)`, `int(q)`, `complex(q)`, `abs(q)`, `np.asarray(q)` |

### `QuatVector` / `QuatMatrix` / `QuatTensor`

|  | QuatVector | QuatMatrix | QuatTensor |
|---|---|---|---|
| ndim | 1D | 2D | 3D |
| shape | `(n,)` | `(m, n)` | `(p, q, r)` |
| create | `zeros`, `ones` | `zeros`, `eye` | `zeros` |
| access | `.data`, `.real/.i/.j/.k` | `.data`, `.real/.i/.j/.k`, `.row(i)`, `.col(j)` | `.data`, `.real/.i/.j/.k` |
| ops | `.inner(v)`, `.norm()`, `.component_wise_mul(v)` | `.norm()`, `.T`, `.H`, `.conjugate()`, `.component_wise_mul(A)` | `.inner(T)`, `.unfold(mode)`, `.mode_n_product(A)`, `.component_wise_mul(T)` |

### Module overview

| module | exports |
|---|---|
| `quat.random` | `random_quat`, `random_unit_quat`, `random_quat_vector`, `random_quat_matrix`, `random_quat_tensor` |
| `quat.interpolate` | `slerp`, `slerp_vector`, `squad`, `angular_velocity`, `integrate_angular_velocity`, `rotate_frame` |
| `quat.linalg` | `svd`, `svd_values`, `rank`, `condition_number`, `pseudo_inverse`, `trace`, `det`, `norm`, `solve` |
| `quat.serialization` | `to_json`, `from_json`, `to_bytes`, `from_bytes`, `to_scipy_rotation`, `from_scipy_rotation` |
| `quat.signal` | `qfft`, `iqfft`, `qfft2`, `iqfft2`, `qconv`, `qconv2`, `lowpass`, `highpass`, `bandpass`, `bandstop` |
| `quat.stats` | `rotation.intrinsic`, `rotation.chordal`, `rotor.intrinsic`, `rotor.chordal`, `mean_rotation`, `approximate_karcher_mean`, `quaternion_mean`, `quaternion_cov`, `quaternion_pca` |
| `quat.algebra` | `hamilton_einsum`, `quat_matmul`, `component_wise_mul`, `conjugate_batch`, `norm_squared_batch`, `normalize_batch` |

### Performance

The Hamilton product kernel uses a **three-tier dispatch** that selects the
fastest strategy based on data size:

| size | kernel | strategy |
|---|---|---|
| ≤ 500 elements | `_hamilton_component` | direct float arithmetic |
| 500 – 5000 | `_hamilton_einsum_noopt` | einsum without contraction-path optimisation |
| > 5000 | `_hamilton_einsum` | einsum with cached contraction-path optimisation |

**SVD fast-path:** `rank()`, `condition_number()`, and `norm(A, 2)` compute
only singular values via the complex (2×2) representation — **~20× faster**
than the full 4m×4n real SVD.

**Zero-copy interop:** `q.to_numpy(copy=False)` returns the internal buffer
without copying.  `.to_complex_matrix()` and `.to_real_matrix_left()` use
`np.empty` to skip zero-initialisation.

## Install

```bash
pip install quat-numpy
```

Requires **Python ≥ 3.9**, **NumPy ≥ 1.21**.  Optional `scipy` for
`to_scipy_rotation` / `from_scipy_rotation`.

For development:

```bash
git clone https://github.com/gealachlee/quat.git
cd quat
pip install -e .
pytest tests/
```
