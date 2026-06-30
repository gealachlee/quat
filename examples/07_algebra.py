"""07_algebra -- multiplication patterns across all quaternion types."""
import numpy as np
from quat import (
    Quaternion, QuatVector, QuatMatrix, QuatTensor,
    _I, _J, _K, _R, _ZERO,
    hamilton_einsum, quat_matmul,
)

HR = "=" * 66
SR = "-" * 66


def section(title):
    print(f"\n{HR}\n{title}\n{HR}")


def subsection(title):
    print(f"\n{SR}\n  {title}\n{SR}")


# =========================================================================
# 1. Scalar Quaternion * Quaternion (Hamilton product)
# =========================================================================

section("1. Quaternion * Quaternion -- Hamilton Product")

subsection("1.1 Basis identities")
print(f"  _R * _I          = {_R * _I}           (real identity)")
print(f"  _I * _I          = {_I * _I}               (i^2 = -1)")
print(f"  _J * _J          = {_J * _J}               (j^2 = -1)")
print(f"  _K * _K          = {_K * _K}               (k^2 = -1)")
print(f"  _I * _J          = {_I * _J}               (ij =  k)")
print(f"  _J * _K          = {_J * _K}               (jk =  i)")
print(f"  _K * _I          = {_K * _I}               (ki =  j)")
print(f"  _J * _I          = {_J * _I}               (ji = -k)")
print(f"  _I * _J * _K     = {_I * _J * _K}              (ijk = -1)")

subsection("1.2 General Hamilton product")
p = Quaternion(2, 3, -1, 4)
q = Quaternion(1, -2, 0, 5)
print(f"  p               = {p}")
print(f"  q               = {q}")
print(f"  p * q           = {p * q}")
print(f"  q * p           = {q * p}  (non-commutative)")
print(f"  p*q != q*p?     = {(p * q) != (q * p)}")
print(f"  commutator [p,q]= {p.commutator(q)}")

subsection("1.3 Scalar and complex multiplication")
print(f"  3.0 * p         = {3.0 * p}")
print(f"  p * 2.5         = {p * 2.5}")
print(f"  (2+3j) * p      = {(2 + 3j) * p}")
print(f"  p * (1-2j)      = {p * (1 - 2j)}")

subsection("1.4 Inverse and division")
print(f"  p * p^-1        = {p * p.inverse()}          (identity)")
print(f"  p / q           = {p / q}")

subsection("1.5 Powers")
print(f"  p^2             = {p * p}")
print(f"  p.pow(2)        = {p.pow(2)}")
print(f"  p^3             = {p * p * p}")
print(f"  p.pow(3)        = {p.pow(3)}")
print(f"  match?          = {(p * p * p).isclose(p.pow(3))}")

# =========================================================================
# 2. Quaternion with collections
# =========================================================================

section("2. Quaternion with Collections")

subsection("2.1 QuatVector * Quaternion (element-wise)")
v = QuatVector([_I, _J, _K, _R])
q = Quaternion(0, 1, 0, 0)
r = v * q
print(f"  v                = {v}")
print(f"  q                = {q}")
print(f"  v * q            = {r}")
print(f"  check: i*i=-1    = {(v[0] * q).isclose(Quaternion(-1, 0, 0, 0))}")

subsection("2.2 Quaternion * QuatVector (via __rmul__)")
r2 = q * v
print(f"  q * v            = {r2}")
print(f"  q * v[0]         = {q * v[0]}")

subsection("2.3 QuatMatrix * Quaternion (element-wise)")
A = QuatMatrix([[_I, _J], [_K, _R]])
r3 = A * q
print(f"  A                = {A}")
print(f"  A * q            = {r3}")
print(f"  A[0,0]*q         = {A[0, 0] * q}")
print(f"  A[0,1]*q         = {A[0, 1] * q}")

subsection("2.4 Quaternion * QuatMatrix (via __rmul__)")
r4 = q * A
print(f"  q * A            = {r4}")

subsection("2.5 QuatTensor * Quaternion (element-wise)")
T = QuatTensor([[[_I, _J], [_K, _R]], [[_R, _ZERO], [_J, _I]]])
r5 = T * q
print(f"  T shape          = {T.shape}")
print(f"  T[0,0,0] * q     = {(T[0, 0, 0] * q)}")
print(f"  T * q[0,0,0]     = {r5[0, 0, 0]}")
print(f"  match?           = {(T[0, 0, 0] * q).isclose(r5[0, 0, 0])}")

# =========================================================================
# 3. QuatVector * QuatVector (element-wise Hamilton product)
# =========================================================================

section("3. QuatVector * QuatVector -- Element-wise Hamilton")

v1 = QuatVector([_I, _J, _K, _R])
v2 = QuatVector([_J, _K, _I, _ZERO])
r = v1 * v2
print(f"  v1               = {v1}")
print(f"  v2               = {v2}")
print(f"  v1 * v2          = {r}")
print(f"  v1[0]*v2[0]      = {v1[0] * v2[0]}   (i*j =  k)")
print(f"  v1[1]*v2[1]      = {v1[1] * v2[1]}   (j*k =  i)")
print(f"  v1[2]*v2[2]      = {v1[2] * v2[2]}   (k*i =  j)")

subsection("3.1 Vector inner product")
inner = v1.inner(v2)
print(f"  inner(v1, v2)    = {inner}")

subsection("3.2 Scalar multiplication")
print(f"  v1 * 2.5         = {v1 * 2.5}")
print(f"  3.0 * v1         = {3.0 * v1}")

subsection("3.3 Normalization")
u = QuatVector([Quaternion(3, 4, 0, 0), Quaternion(0, 3, 4, 0)])
print(f"  u                = {u}")
print(f"  u.normalize()    = {u.normalize()}")
print(f"  norms            = {[u.normalize()[i].norm() for i in range(2)]}")

# =========================================================================
# 4. QuatMatrix * QuatMatrix (quaternion matrix multiplication)
# =========================================================================

section("4. QuatMatrix * QuatMatrix -- Matrix Multiplication")

A = QuatMatrix([[_I, _J], [_K, _R]])
B = QuatMatrix([[_R, _K], [_J, _I]])
C = A * B
print(f"  A                = {A}")
print(f"  B                = {B}")
print(f"  A * B            = {C}")
print(f"  B * A            = {B * A}  (non-commutative)")

subsection("4.1 Manual verification of one element")
c00 = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]
print(f"  (AB)[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]")
print(f"           = {A[0,0]} * {B[0,0]} + {A[0,1]} * {B[1,0]}")
print(f"           = {A[0,0] * B[0,0]} + {A[0,1] * B[1,0]}")
print(f"           = {c00}")
print(f"  match?           = {c00.isclose(C[0, 0])}")

subsection("4.2 Non-square multiplication")
A42 = QuatMatrix(np.random.randn(4, 2, 4))
A23 = QuatMatrix(np.random.randn(2, 3, 4))
C43 = A42 * A23
print(f"  A42 shape        = {A42.shape}")
print(f"  A23 shape        = {A23.shape}")
print(f"  A42 * A23 shape  = {C43.shape}")

subsection("4.3 Identity matrix")
I3 = QuatMatrix.eye(3)
M = QuatMatrix(np.random.randn(3, 3, 4))
print(f"  I3 * M == M?     = {(I3 * M).isclose(M).all()}")

# =========================================================================
# 5. QuatMatrix * QuatVector (matrix-vector multiplication)
# =========================================================================

section("5. QuatMatrix * QuatVector -- Matrix-Vector")

A = QuatMatrix([[_I, _J, _K], [_R, _ZERO, _I]])
x = QuatVector([_I, _J, _K])
y = A * x
print(f"  A (2x3)          = {A}")
print(f"  x (3,)           = {x}")
print(f"  A * x (2,)       = {y}")
print(f"  y[0] = sum A[0,j]*x[j]")
print(f"       = {A[0,0]*x[0]} + {A[0,1]*x[1]} + {A[0,2]*x[2]}")
print(f"       = {A[0,0]*x[0] + A[0,1]*x[1] + A[0,2]*x[2]}")

# =========================================================================
# 6. QuatMatrix transpose, conjugate, adjoint
# =========================================================================

section("6. Matrix Adjoint and Transpose")

A = QuatMatrix([[_I, _J], [_K, _R]])
print(f"  A                = {A}")
print(f"  A.T (transpose)  = {A.T}")
print(f"  A.H (adjoint)    = {A.H}")
print(f"  conj(A)          = {A.conjugate()}")

# =========================================================================
# 7. QuatTensor arithmetic and mode-n products
# =========================================================================

section("7. QuatTensor -- Arithmetic and Mode Products")

T = QuatTensor([
    [[_I, _J], [_K, _R]],
    [[_R, _ZERO], [_J, _I]],
])
print(f"  T shape          = {T.shape}")
print(f"  T[0]             = {T[0]}")
print(f"  T[0,0,0]         = {T[0, 0, 0]}")

subsection("7.1 Scalar and quaternion multiplication")
print(f"  T * 2.0          = {T * 2.0}")
q = Quaternion(0, 1, 0, 0)
Tq = T * q
print(f"  T * i            = Tq")
print(f"  T[0,0,0]*i       = {T[0, 0, 0] * q}")
print(f"  Tq[0,0,0]        = {Tq[0, 0, 0]}")

subsection("7.2 Mode-1 product (left-multiply first axis)")
U1 = QuatMatrix([[_R, _K], [_J, _R]])
T1 = T.mode_1_product(U1)
print(f"  U1               = {U1}")
print(f"  T.mode_1(U1) shp = {T1.shape}")
print(f"  unfold(1) consis?= {np.allclose(T1.unfold(1).to_array(), (U1 * T.unfold(1)).to_array())}")

subsection("7.3 Mode-2 product")
U2 = QuatMatrix([[_I, _ZERO], [_R, _J]])
T2 = T.mode_2_product(U2)
print(f"  T.mode_2(U2) shp = {T2.shape}")

subsection("7.4 Mode-3 product")
U3 = QuatMatrix([[_K, _R], [_ZERO, _I]])
T3 = T.mode_3_product(U3)
print(f"  T.mode_3(U3) shp = {T3.shape}")

subsection("7.5 All three modes sequentially")
result = T.mode_1_product(U1).mode_2_product(U2).mode_3_product(U3)
print(f"  T x1 U1 x2 U2 x3 U3 = {result.shape}")

subsection("7.6 Tensor inner product")
inner_tt = T.inner(T)
print(f"  inner(T,T)       = {inner_tt}")
print(f"  norm(T)^2        = {T.norm()**2:.6f}")
print(f"  match?           = {abs(inner_tt.r - T.norm()**2) < 1e-10}")

subsection("7.7 Tensor unfolding")
for mode in (1, 2, 3):
    unfolded = T.unfold(mode)
    print(f"  unfold({mode})      shape = {unfolded.shape}")

# =========================================================================
# 8. Cross-type multiplication summary
# =========================================================================

section("8. Cross-Type Multiplication Summary")

line = "-" * 20
print(f"  {'Left \\\\ Right':<20s} {'Quaternion':<25s} {'QuatVector':<25s} {'QuatMatrix':<25s}")
print(f"  {line:<20s} {line:<25s} {line:<25s} {line:<25s}")
print(f"  {'Quaternion':<20s} {'-> Quat (hamilton)':<25s} {'':<25s} {'':<25s}")
print(f"  {'QuatVector':<20s} {'-> QuatVec (elem)':<25s} {'-> QuatVec (elem-h)':<25s} {'':<25s}")
print(f"  {'QuatMatrix':<20s} {'-> QuatMat (elem)':<25s} {'-> QuatVec (matvec)':<25s} {'-> QuatMat (matmul)':<25s}")
print(f"  {'QuatTensor':<20s} {'-> QuatTen (elem)':<25s} {'':<25s} {'':<25s}")
print(f"  {'Real/Complex':<20s} {'-> Quat (scalar)':<25s} {'-> QuatVec (scalar)':<25s} {'-> QuatMat (scalar)':<25s}")
print()
print(f"  Note: QuatTensor * QuatTensor not supported; use mode-n products instead.")
print(f"  Note: Real/Complex * collection types work via __rmul__.")

# =========================================================================
# 9. Batch operations (low-level algebra module)
# =========================================================================

section("9. Batch Algebra Operations")

data_p = np.random.randn(10, 4)
data_q = np.random.randn(10, 4)

subsection("9.1 hamilton_einsum (batch Hamilton product)")
r_einsum = hamilton_einsum(data_p, data_q)
print(f"  input shapes     = {data_p.shape}, {data_q.shape}")
print(f"  output shape     = {r_einsum.shape}")

subsection("9.2 quat_matmul (raw matrix multiply)")
A_data = np.random.randn(3, 4, 4)
B_data = np.random.randn(4, 5, 4)
C_data = quat_matmul(A_data, B_data)
print(f"  A shape          = {A_data.shape}")
print(f"  B shape          = {B_data.shape}")
print(f"  C shape          = {C_data.shape}")
print(f"  matches QuatMatrix? = {np.allclose(C_data, (QuatMatrix(A_data) * QuatMatrix(B_data)).to_array())}")

subsection("9.3 Batch conjugate, norm, normalize")
from quat import conjugate_batch, norm_squared_batch, normalize_batch
data = np.random.randn(20, 4)
print(f"  conj shape       = {conjugate_batch(data).shape}")
print(f"  norm^2 shape     = {norm_squared_batch(data).shape}")
print(f"  norm^2[0]        = {norm_squared_batch(data)[0]:.6f}")
n_data = normalize_batch(data)
print(f"  normalized[0]    = {np.linalg.norm(n_data[0]):.6f}")

# =========================================================================
# 10. NumPy ufunc integration with multiplication
# =========================================================================

section("10. NumPy Ufunc Integration")

q1 = Quaternion(1, 2, 3, 4)
q2 = Quaternion(5, 6, 7, 8)

print(f"  np.add(q1, q2)   = {np.add(q1, q2)}")
print(f"  np.multiply(3, q1) = {np.multiply(3.0, q1)}")
print(f"  np.multiply(q1, q2) = {np.multiply(q1, q2)}")
print(f"  np.negative(q1)  = {np.negative(q1)}")

v = QuatVector(np.ones((3, 4)))
print(f"  np.add(v, v)     type = {type(np.add(v, v)).__name__}")
print(f"  np.multiply(v, v) type = {type(np.multiply(v, v)).__name__}")
