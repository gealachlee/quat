"""08_conversions -- type conversions and representations."""
import numpy as np
from quat import (
    Quaternion, QuatVector, QuatMatrix, QuatTensor,
    _I, _J, _K, _R, _ZERO,
    to_ndarray, from_ndarray, from_components, stack_quat,
    dict_to_quat_matrix, dict_to_quat_tensor,
    _REAL_LEFT,
)

HR = "=" * 72
SR = "-" * 72


def section(title):
    print(f"\n{HR}\n{title}\n{HR}")


def subsection(title):
    print(f"\n{SR}\n  {title}\n{SR}")


# =========================================================================
# 1. Quaternion <-> Matrix Representations
# =========================================================================

section("1. Quaternion <-> Matrix Representations")

q = Quaternion(1, 2, 3, 4)
print(f"  q                  = {q}")

subsection("1.1 Complex 2x2 representation")
M_c = q.to_complex_matrix()
q_back = Quaternion.from_complex_matrix(M_c)
print(f"  M_c(q) =\n{M_c}")
print(f"  roundtrip          = {q == q_back}")
print(f"  det(M_c)           = {np.linalg.det(M_c):.6f}")
print(f"  |q|^2              = {q.norm_squared():.6f}")
print(f"  det(M_c) = |q|^2?  = {np.allclose(np.linalg.det(M_c), q.norm_squared())}")

subsection("1.2 Real 4x4 left-representation")
M_l = q.to_real_matrix_left()
q_back_l = Quaternion.from_real_matrix_left(M_l)
print(f"  M_left(q) =\n{M_l}")
print(f"  roundtrip          = {q == q_back_l}")

x = Quaternion(5, 6, 7, 8)
Lx = M_l @ x._data
qx = (q * x)._data
print(f"  L(q) @ vec(x)      = {Lx}")
print(f"  vec(q * x)         = {qx}")
print(f"  match?             = {np.allclose(Lx, qx)}")

subsection("1.3 Real 4x4 right-representation")
M_r = q.to_real_matrix_right()
Rx = M_r @ x._data
xq = (x * q)._data
print(f"  R(q) @ vec(x)      = {Rx}")
print(f"  vec(x * q)         = {xq}")
print(f"  match?             = {np.allclose(Rx, xq)}")

subsection("1.4 Left vs Right representation difference")
print(f"  M_left(q) =\n{M_l}")
print(f"  M_right(q) =\n{M_r}")
print(f"  same?              = {np.allclose(M_l, M_r)}")

# =========================================================================
# 2. QuatVector <-> Matrix Representations
# =========================================================================

section("2. QuatVector <-> Matrix Representations")

v = QuatVector([_I, _J, _K, _R])
print(f"  v                  = {v}")
print(f"  v.shape            = {v.shape}")

subsection("2.1 Complex matrix (2n x 2)")
M_cv = v.to_complex_matrix()
print(f"  M_c(v) shape       = {M_cv.shape}")
print(f"  M_c(v) =\n{M_cv}")

subsection("2.2 Real matrix left (4n x 4)")
M_rv = v.to_real_matrix_left()
print(f"  M_left(v) shape    = {M_rv.shape}")
print(f"  M_left(v) =\n{M_rv}")

subsection("2.3 from_real_matrix_left roundtrip")
v_back = QuatVector.from_real_matrix_left(M_rv)
print(f"  roundtrip          = {np.allclose(v.to_array(), v_back.to_array())}")

subsection("2.4 Vector of length 1 (single quaternion)")
v1 = QuatVector([Quaternion(1, 2, 3, 4)])
M_c1 = v1.to_complex_matrix()
print(f"  M_c([q]) shape     = {M_c1.shape}")
print(f"  equals q.to_complex_matrix()? = {np.allclose(M_c1, q.to_complex_matrix())}")

# =========================================================================
# 3. QuatMatrix <-> Matrix Representations
# =========================================================================

section("3. QuatMatrix <-> Matrix Representations")

A = QuatMatrix([[_I, _J], [_K, _R]])
print(f"  A (2x2)            = {A}")

subsection("3.1 Complex matrix (2m x 2n)")
M_cm = A.to_complex_matrix()
print(f"  M_c(A) shape       = {M_cm.shape}")
print(f"  M_c(A) =\n{M_cm}")

subsection("3.2 from_complex_matrix roundtrip")
A_back = QuatMatrix.from_complex_matrix(M_cm)
print(f"  roundtrip          = {np.allclose(A.to_array(), A_back.to_array())}")

subsection("3.3 Real matrix left (4m x 4n)")
M_rm = A.to_real_matrix_left()
print(f"  M_left(A) shape    = {M_rm.shape}")
print(f"  M_left(A) =\n{M_rm}")

subsection("3.4 from_real_matrix_left roundtrip")
A_back_r = QuatMatrix.from_real_matrix_left(M_rm)
print(f"  roundtrip          = {np.allclose(A.to_array(), A_back_r.to_array())}")

subsection("3.5 Verify matrix multiplication preservation")
B = QuatMatrix([[_R, _K], [_J, _I]])
AB = A * B
M_ab = AB.to_real_matrix_left()
M_a = A.to_real_matrix_left()
M_b = B.to_real_matrix_left()
print(f"  M(A) @ M(B) ~ M(AB)? = {np.allclose(M_a @ M_b, M_ab)}")

subsection("3.6 Non-square matrix")
A35 = QuatMatrix(np.random.randn(3, 5, 4))
M_c35 = A35.to_complex_matrix()
M_r35 = A35.to_real_matrix_left()
print(f"  A shape            = {A35.shape}")
print(f"  M_c shape          = {M_c35.shape}")
print(f"  M_left shape       = {M_r35.shape}")

# =========================================================================
# 4. QuatTensor <-> Matrix Representations
# =========================================================================

section("4. QuatTensor <-> Matrix Representations")

T = QuatTensor([
    [[_I, _J], [_K, _R]],
    [[_R, _ZERO], [_J, _I]],
])
print(f"  T shape            = {T.shape}")

subsection("4.1 Complex matrix via unfolding")
for mode in (1, 2, 3):
    M_ct = T.to_complex_matrix(mode=mode)
    print(f"  M_c(T, mode={mode})  shape = {M_ct.shape}")

subsection("4.2 Real matrix left via unfolding")
for mode in (1, 2, 3):
    M_rt = T.to_real_matrix_left(mode=mode)
    print(f"  M_left(T, mode={mode}) shape = {M_rt.shape}")

# =========================================================================
# 5. NumPy Interop
# =========================================================================

section("5. NumPy Interop")

q = Quaternion(1, 2, 3, 4)
v = QuatVector(np.random.randn(5, 4))
M = QuatMatrix(np.random.randn(3, 4, 4))
T = QuatTensor(np.random.randn(2, 3, 4, 4))

subsection("5.1 to_ndarray / from_ndarray (universal conversion)")
for label, obj in [("Quaternion", q), ("QuatVector", v),
                   ("QuatMatrix", M), ("QuatTensor", T)]:
    arr = to_ndarray(obj)
    obj_back = from_ndarray(arr)
    print(f"  {label:<12s} -> {str(arr.shape):<16s} -> {type(obj_back).__name__}")

subsection("5.2 np.asarray / to_array")
arr_q = np.asarray(q)
arr_v = np.asarray(v)
arr_M = np.asarray(M)
arr_T = np.asarray(T)
print(f"  np.asarray(q)      shape = {arr_q.shape}")
print(f"  np.asarray(v)      shape = {arr_v.shape}")
print(f"  np.asarray(M)      shape = {arr_M.shape}")
print(f"  np.asarray(T)      shape = {arr_T.shape}")

subsection("5.3 numpy array protocol (__array__)")
arr_v2 = np.asarray(v)
print(f"  np.asarray(v)      shape = {arr_v2.shape}")
print(f"  same as to_array?  = {np.allclose(arr_v2, v.to_array())}")

# =========================================================================
# 6. Build from Components
# =========================================================================

section("6. Build from Components")

r = np.array([1., 2., 3.])
i = np.array([4., 5., 6.])
j = np.array([7., 8., 9.])
k = np.array([0., 1., 2.])

qv = from_components(r, i, j, k)
print(f"  from_components    = {qv}")
print(f"  shape              = {qv.shape}")
print(f"  real               = {qv.real}")
print(f"  i                  = {qv.i}")
print(f"  j                  = {qv.j}")
print(f"  k                  = {qv.k}")

# =========================================================================
# 7. Stacking
# =========================================================================

section("7. Stacking")

qs = [Quaternion(1, 0, 0, 0), Quaternion(0, 1, 0, 0), Quaternion(0, 0, 1, 0)]
stacked = stack_quat(qs)
print(f"  stack_quat(3 quats)= {stacked}")
print(f"  shape              = {stacked.shape}")
print(f"  type               = {type(stacked).__name__}")

stacked_axis1 = stack_quat([QuatVector([_I, _J]), QuatVector([_K, _R])], axis=1)
print(f"  stack_quat(vecs, axis=1) shape = {stacked_axis1.shape}")

# =========================================================================
# 8. Dict Conversions
# =========================================================================

section("8. Dict Conversions")

X_dict = {
    'real': np.arange(6.).reshape(2, 3),
    'i': np.ones((2, 3)),
    'j': np.zeros((2, 3)),
    'k': np.full((2, 3), -1.),
}
Qm = dict_to_quat_matrix(X_dict)
print(f"  dict_to_quat_matrix shape = {Qm.shape}")
print(f"  Qm[0,0]            = {Qm[0, 0]}")

X_dict_batch = {
    'real': np.arange(12.).reshape(2, 2, 3),
    'i': np.ones((2, 2, 3)),
    'j': np.zeros((2, 2, 3)),
    'k': np.full((2, 2, 3), -1.),
}
Qt = dict_to_quat_tensor(X_dict_batch)
print(f"  dict_to_quat_tensor shape = {Qt.shape}")
print(f"  Qt[0,0,0]          = {Qt[0, 0, 0]}")

# =========================================================================
# 9. _REAL_LEFT tensor structure
# =========================================================================

section("9. _REAL_LEFT Tensor (4x4x4)")

print(f"  _REAL_LEFT is the left-regular real-representation tensor:")
print(f"  L(q)[r,c] = sum_k _REAL_LEFT[r,c,k] * q[k]")
print(f"  _REAL_LEFT shape = {_REAL_LEFT.shape}")
print()
print(f"  Non-zero entries:")
for r in range(4):
    for c in range(4):
        coeffs = _REAL_LEFT[r, c]
        nonzero = [(k, v) for k, v in enumerate(coeffs) if v != 0]
        if nonzero:
            terms = " + ".join(
                f"{'' if v > 0 else '-'}q[{k}]" if abs(v) == 1
                else f"{v}*q[{k}]"
                for k, v in nonzero)
            print(f"    L[{r},{c}] = {terms}")
