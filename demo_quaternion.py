# =========================================================================
#  LLM-generated code.  See README.md for full disclosure.
# =========================================================================
#!/usr/bin/env python3
"""Quaternion Algebra Library --- Demo & Usage Examples"""

import sys; sys.path.insert(0, '/sessions/practical-eloquent-bell/mnt/Quaternion')
from quat import (
    Quaternion, QuatVector, QuatMatrix, QuatTensor,
    quat, _I, _J, _K, _ZERO, _R, _ONE_Q,
)
import numpy as np

HR = '-'*60

def hdr(title):
    print(f"\n{HR}\n  {title}\n{HR}")

# 1. Construction & accessors
hdr("1. Quaternion Construction & Coefficient Accessors")
q1 = Quaternion(1, 2, 3, 4)
q2 = Quaternion(5.0)
q3 = Quaternion(2, -1)
q5 = Quaternion.from_axis_angle((1,0,0), np.pi/2)
print(f"  q1 = {q1}")
print(f"  q1.r={q1.r}  q1.i={q1.i}  q1.j={q1.j}  q1.k={q1.k}")
print(f"  q2 = {q2}  (pure real)")
print(f"  q3 = {q3}  (a+bi)")
print(f"  q5 = {q5}  (rotation quaternion)")
print(f"  _R = {_R}       (real multiplicative identity)")
print(f"  _ONE_Q = {_ONE_Q}   (1+1i+1j+1k)")

# 2. Arithmetic
hdr("2. Basic Arithmetic")
a = Quaternion(2, 3, -1, 4)
b = Quaternion(1, -2, 0, 5)
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  a + b   = {a + b}")
print(f"  a - b   = {a - b}")
print(f"  a * b   = {a * b}")
print(f"  3.0 * a = {3.0 * a}")
print(f"  a / 2   = {a / 2}")

# 3. Hamilton identities
hdr("3. Hamilton Identities")
print(f"  i^2 = {_I*_I}")
print(f"  j^2 = {_J*_J}")
print(f"  k^2 = {_K*_K}")
print(f"  ij  = {_I*_J}")
print(f"  jk  = {_J*_K}")
print(f"  ki  = {_K*_I}")
print(f"  ji  = {_J*_I}")
print(f"  ijk = {_I*_J*_K}")

# 4. Conjugate / Norm / Inverse
hdr("4. Conjugate, Norm, Inverse")
q = Quaternion(3, 4, 0, 0)
print(f"  q       = {q}")
print(f"  q*      = {q.conjugate()}")
print(f"  |q|     = {q.norm()}")
print(f"  q q*    = {q * q.conjugate()}")
print(f"  q^-1    = {q.inverse()}")
print(f"  q^-1 q  = {q.inverse() * q}")
print(f"  q/|q|   = {q.normalize()}")

# 5. Exp / Log / Pow
hdr("5. Exp, Log, Pow")
v = Quaternion(0, 0.3, 0.4, 0)
print(f"  v        = {v}")
print(f"  exp(v)   = {v.exp()}")
print(f"  log(exp(v)) = {v.exp().log()}")
print(f"  v^3      = {v.pow(3)}")
print(f"  v^3 ~ v*v*v: {(v.pow(3) - v*v*v).norm() < 1e-12}")

# 6. 3D Rotation
hdr("6. 3D Vector Rotation")
q_rot = Quaternion.from_axis_angle((0,0,1), np.pi/2)
v0 = (1.0, 0.0, 0.0)
vr = q_rot.rotate_vector(v0)
print(f"  Rotate (1,0,0) around z by pi/2 -> ({vr[0]:.4f}, {vr[1]:.4f})")

# 7. Axis-Angle roundtrip
hdr("7. Axis-Angle Roundtrip")
axis, angle = q_rot.to_axis_angle()
q_back = Quaternion.from_axis_angle(axis, angle)
print(f"  to_axis_angle() -> axis=({axis[0]:.3f},{axis[1]:.3f},{axis[2]:.3f}) angle={angle:.4f}")
print(f"  roundtrip OK: {q_rot.isclose(q_back)}")

# 8. Euler angles
hdr("8. Euler Angles (intrinsic ZYX)")
angles = (0.1, 0.2, 0.3)
q_euler = Quaternion.from_euler(angles, 'zyx')
a_out = q_euler.to_euler('zyx')
print(f"  from_euler({angles}) = {q_euler}")
print(f"  to_euler('zyx')  = ({a_out[0]:.6f}, {a_out[1]:.6f}, {a_out[2]:.6f})")
print(f"  roundtrip OK: {np.allclose(angles, a_out)}")

q_zxz = Quaternion.from_euler((0.4, 0.6, 0.8), 'zxz')
a_zxz = q_zxz.to_euler('zxz')
print(f"  ZXZ proper Euler roundtrip: {Quaternion.from_euler(a_zxz, 'zxz').isclose(q_zxz)}")

# 9. NumPy ufunc integration
hdr("9. NumPy Ufunc Integration (type-preserving)")
q_a = Quaternion(1, 2, 3, 4)
q_b = Quaternion(5, 6, 7, 8)
print(f"  np.add({q_a}, {q_b}) = {np.add(q_a, q_b)}  type={type(np.add(q_a, q_b)).__name__}")
print(f"  np.multiply(3.0, {q_a}) = {np.multiply(3.0, q_a)}  type={type(np.multiply(3.0, q_a)).__name__}")
print(f"  np.negative({q_a}) = {np.negative(q_a)}  type={type(np.negative(q_a)).__name__}")
print(f"  np.conjugate({q_a}) = {np.conjugate(q_a)}")
print(f"  np.absolute({Quaternion(3,4,0,0)}) = {np.absolute(Quaternion(3,4,0,0))}")

v = QuatVector(np.ones((3, 4)))
print(f"  np.add(vector, vector) -> {type(np.add(v, v)).__name__}")

# 10. Interpolation: SLERP
hdr("10. SLERP — Spherical Linear Interpolation")
from quat import slerp
q0 = Quaternion(1, 0, 0, 0)
q1 = Quaternion(0, 1, 0, 0)
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    r = slerp(q0, q1, t)
    print(f"  slerp(t={t:.2f}) = {r}  |r|={r.norm():.1f}")

# 11. Interpolation: squad
hdr("11. Squad — Spherical Cubic Spline (Shoemake 1987)")
from quat import squad, random_unit_quat
np.random.seed(0)
kf = [random_unit_quat() for _ in range(4)]
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    r = squad(kf[0], kf[1], kf[2], kf[3], t)
    print(f"  squad(t={t:.2f}) |r|={r.norm():.6f}")

# 12. Random generators
hdr("12. Random Quaternion Generators")
from quat.random import random_quat, random_unit_quat, random_quat_matrix
print(f"  random_quat(42) = {random_quat(42)}")
print(f"  random_unit_quat(42) norm={random_unit_quat(42).norm():.10f}")
M = random_quat_matrix(2, 3, 42)
print(f"  random_quat_matrix(2,3,42) shape={M.shape}")

# 13. Commutator
hdr("13. Commutator [p,q] = pq - qp")
print(f"  [i,j] = {_I.commutator(_J)}")

# 14. Complex representation
hdr("14. Complex Representation (2x2)")
q = Quaternion(1, 2, 3, 4)
M = q.to_complex_matrix()
print(f"  M(q) =\n{M}")
print(f"  roundtrip OK: {q == Quaternion.from_complex_matrix(M)}")
print(f"  det(M) = {np.linalg.det(M):.6f}  (= |q|^2 = {q.norm_squared()})")

# 15. Real representation
hdr("15. Real Representation (4x4)")
L = q.to_real_matrix_left()
print(f"  L^T L = |q|^2 I_4: {np.allclose(L.T @ L, q.norm_squared() * np.eye(4))}")

# 16. QuatVector
hdr("16. QuatVector")
v = QuatVector([_I, _J, _K, _ONE_Q])
w = QuatVector([Quaternion(2,0,0,0), _ZERO, _K, _I])
print(f"  v = {v}")
print(f"  inner(v,w) = {v.inner(w)}")
print(f"  |v| = {v.norm()}")

# 17. QuatMatrix
hdr("17. QuatMatrix")
A = QuatMatrix([[_I, _J], [_K, _ONE_Q]])
B = QuatMatrix([[_ONE_Q, _ZERO], [_I, _J]])
print(f"  A = {A}")
print(f"  A * B = {A * B}")
x = QuatVector([_I, _J])
print(f"  A * [i, j] = {A * x}")
print(f"  A^dag = {A.H}")

# 18. QuatTensor
hdr("18. QuatTensor (3rd-order)")
T = QuatTensor([
    [[_I, _J],     [_K, _R]],
    [[_R, _ZERO], [_J, _I]],
])
print(f"  T shape: {T.shape}")
print(f"  T[0,0,0] = {T[0,0,0]}")
print(f"  T[0] (2x2 page) =\n{T[0]}")
print(f"  inner(T,T) = {T.inner(T).r}")
print(f"  |T|_F = {T.norm():.6f}")
print(f"  unfold(1) shape: {T.unfold(1).shape}")

# Mode-n product demo
M1 = QuatMatrix([[_J, _K], [_R, _J]])
T1 = T.mode_1_product(M1)
print(f"  T x_1 M1 shape: {T1.shape}")
print(f"  unfold consistency: {np.allclose(T1.unfold(1).to_array(), (M1 * T.unfold(1)).to_array())}")

# All three modes
U1 = QuatMatrix([[_R, _K], [_J, _R]])
U2 = QuatMatrix([[_I, _ZERO], [_R, _J]])
U3 = QuatMatrix([[_K, _R], [_ZERO, _I]])
result = T.mode_1_product(U1).mode_2_product(U2).mode_3_product(U3)
print(f"  T x_1 U1 x_2 U2 x_3 U3 shape: {result.shape}, norm: {result.norm():.6f}")

# 19. Type conversion
hdr("19. Type Conversion")
print(f"  float({Quaternion(5.0)}):      {float(Quaternion(5.0))}")
print(f"  complex({Quaternion(2,3)}):    {complex(Quaternion(2,3))}")
print(f"  bool({Quaternion(1,2,3,4)}):   {bool(Quaternion(1,2,3,4))}")
print(f"  {Quaternion(1,2,3,4)}.components: {Quaternion(1,2,3,4).components}")

print(f"\n{HR}")
print(f"  End of demo. Library: quat")
print(f"{HR}")
