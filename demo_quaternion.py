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

# 7. Commutator
hdr("7. Commutator [p,q] = pq - qp")
print(f"  [i,j] = {_I.commutator(_J)}")

# 8. Complex representation
hdr("8. Complex Representation (2x2)")
q = Quaternion(1, 2, 3, 4)
M = q.to_complex_matrix()
print(f"  M(q) =\n{M}")
print(f"  roundtrip OK: {q == Quaternion.from_complex_matrix(M)}")
print(f"  det(M) = {np.linalg.det(M):.6f}  (= |q|^2 = {q.norm_squared()})")

# 9. Real representation
hdr("9. Real Representation (4x4)")
L = q.to_real_matrix_left()
print(f"  L^T L = |q|^2 I_4: {np.allclose(L.T @ L, q.norm_squared() * np.eye(4))}")

# 10. QuatVector
hdr("10. QuatVector")
v = QuatVector([_I, _J, _K, _ONE_Q])
w = QuatVector([Quaternion(2,0,0,0), _ZERO, _K, _I])
print(f"  v = {v}")
print(f"  inner(v,w) = {v.inner(w)}")
print(f"  |v| = {v.norm()}")

# 11. QuatMatrix
hdr("11. QuatMatrix")
A = QuatMatrix([[_I, _J], [_K, _ONE_Q]])
B = QuatMatrix([[_ONE_Q, _ZERO], [_I, _J]])
print(f"  A = {A}")
print(f"  A * B = {A * B}")
x = QuatVector([_I, _J])
print(f"  A * [i, j] = {A * x}")
print(f"  A^dag = {A.H}")

# 12. QuatTensor
hdr("12. QuatTensor (3rd-order)")
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

# 13. Type conversion
hdr("13. Type Conversion")
print(f"  float({Quaternion(5.0)}):      {float(Quaternion(5.0))}")
print(f"  complex({Quaternion(2,3)}):    {complex(Quaternion(2,3))}")
print(f"  bool({Quaternion(1,2,3,4)}):   {bool(Quaternion(1,2,3,4))}")
print(f"  {Quaternion(1,2,3,4)}.components: {Quaternion(1,2,3,4).components}")

print(f"\n{HR}")
print(f"  End of demo. Library: quaternion.py")
print(f"{HR}")
