"""01_quickstart — basic quaternion construction, algebra, and conversion."""
from quat import Quaternion, QuatVector, QuatMatrix, _I, _J, _K, _R
from quat import random_unit_quat, slerp
from quat.stats import rotation
import numpy as np

# -- Construction ---------------------------------------------------------
q = Quaternion(1, 2, 3, 4)
r = Quaternion(5.0)                         # pure real
u = Quaternion.from_axis_angle((0, 0, 1), np.pi / 2)
e = Quaternion.from_euler((0.1, 0.2, 0.3))  # intrinsic ZYX

print("q =", q)
print("r =", r)
print("u (90 deg about z) =", u)

# -- Hamilton product (basis identities) ---------------------------------
print("\ni^2 =", _I * _I)                    # -1
print("j^2 =", _J * _J)                    # -1
print("ij  =", _I * _J)                   # k
print("ijk =", _I * _J * _K)             # -1
print("ji  =", _J * _I)                   # -k (non-commutative)
print("pq vs qp:",
      (Quaternion(1, 2, 3, 4) * Quaternion(5, 6, 7, 8)),
      "!=",
      (Quaternion(5, 6, 7, 8) * Quaternion(1, 2, 3, 4)))

# -- Algebra --------------------------------------------------------------
print("\n|q| =", q.norm())
print("q* =", q.conjugate())
print("q * q* =", q * q.conjugate())
print("q^-1 =", q.inverse())
print("q * q^-1 =", q * q.inverse())
print("exp(q) =", q.exp())

# -- 3D rotation ---------------------------------------------------------
v = (1, 0, 0)
rotated = u.rotate_vector(v)
print(f"\nrotate({v}) by 90 deg around z ~", np.round(rotated, 3))

# -- Interpolation -------------------------------------------------------
a = Quaternion(1, 0, 0, 0)
b = Quaternion(0, 1, 0, 0)
mid = slerp(a, b, 0.5)
print("slerp halfway =", mid)

# -- Collections ---------------------------------------------------------
vec = QuatVector([_I, _J, _K, _R])
print("\nQuatVector shape:", vec.shape)
print("vec * _I (element-wise):", vec * _I)
print("vec * 3.0 (scalar):", vec * 3.0)

M = QuatMatrix([[_I, _J], [_K, _R]])
print("\nQuatMatrix 2x2:")
print("M * M (matrix mult):", M * M)
print("M * vec (matrix-vec):", M * QuatVector([_I, _J]))

# -- Distance ------------------------------------------------------------
q1 = Quaternion(1, 0, 0, 0)
q2 = Quaternion(0, 1, 0, 0)
d = rotation.intrinsic(q1, q2)
print("\nrotation distance between identity and 90 deg:", d)

# -- Matrix representations ----------------------------------------------
print("\nq complex 2x2:\n", q.to_complex_matrix())
print("q real left 4x4:\n", q.to_real_matrix_left().astype(int))
print("q real right 4x4:\n", q.to_real_matrix_right().astype(int))
print("roundtrip real left:", Quaternion.from_real_matrix_left(
    q.to_real_matrix_left()).isclose(q))
