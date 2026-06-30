"""01_quickstart — basic quaternion algebra and construction."""
from quat import Quaternion, QuatVector, QuatMatrix
from quat import slerp, random_unit_quat, _I, _J, _K
import numpy as np

# -- Construction -----------------------------------------------------
q = Quaternion(1, 2, 3, 4)
r = Quaternion(5.0)                         # pure real
u = Quaternion.from_axis_angle((0, 0, 1), np.pi / 2)
e = Quaternion.from_euler((0.1, 0.2, 0.3))  # intrinsic ZYX

print("q =", q)
print("r =", r)
print("u (90° about z) =", u)

# -- Hamilton product -------------------------------------------------
print("i * j =", _I * _J)                   # k
print("i * j * k =", _I * _J * _K)          # -1

# -- Algebra ----------------------------------------------------------
print("|q| =", q.norm())
print("q * inverse(q) =", q * q.inverse())
print("exp(q) =", q.exp())

# -- 3D rotation ------------------------------------------------------
v = (1, 0, 0)
rotated = u.rotate_vector(v)
print(f"rotate({v}) by 90° around z ≈", np.round(rotated, 3))

# -- Interpolation ----------------------------------------------------
a = Quaternion(1, 0, 0, 0)
b = Quaternion(0, 1, 0, 0)
mid = slerp(a, b, 0.5)
print("slerp halfway =", mid)

# -- Collections ------------------------------------------------------
vec = QuatVector([
    Quaternion(1, 0, 0, 0),
    Quaternion(0, 1, 0, 0),
    Quaternion(0, 0, 1, 0),
])
print("QuatVector shape:", vec.shape)
print("real parts:", vec.real)

M = QuatMatrix.eye(3)
print("3x3 identity trace:", M * vec)

# -- Distance ---------------------------------------------------------
from quat.stats import rotation
q1 = Quaternion(1, 0, 0, 0)
q2 = Quaternion(0, 1, 0, 0)
d = rotation.intrinsic(q1, q2)
print("rotation distance between identity and 90°:", d)
