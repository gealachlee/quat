"""04_stats — distance metrics, mean rotation, PCA."""
import numpy as np
from quat import Quaternion, QuatVector, QuatMatrix
from quat.stats import rotation, rotor, mean_rotation, karcher_mean
from quat.stats import quaternion_mean, quaternion_cov, quaternion_pca

print("=" * 60)
print("Distance metrics")
print("=" * 60)

q_id = Quaternion(1, 0, 0, 0)
q_90z = Quaternion.from_axis_angle((0, 0, 1), np.pi / 2)

print("rotation.intrinsic(id, 90°z) =", rotation.intrinsic(q_id, q_90z))
print("rotation.chordal(id, 90°z)  =", rotation.chordal(q_id, q_90z))
print("rotor.intrinsic(id, -id)     =", rotor.intrinsic(q_id, -q_id))
print("  (non-zero — rotor skips sign)")

print()
print("=" * 60)
print("Mean rotation")
print("=" * 60)

rng = np.random.default_rng(42)
samples = []
base = Quaternion.from_axis_angle((1, 2, 3), 1.0)
for _ in range(50):
    noise_axis = rng.normal(size=3)
    noise_angle = rng.normal(scale=0.1)
    noise_q = Quaternion.from_axis_angle(noise_axis, noise_angle)
    samples.append(noise_q * base)
qv = QuatVector(samples)

m_svd = mean_rotation(qv)
m_kar = karcher_mean(qv, max_iter=30)
print("SVD mean norm:", m_svd.norm())
print("Karcher mean norm:", m_kar.norm())
print("Distance between two means:", rotation.intrinsic(m_svd, m_kar))

print()
print("=" * 60)
print("Quaternion PCA")
print("=" * 60)

data = rng.normal(size=(200, 4))
mu = quaternion_mean(data)
C = quaternion_cov(data)
components, explained = quaternion_pca(data, n_components=4)
print("4x4 covariance trace:", np.trace(C))
print("explained variance ratio:", np.round(explained / explained.sum(), 3))

print()
print("=" * 60)
print("SVD on a quaternion matrix")
print("=" * 60)

from quat import svd, svd_values, rank
A = QuatMatrix(rng.normal(size=(4, 5, 4)))
s = svd_values(A)
print("singular values:", np.round(s, 4))
print("rank:", rank(A))
print("condition number:", np.round(s[0] / s[-1], 2))
