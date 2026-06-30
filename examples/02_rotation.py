"""02_rotation — rotation math, Euler angles, axis-angle, SLERP, kinematics."""
import numpy as np
from quat import Quaternion, QuatVector
from quat import slerp, squad, random_unit_quat
from quat.interpolate import angular_velocity, integrate_angular_velocity

print("=" * 60)
print("Rotation round-trip: euler → quat → euler")
print("=" * 60)

angles_in = (0.1, 0.2, 0.3)
q = Quaternion.from_euler(angles_in, seq="zyx")
angles_out = q.to_euler(seq="zyx")
print("in:", np.round(angles_in, 6))
print("out:", np.round(angles_out, 6))
print("match:", np.allclose(angles_in, angles_out))

print()
print("=" * 60)
print("Axis-angle round-trip")
print("=" * 60)

axis_in = np.array([0, 0, 1])
angle_in = np.pi / 3
q = Quaternion.from_axis_angle(axis_in, angle_in)
axis_out, angle_out = q.to_axis_angle()
print("axis in:", axis_in, "axis out:", np.round(axis_out, 4))
print("angle in:", angle_in, "angle out:", round(angle_out, 6))

print()
print("=" * 60)
print("SLERP animation (5 frames)")
print("=" * 60)

q0 = Quaternion(1, 0, 0, 0)
q1 = Quaternion.from_axis_angle((0, 0, 1), np.pi)  # 180° about z
for i in range(6):
    t = i / 5
    qi = slerp(q0, q1, t)
    print(f"  t={t:.1f}: axis={np.round(qi.to_axis_angle()[0], 3)}, angle={qi.angle:.3f}")

print()
print("=" * 60)
print("SQUAD (cubic spline)")
print("=" * 60)

q0 = random_unit_quat(42)
q1 = random_unit_quat(43)
q2 = random_unit_quat(44)
q3 = random_unit_quat(45)
mid = squad(q0, q1, q2, q3, 0.5)
print("midpoint norm:", mid.norm())

print()
print("=" * 60)
print("Angular velocity estimation")
print("=" * 60)

dt = 0.01
t = np.arange(100) * dt
q_list = [Quaternion.from_axis_angle((0, 0, 1), ti) for ti in t]
q_seq = QuatVector(q_list)
omega = angular_velocity(q_seq, dt)
print("omega_z mean (expected 1.0):", np.mean(omega[:, 2]))
