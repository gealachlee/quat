#!/usr/bin/env python3
"""Demo: 生成一个2×2纯四元数矩阵（实部全为0），并输出其实表示矩阵。"""

import numpy as np
from quat import Quaternion, QuatMatrix, _REAL_LEFT

# 构造 2×2 四元数矩阵，每个元素实部为 0（纯四元数）
# q11 = 0 + 1i + 2j + 3k
# q12 = 0 + 4i + 5j + 6k
# q21 = 0 + 7i + 8j + 9k
# q22 = 0 + 0i + 1j + 2k

data = np.array([
    [[0, 1, 2, 3],   # q11
     [0, 4, 5, 6]],  # q12
    [[0, 7, 8, 9],   # q21
     [0, 0, 1, 2]]   # q22
], dtype=float)

Q = QuatMatrix(data)

print("=" * 60)
print("原始 2×2 四元数矩阵 Q:")
print("=" * 60)
for i in range(2):
    for j in range(2):
        q = Q[i, j]
        a, b, c, d = q._data
        print(f"  Q[{i},{j}] = {a} + {b}i + {c}j + {d}k")
print()

# 显示 Q 的内部数据表示 (2,2,4)
print("Q._data shape:", Q._data.shape)
print("Q._data (m=2, n=2, component=4):")
print(Q._data)
print()

# 计算实表示矩阵 R(Q): 8×8
R = Q.to_real_matrix_left()

print("=" * 60)
print("实表示矩阵 R(Q)  (8×8):")
print("=" * 60)
print()

# 显示 _REAL_LEFT 张量的含义：每个 4×4 块的结构
print("单元素 q = a+bi+cj+dk 的 4×4 实表示块 R(q):")
print("  R(q) = [[a, -b, -c, -d],")
print("          [b,  a, -d,  c],")
print("          [c,  d,  a, -b],")
print("          [d, -c,  b,  a]]")
print()

# 打印整个 8×8 矩阵
print("完整 8×8 实表示矩阵 (4行块 × 4列块，每块对应一个四元数):")
print("-" * 60)
np.set_printoptions(precision=0, suppress=True, linewidth=120, threshold=100)
print(R.astype(int))
np.set_printoptions(precision=8, suppress=False)

print()
print("-" * 60)
print("矩阵结构说明:")
print("  行块 0 (第0-3行): Q[0,0] 块 | Q[0,1] 块")
print("  行块 1 (第4-7行): Q[1,0] 块 | Q[1,1] 块")
print()

# 验证：R(Q) · vec(v) 表示四元数矩阵-向量积 Q @ v
# 对任意四元数向量 v = (v0, v1, v2, v3)
v = np.array([1.0, -1.0, 0.5, -0.5,  # v0
               2.0,  0.0, 1.0,  0.0])  # v1
print("验证  R(Q) · v_R = (Q @ v)_R  (取随机向量 v):")
qv_result = np.zeros((2, 4))
for i in range(2):
    for j in range(2):
        qi = Q[i, j]._data          # Q[i,j] 四元数
        vj = v[j*4:(j+1)*4]         # v[j] 四元数
        # Hamilton 积: qi * vj
        a1, b1, c1, d1 = qi
        a2, b2, c2, d2 = vj
        r0 = a1*a2 - b1*b2 - c1*c2 - d1*d2
        r1 = a1*b2 + b1*a2 + c1*d2 - d1*c2
        r2 = a1*c2 - b1*d2 + c1*a2 + d1*b2
        r3 = a1*d2 + b1*c2 - c1*b2 + d1*a2
        qv_result[i, 0] += r0
        qv_result[i, 1] += r1
        qv_result[i, 2] += r2
        qv_result[i, 3] += r3

mv_result = (R @ v).reshape(2, 4)
print(f"  直接四元数乘法结果: {qv_result}")
print(f"  R(Q) @ v_R 结果:     {mv_result}")
print(f"  差异: {np.abs(qv_result - mv_result).max():.2e}")
print()

print("=" * 60)
print("_REAL_LEFT 张量 (4×4×4) — 定义每个 4×4 块的结构:")
print("  _REAL_LEFT[r, c, k]: 在块位置 (r,c) 处，四元数分量 k 的系数")
print("=" * 60)
for r in range(4):
    print(f"  row r={r}:")
    for c in range(4):
        coeffs = _REAL_LEFT[r, c]
        nonzero = [(k, v) for k, v in enumerate(coeffs) if v != 0]
        terms = " + ".join(f"{int(v)}·k{k}" for k, v in nonzero)
        print(f"    col c={c}: {terms}")
