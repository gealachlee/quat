"""Quaternion Algebra Library — quat package.

Provides:
  Quaternion  — single quaternion value
  QuatVector  — 1-d collection of quaternions
  QuatMatrix  — 2-d quaternion matrix
  QuatTensor  — 3-d quaternion tensor

  quat()      — convenience constructor
  dict_to_quat_matrix, dict_to_quat_tensor, labels_to_quat_vector

Algebra primitives (from quat.algebra):
  _hamilton, _CONJ, _REAL_LEFT

Basis constants (from quat.core):
  _I, _J, _K, _ZERO, _R, _ONE_Q

Utilities (from quat.utils):
  to_ndarray, from_ndarray, from_components, broadcast_quat_shapes, stack_quat,
  isnan, isinf, isfinite, isclose

Serialization (from quat.serialization):
  to_json, from_json, to_bytes, from_bytes,
  to_scipy_rotation, from_scipy_rotation

Optimized (from quat.optimized):
  hamilton_einsum, quat_matmul, conjugate_batch,
  norm_squared_batch, normalize_batch

Algorithms (from quat.algorithms):
  cubic_kernel_matrix, gaussian_kernel_matrix,
  compute_kernel_matrix, compute_kernel_cross, normalize_kernel,
  solve_ksqmm, predict_ksqmm

Linear algebra (from quat.linalg):
  svd, rank, condition_number, pseudo_inverse,
  trace, det, norm, solve
"""
from quat.algebra import _hamilton, _CONJ, _REAL_LEFT
from quat.core import Quaternion, quat, _I, _J, _K, _ZERO, _R, _ONE_Q
from quat.collections import (
    QuatVector, QuatMatrix, QuatTensor,
    dict_to_quat_matrix, dict_to_quat_tensor, labels_to_quat_vector,
)
from quat.utils import to_ndarray, from_ndarray, from_components, broadcast_quat_shapes, stack_quat
from quat.utils import isnan, isinf, isfinite, isclose
from quat.serialization import (
    to_json, from_json, to_bytes, from_bytes,
    to_scipy_rotation, from_scipy_rotation,
)
from quat.optimized import (
    hamilton_einsum, quat_matmul,
    conjugate_batch, norm_squared_batch, normalize_batch,
)
from quat.algorithms import (
    cubic_kernel_matrix, cubic_kernel_cross,
    gaussian_kernel_matrix, gaussian_kernel_cross,
    normalize_kernel, compute_kernel_matrix, compute_kernel_cross,
    solve_ksqmm, predict_ksqmm,
)
from quat.linalg import (
    svd, rank, condition_number, pseudo_inverse,
    trace, det, norm, solve,
)

__all__ = [
    'Quaternion', 'QuatVector', 'QuatMatrix', 'QuatTensor',
    'quat', 'dict_to_quat_matrix', 'dict_to_quat_tensor', 'labels_to_quat_vector',
    '_hamilton', '_CONJ', '_REAL_LEFT', '_I', '_J', '_K', '_ZERO', '_R', '_ONE_Q',
    'to_ndarray', 'from_ndarray', 'from_components', 'broadcast_quat_shapes', 'stack_quat',
    'isnan', 'isinf', 'isfinite', 'isclose',
    'to_json', 'from_json', 'to_bytes', 'from_bytes',
    'to_scipy_rotation', 'from_scipy_rotation',
    'hamilton_einsum', 'quat_matmul',
    'conjugate_batch', 'norm_squared_batch', 'normalize_batch',
    'cubic_kernel_matrix', 'cubic_kernel_cross',
    'gaussian_kernel_matrix', 'gaussian_kernel_cross',
    'normalize_kernel', 'compute_kernel_matrix', 'compute_kernel_cross',
    'solve_ksqmm', 'predict_ksqmm',
    'svd', 'rank', 'condition_number', 'pseudo_inverse',
    'trace', 'det', 'norm', 'solve',
]
