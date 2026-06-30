# =========================================================================
#  LLM-generated code.  See README.md for full disclosure.
# =========================================================================

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

Linear algebra (from quat.linalg):
  svd, svd_values, rank, condition_number, pseudo_inverse,
  trace, det, norm, solve

Random (from quat.random):
  random_quat, random_unit_quat, random_quat_vector,
  random_quat_matrix, random_quat_tensor

Interpolation (from quat.interpolate):
  slerp, slerp_vector, squad

Signal processing (from quat.signal):
  qfft, iqfft, qfft2, iqfft2,
  qconv, qconv2,
  lowpass, highpass, bandpass, bandstop

Distance (from quat.distance):
  rotation.intrinsic, rotation.chordal, rotor.intrinsic, rotor.chordal

Kinematics (from quat.kinematics):
  angular_velocity, integrate_angular_velocity, rotate_frame

Mean (from quat.mean):
  mean_rotation, karcher_mean

Statistics (from quat.stats):
  quaternion_mean, quaternion_cov, quaternion_pca
"""

__version__ = "0.2.0"
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
from quat.algebra import (
    hamilton_einsum, quat_matmul,
    conjugate_batch, norm_squared_batch, normalize_batch,
)
from quat.linalg import (
    svd, svd_values, rank, condition_number, pseudo_inverse,
    trace, det, norm, solve,
)
from quat.random import (
    random_quat, random_unit_quat, random_quat_vector,
    random_quat_matrix, random_quat_tensor,
)
from quat.interpolate import slerp, slerp_vector, squad
from quat.interpolate import angular_velocity, integrate_angular_velocity, rotate_frame
from quat.signal import (
    qfft, iqfft, qfft2, iqfft2,
    qconv, qconv2,
    lowpass, highpass, bandpass, bandstop,
)
from quat.stats import rotation, rotor, mean_rotation, karcher_mean
from quat.stats import quaternion_mean, quaternion_cov, quaternion_pca

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
    'svd', 'svd_values', 'rank', 'condition_number', 'pseudo_inverse',
    'trace', 'det', 'norm', 'solve',
    'random_quat', 'random_unit_quat', 'random_quat_vector',
    'random_quat_matrix', 'random_quat_tensor',
    'slerp', 'slerp_vector', 'squad',
    'qfft', 'iqfft', 'qfft2', 'iqfft2',
    'qconv', 'qconv2',
    'lowpass', 'highpass', 'bandpass', 'bandstop',
    'rotation', 'rotor',
    'angular_velocity', 'integrate_angular_velocity', 'rotate_frame',
    'mean_rotation', 'karcher_mean',
    'quaternion_mean', 'quaternion_cov', 'quaternion_pca',
    '__version__',
]
