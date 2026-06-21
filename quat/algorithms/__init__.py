"""Quaternion learning algorithms — kernel methods and ADMM solver."""
from quat.algorithms.kernels import (
    cubic_kernel_matrix,
    cubic_kernel_cross,
    gaussian_kernel_matrix,
    gaussian_kernel_cross,
    normalize_kernel,
    compute_kernel_matrix,
    compute_kernel_cross,
)
from quat.algorithms.solver import (
    solve_ksqmm,
    predict_ksqmm,
)
