#!/usr/bin/env python3
"""Q-ADMM solver for KSQMM with ramp loss.

Implements Algorithm 1 from:
  Li, Lin & Liu, "Kernel Support Quaternion Matrix Machines with Ramp Loss"

The solver operates on the REAL representation of quaternion matrices (R-variant),
converting the quaternion linear system (K + σ K²) α = -K conj(π) into a
4N×4N real linear system solved via lstsq with ridge regularization.
"""

import numpy as np
from quaternion import Quaternion, QuatMatrix, _hamilton, _CONJ, _RW

# 1_q = 1 + i + j + k
_ONE_Q = np.array([1., 1., 1., 1.])


# ---------------------------------------------------------------------------
# Ramp-loss proximal operator  (Lemma 4.3 / Eq 4.9b)
# ---------------------------------------------------------------------------
def _prox_ramp(t, beta):
    r"""Proximal operator of ramp loss (vectorized per component).

    prox_{β l_r}(t) =
        t,       if t > 1 + β/2  or  t ≤ 0
        t - β,   if  β ≤ t < 1 + β/2
        0,       if 0 < t < β

    Args:
        t:    (N, 4) array
        beta: (4,) or scalar

    Returns:
        (N, 4) array
    """
    t = np.asarray(t, dtype=float)
    beta = np.asarray(beta, dtype=float)
    out = np.empty_like(t)
    for c in range(t.shape[1]):
        tc = t[:, c]
        bc = beta[c] if beta.ndim > 0 else beta
        oc = out[:, c]
        m1 = tc <= 0.
        m2 = tc > 1. + bc / 2.
        m3 = (tc > 0.) & (tc < bc)
        m4 = ~(m1 | m2 | m3)
        oc[m1] = tc[m1]
        oc[m2] = tc[m2]
        oc[m3] = 0.
        oc[m4] = tc[m4] - bc
    return out


# ---------------------------------------------------------------------------
# Quaternion real-representation helpers
# ---------------------------------------------------------------------------
def _quat_to_real_vector(Q):
    """Quaternion vector (N, 4) → real vector (4N,)."""
    return Q.reshape(-1)


def _real_to_quat_vector(v):
    """Real vector (4N,) → quaternion vector (N, 4)."""
    return v.reshape(-1, 4)


def _quat_conj(Q):
    """Element-wise quaternion conjugate. Q: (..., 4)."""
    return Q * _CONJ


def _kernel_real_matrix(K):
    """Build 4N×4N real representation of quaternion kernel matrix K (N,N,4)."""
    return QuatMatrix(K).to_real_matrix()


# ---------------------------------------------------------------------------
# Core ADMM operations
# ---------------------------------------------------------------------------
def _compute_pred(K, alpha, b):
    r"""Compute α^* K + b (quaternion vector of length N).

    (α^* K)_n = Σ_j conj(α_j) K_{jn}  =  Σ_j K_{jn} · conj(α_j)? No —
    quaternion multiplication is non-commutative.  Using the formula:
        (α^* K)_n = Σ_j conj(α_j) · K_{jn}   (Hamilton product)

    Vectorized: _hamilton(conj(α)[:, None, :], K.transpose(1,0,2)) → (N, N, 4)
                sum over j → (N, 4)
    """
    # conj(α): (N, 4), we need conj(α_j) * K_{jn}
    # = _hamilton(conj(α)[:, None, :], K) → (N, N, 4), sum over axis=0 → (N, 4)
    # Wait: for column j of K (shape N, 4):
    #   For each n: result_n = Σ_j conj(α_j) · K[j, n]
    #   = Σ_j K[j, n, :] * conj(α_j) on the left? No!
    #
    # conj(α_j) · K_{jn}: left-multiply K_{jn} by conj(α_j)
    # = _hamilton(conj(α)[:, None, :], K) → (N, N, 4)
    # Then sum over axis=0 (j dimension) → (N, 4)
    conj_a = _quat_conj(alpha)  # (N, 4)
    # (N, 4) × (N, N, 4) — hamilton(conj_a_j, K[j,n])
    prod = _hamilton(conj_a[:, None, :], K)  # (N, N, 4)
    return prod.sum(axis=0) + b  # (N, 4)


def _solve_alpha(K_real, pi, sigma, ridge):
    r"""Solve  (K + σ K²) α = -K conj(π)   via real linear system.

    In real representation: (K_R + σ K_R²) α_R = -K_R · conj(π)_R
    Use stable factorization: K_R (I + σ K_R) α_R = -K_R · conj(π)_R
    """
    N4 = K_real.shape[0]
    I = np.eye(N4)
    A = K_real @ (I + sigma * K_real) + ridge * I
    pi_real = _quat_to_real_vector(_quat_conj(pi))
    rhs = -K_real @ pi_real
    alpha_real = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return _real_to_quat_vector(alpha_real)


# ---------------------------------------------------------------------------
# Main ADMM solver
# ---------------------------------------------------------------------------
def solve_ksqmm(K_train, y_labels, K_test=None,
                C=1., sigma=0.1, mu=0.1, ridge=1e-6,
                max_iter=1000, tol=1e-4, verbose=False):
    r"""Solve KSQMM with ramp loss via Q-ADMM (Algorithm 1).

    Args:
        K_train:   (N, N, 4)  quaternion kernel matrix
        y_labels:  (N, 4)     quaternion labels (per sample)
        K_test:    (N_test, N, 4)  optional cross kernel for evaluation
        C:         regularization parameter
        sigma:     ADMM penalty
        mu:        dual step-size
        ridge:     regularization for linear system
        max_iter:  maximum ADMM iterations
        tol:       convergence tolerance (primal residual)
        verbose:   print progress

    Returns:
        alpha: (N, 4)    learned coefficients
        b:     (4,)      learned bias
        info:  dict with residuals, n_iter
    """
    N = K_train.shape[0]
    C_i = np.array([C, C, C, C])  # per-component C (can be extended)

    # Precompute real representation of K
    K_real = _kernel_real_matrix(K_train)  # 4N × 4N

    # Initialize
    alpha = np.zeros((N, 4))
    u     = np.zeros((N, 4))
    b     = np.zeros(4)
    lam   = np.zeros((N, 4))  # λ: (N, 4) real Lagrange multipliers

    y_q = y_labels  # (N, 4)

    # Use ridge proportional to N for stability
    effective_ridge = ridge * max(1., N / 10.)

    for k in range(max_iter):
        # ---- Step 1: update u ----
        # τ_{ni} = proj_i(1_q - y_n ∘ (α^* K_n + b))
        pred = _compute_pred(K_train, alpha, b)  # (N, 4)
        tau = _ONE_Q - y_q * pred  # Hadamard ∘ = element-wise (N,4)

        # η_{ni} = τ_{ni} - λ_{ni} / σ
        eta = tau - lam / sigma  # (N, 4)

        # u_{ni} = prox_{C_i/σ · l_r}(η_{ni})
        beta = C_i / sigma  # (4,)
        u = _prox_ramp(eta, beta)

        # ---- Step 2: update alpha ----
        # λ^n_q = reconstitute quaternion from λ
        lam_q = lam.copy()  # already (N, 4) quaternion form with real λ values

        # π_n = b + y_n ∘ (u_n - 1_q + λ^n_q / σ)
        pi = b + y_q * (u - _ONE_Q + lam_q / sigma)  # Hadamard ∘

        alpha = _solve_alpha(K_real, pi, sigma, effective_ridge)

        # ---- Step 3: update b ----
        # ξ_n = α^* K_n + y_n ∘ (u_n - 1_q + λ^n_q / σ)
        pred_new = _compute_pred(K_train, alpha, 0.)  # (N, 4) without b
        xi = pred_new + y_q * (u - _ONE_Q + lam_q / sigma)  # Hadamard ∘

        b = -xi.mean(axis=0)  # (4,)

        # ---- Step 4: update λ ----
        # z_n = u_n - (1_q - y_n ∘ (α^* K_n + b))
        pred_full = _compute_pred(K_train, alpha, b)  # (N, 4)
        z = u - (_ONE_Q - y_q * pred_full)  # Hadamard ∘
        lam = lam + mu * sigma * z  # (N, 4)

        # ---- Check convergence ----
        primal_res = np.abs(z).max()
        if verbose and (k % 100 == 0 or k < 5):
            print(f"  iter {k:4d}: primal_res={primal_res:.6f}")
        if primal_res < tol:
            if verbose:
                print(f"  Converged at iter {k}")
            break

    info = {'n_iter': k + 1, 'primal_res': primal_res}
    return alpha, b, info


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_ksqmm(K_test, alpha, b):
    r"""Predict quaternion labels for test samples.

    f = α^* K_test_conj + b  where K_test_conj[a,i] = conj(κ(X^a, X^i))
    ŷ = qsgn(f)

    Note: K_test[a,i] = κ(X^a_test, X^i_train), but prediction needs
    κ(X^i_train, X^a_test) = conj(κ(X^a_test, X^i_train)), so we
    conjugate K_test before the matrix-vector product.

    Args:
        K_test: (N_test, N_train, 4)  cross kernel
        alpha:  (N_train, 4)
        b:      (4,)

    Returns:
        y_pred: (N_test, 4)  quaternion labels in {±1 ± i ± j ± k}
    """
    conj_a = _quat_conj(alpha)       # (N_train, 4) = β
    conj_K = _quat_conj(K_test)      # κ(X^i_train, X^a_test)

    # (α^* K^T)_m = Σ_i β_i · κ(X^i_train, X^m_test) = Σ_i β_i · conj_K[m, i]
    prod = _hamilton(conj_a[None, :, :], conj_K)  # (N_test, N_train, 4)
    f = prod.sum(axis=1) + b  # (N_test, 4)

    y_pred = np.sign(f)
    y_pred[y_pred == 0] = 1
    return y_pred
