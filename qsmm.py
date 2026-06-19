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
    r"""Solve (K + sigma K^2) alpha = -K conj(pi) via real linear system.

    Correct form:  K_R (I + sigma K_R) alpha_R = -K_R * conj(pi)_R
    With ridge:    [K_R(I + sigma K_R) + ridge*I] alpha_R = -K_R * conj(pi)_R
    """
    N4 = K_real.shape[0]
    I = np.eye(N4)

    A = K_real @ (I + sigma * K_real)
    A_norm = np.linalg.norm(A, ord=2)
    A += ridge * max(1., A_norm) * I

    pi_real = _quat_to_real_vector(_quat_conj(pi))
    rhs = -K_real @ pi_real

    alpha_real = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return _real_to_quat_vector(alpha_real)


# ---------------------------------------------------------------------------
# Main ADMM solver
# ---------------------------------------------------------------------------
def solve_ksqmm(K_train, y_labels, K_test=None, y_test_labels=None,
                C=1., sigma=0.1, mu=0.1, ridge=1e-6,
                max_iter=1000, tol=1e-4, verbose=False):
    r"""Solve KSQMM with ramp loss via Q-ADMM (Algorithm 1).

    Args:
        K_train:       (N, N, 4)  quaternion kernel matrix
        y_labels:      (N, 4)     quaternion labels (per sample)
        K_test:        (N_test, N, 4)  optional cross kernel for eval accuracy
        y_test_labels: (N_test, 4)     quaternion test labels
        C:             regularization parameter
        sigma:         ADMM penalty
        mu:            dual step-size
        ridge:         regularization for linear system
        max_iter:      maximum ADMM iterations
        tol:           convergence tolerance (primal residual)
        verbose:       print per-iteration progress with accuracies

    Returns:
        alpha: (N, 4)    learned coefficients
        b:     (4,)      learned bias
        info:  dict with residuals, n_iter, train_acc, test_acc
    """
    N = K_train.shape[0]
    C_i = np.array([C, C, C, C])

    K_real = _kernel_real_matrix(K_train)
    y_q = y_labels

    # Initialize
    alpha   = np.zeros((N, 4))
    u       = np.zeros((N, 4))
    b_val   = np.zeros(4)
    lam     = np.zeros((N, 4))

    # Early-exit tracking
    best_alpha   = alpha.copy()
    best_b       = b_val.copy()
    best_test_acc = -1.
    alpha_max     = 1e6   # overflow threshold

    effective_ridge = ridge * max(1., N / 10.)
    train_acc = test_acc = None

    for k in range(max_iter):
        # ---- Step 1: update u ----
        pred = _compute_pred(K_train, alpha, b_val)
        tau  = _ONE_Q - y_q * pred
        eta  = tau - lam / sigma
        beta = C_i / sigma
        u    = _prox_ramp(eta, beta)

        # ---- Step 2: update alpha ----
        lam_q = lam.copy()
        pi    = b_val + y_q * (u - _ONE_Q + lam_q / sigma)
        alpha = _solve_alpha(K_real, pi, sigma, ridge)

        # ---- Overflow check: exit early if alpha explodes ----
        alpha_norm = np.sqrt((alpha * alpha).sum())
        if alpha_norm > alpha_max or not np.isfinite(alpha_norm):
            if verbose:
                print(f"  iter {k:4d}: ALPHA OVERFLOW |a|={alpha_norm:.2e} -- stopping, restoring best {best_test_acc:.4f}")
            alpha = best_alpha
            b_val = best_b
            break

        # ---- Step 3: update b ----
        pred_new = _compute_pred(K_train, alpha, 0.)
        xi       = pred_new + y_q * (u - _ONE_Q + lam_q / sigma)
        b_val    = -xi.mean(axis=0)

        # ---- Step 4: update lambda ----
        pred_full = _compute_pred(K_train, alpha, b_val)
        z         = u - (_ONE_Q - y_q * pred_full)
        lam       = lam + mu * sigma * z

        primal_res = np.abs(z).max()

        # ---- Per-iteration metrics & best-model tracking ----
        if verbose and (k % 10 == 0 or k < 5 or primal_res < tol):
            msg = f"  iter {k:4d}: pr={primal_res:.6f}  |a|={alpha_norm:.2e}"

            yp_train = predict_ksqmm(K_train, alpha, b_val)
            train_acc = (np.sign(yp_train[:, 0]) == np.sign(y_q[:, 0])).mean()
            msg += f"  tr_acc={train_acc:.4f}"

            if K_test is not None and y_test_labels is not None:
                yp_test = predict_ksqmm(K_test, alpha, b_val)
                test_acc = (np.sign(yp_test[:, 0]) == np.sign(y_test_labels[:, 0])).mean()
                msg += f"  te_acc={test_acc:.4f}"
            print(msg)

        # Track best model by test accuracy
        if K_test is not None and y_test_labels is not None:
            yp_test = predict_ksqmm(K_test, alpha, b_val)
            cur_test = (np.sign(yp_test[:, 0]) == np.sign(y_test_labels[:, 0])).mean()
            if cur_test > best_test_acc:
                best_test_acc = cur_test
                best_alpha = alpha.copy()
                best_b = b_val.copy()

        if primal_res < tol:
            if verbose:
                print(f"  Converged at iter {k}")
            break

    info = {'n_iter': k + 1, 'primal_res': primal_res,
            'train_acc': train_acc, 'test_acc': test_acc,
            'best_test_acc': best_test_acc}
    return alpha, b_val, info


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
