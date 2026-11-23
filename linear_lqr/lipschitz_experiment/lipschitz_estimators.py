"""Memory-bounded pairwise estimator and the exact local Lipschitz constant of K_unsafe."""

import os
import sys

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.solvers import solve_lqr

EPS = 1e-12


def _block_max(Qa, Ka, Qb=None, Kb=None, eps=EPS):
    """Largest ||K_i-K_j||_F / (||Q_i-Q_j||_F + eps) between two blocks (or within one)."""
    if Qb is None:
        if len(Qa) < 2:
            return -np.inf
        iu = np.triu_indices(len(Qa), k=1)
        return float((cdist(Ka, Ka)[iu] / (cdist(Qa, Qa)[iu] + eps)).max())
    if len(Qa) == 0 or len(Qb) == 0:
        return -np.inf
    return float((cdist(Ka, Kb) / (cdist(Qa, Qb) + eps)).max())


def pairwise_max(Qs, Ks, block_size=2000, eps=EPS):
    """max_{i<j} ||K_i-K_j||_F / (||Q_i-Q_j||_F + eps), computed in blocks.

    Peak memory is about 3 * block_size^2 * 8 bytes, independent of the task count.
    """
    Q_flat = np.asarray(Qs).reshape(len(Qs), -1)
    K_flat = np.asarray(Ks).reshape(len(Ks), -1)
    N = len(Q_flat)
    if N < 2:
        return float("nan")
    starts = range(0, N, block_size)
    best = -np.inf
    for i in starts:
        Qa, Ka = Q_flat[i:i + block_size], K_flat[i:i + block_size]
        best = max(best, _block_max(Qa, Ka, eps=eps))
        for j in starts:
            if j > i:
                best = max(best, _block_max(Qa, Ka, Q_flat[j:j + block_size],
                                            K_flat[j:j + block_size], eps=eps))
    return float(best)


def pairwise_max_with_extra(Qs, Ks, n_base, block_size=2000, eps=EPS):
    """Return (max over base pairs, max over all pairs); tasks[n_base:] are the extra batch."""
    Qs, Ks = np.asarray(Qs), np.asarray(Ks)
    base = pairwise_max(Qs[:n_base], Ks[:n_base], block_size=block_size, eps=eps)
    if n_base == len(Qs):
        return base, base
    Q_flat = Qs.reshape(len(Qs), -1)
    K_flat = Ks.reshape(len(Ks), -1)
    extra = _block_max(Q_flat[n_base:], K_flat[n_base:], eps=eps)
    for i in range(0, n_base, block_size):
        extra = max(extra, _block_max(Q_flat[i:i + block_size], K_flat[i:i + block_size],
                                      Q_flat[n_base:], K_flat[n_base:], eps=eps))
    finite_base = base if np.isfinite(base) else -np.inf
    return base, float(max(finite_base, extra))


def symmetric_basis(m):
    """Frobenius-orthonormal basis of the m x m symmetric matrices."""
    basis = []
    for i in range(m):
        for j in range(i, m):
            E = np.zeros((m, m))
            if i == j:
                E[i, i] = 1.0
            else:
                E[i, j] = E[j, i] = 1.0 / np.sqrt(2.0)
            basis.append(E)
    return np.array(basis)


def lqr_jacobian(A, B, R, P, K, basis):
    """Jacobian of Q -> K_lqr(Q) on ``basis``, as an (m_u*m, len(basis)) matrix.

    With K = -X^{-1} B' P A, X = R + B'PB and A_cl = A + BK:
        dP = dQ + A_cl' dP A_cl,    dK = -X^{-1} B' dP A_cl.
    """
    m = A.shape[0]
    A_cl = A + B @ K
    X = R + B.T @ P @ B
    lyap = np.eye(m * m) - np.kron(A_cl.T, A_cl.T)
    rhs = basis.reshape(len(basis), -1, order="F").T
    dPs = np.linalg.solve(lyap, rhs).T.reshape(len(basis), m, m, order="F")
    dKs = -np.linalg.solve(X, B.T @ dPs @ A_cl)
    return dKs.reshape(len(basis), -1, order="F").T


def jacobian_max(A, B, R, Qs, Ps, Ks, n_base):
    """Return (max over base tasks, max over all tasks) of the Jacobian's top singular value."""
    basis = symmetric_basis(A.shape[0])
    norms = np.array([np.linalg.norm(lqr_jacobian(A, B, R, P, K, basis), 2)
                      for P, K in zip(Ps, Ks)])
    return float(norms[:n_base].max()), float(norms.max())


def validate_jacobian(A, B, R, Qs, rng, n_tasks=3, n_directions=3, step=1e-6):
    """Largest relative error of the Jacobian against central finite differences.

    Also checks the Kronecker solve against scipy's discrete Lyapunov solver. Tasks
    should have a positive minimum eigenvalue so that Q +/- step*E stays PSD.
    """
    m = A.shape[0]
    basis = symmetric_basis(m)
    worst_fd, worst_lyap = 0.0, 0.0
    for idx in rng.choice(len(Qs), size=min(n_tasks, len(Qs)), replace=False):
        Q = Qs[idx]
        P, K = solve_lqr(A, B, Q, R)
        J = lqr_jacobian(A, B, R, P, K, basis)
        _, _, Vh = np.linalg.svd(J, full_matrices=False)
        coeffs = [Vh[0]]
        for _ in range(n_directions - 1):
            c = rng.normal(size=len(basis))
            coeffs.append(c / np.linalg.norm(c))
        A_cl = A + B @ K
        X = R + B.T @ P @ B
        for c in coeffs:
            dQ = np.tensordot(c, basis, axes=1)
            dK_fd = (solve_lqr(A, B, Q + step * dQ, R)[1]
                     - solve_lqr(A, B, Q - step * dQ, R)[1]) / (2.0 * step)
            dK_jac = (J @ c).reshape(K.shape, order="F")
            scale = max(np.linalg.norm(dK_jac), 1e-12)
            worst_fd = max(worst_fd, float(np.linalg.norm(dK_fd - dK_jac) / scale))
            dP = solve_discrete_lyapunov(A_cl.T, dQ)
            dK_lyap = -np.linalg.solve(X, B.T @ dP @ A_cl)
            worst_lyap = max(worst_lyap, float(np.linalg.norm(dK_lyap - dK_jac) / scale))
    return worst_fd, worst_lyap
