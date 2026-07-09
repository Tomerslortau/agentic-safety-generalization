"""System and task-matrix (Q) generation shared by both sub-experiments.

Two families of generators, both previously duplicated across the imitation and
Lipschitz experiments:

  * ``make_random_matrix`` / ``make_random_psd`` — the draw used by the imitation
    experiment's data generation (A, B rescaled to a target spectral norm; R = M Mᵀ
    rescaled to a target Frobenius norm; D = I).
  * ``generate_system_general`` / ``generate_system_commuting`` +
    ``sample_Qs_general`` — the draw used by the Lipschitz-ratio experiment
    (spectral-norm-controlled A, B, D; PSD R with a target minimum eigenvalue;
    PSD task matrices Q with controlled Frobenius norm and minimum eigenvalue).

NumPy only.
"""

import numpy as np
import numpy.linalg as la

from .solvers import sym


# ── spectral-norm helper ─────────────────────────────────────────────────────

def scale_to_spectral_norm(M, target):
    """Scale M so that ||M||_2 == target (or target*I if M is numerically zero)."""
    s = la.svd(M, compute_uv=False)[0]
    if s <= 0:
        return target * np.eye(M.shape[0])
    return (target / s) * M


# ── imitation-experiment draw (A,B spectral-norm; R,Q Frobenius PSD) ──────────

def make_random_matrix(n, m, spectral_norm, rng, sigma=5.0):
    """Gaussian matrix rescaled to a target spectral norm."""
    M = rng.normal(0, sigma, (n, m))
    M *= spectral_norm / np.linalg.norm(M, 2)
    return M


def make_random_psd(n, rng, frob_range=(0.25, 1.0)):
    """PSD matrix M Mᵀ rescaled to a Frobenius norm drawn uniformly from frob_range."""
    M = rng.normal(0, 1, (n, n))
    S = M @ M.T
    target_frob = rng.uniform(*frob_range)
    S *= target_frob / np.linalg.norm(S, 'fro')
    return S


# ── Lipschitz-experiment system draws ─────────────────────────────────────────

def _draw_R_with_min_eig(n, min_eig_R, rng):
    """Draw R (PSD) rescaled to the requested minimum eigenvalue."""
    M_R = rng.normal(size=(n, n))
    R_psd = M_R.T @ M_R
    if min_eig_R == 0.0:
        return np.zeros_like(R_psd)
    lambda_min_R_psd = np.linalg.eigvalsh(R_psd).min()
    if lambda_min_R_psd == 0:
        return min_eig_R * np.eye(n)
    return (min_eig_R / lambda_min_R_psd) * R_psd


def _validate_alignment_constant(alignment_constant):
    if not 0.0 < alignment_constant <= 1.0:
        raise ValueError("alignment_constant must be in (0, 1]")


def _draw_diagonal_with_norm(n, norm, rng, first_min_abs=0.0):
    """Draw diagonal entries with exact spectral norm and a first-entry lower bound."""
    if norm < 0.0:
        raise ValueError("norm must be nonnegative")
    if n <= 0:
        raise ValueError("n must be positive")

    first_min_abs = max(0.0, float(first_min_abs))
    if first_min_abs > norm + 1e-12:
        raise ValueError("first-entry lower bound cannot exceed the requested norm")
    if norm == 0.0:
        return np.zeros(n)

    entries = rng.uniform(-norm, norm, size=n)
    first_abs = norm if first_min_abs >= norm else rng.uniform(first_min_abs, norm)
    entries[0] = rng.choice([-1.0, 1.0]) * first_abs

    norm_idx = 0 if first_abs == norm or n == 1 else rng.integers(1, n)
    entries[norm_idx] = rng.choice([-1.0, 1.0]) * norm
    return entries


def _draw_aligned_R_eigendecomposition(n, min_eig_R, rng):
    """Ordered eigendecomposition of a drawn R (used as the commuting basis)."""
    R_raw = _draw_R_with_min_eig(n, min_eig_R, rng)
    eigvals_R, V = np.linalg.eigh(R_raw)
    return eigvals_R, V


def generate_system_general(n, norm_A, norm_B, norm_D, min_eig_R, rng):
    """General system: A,B,D with prescribed spectral norms; R PSD with min eig."""
    A = scale_to_spectral_norm(rng.normal(size=(n, n)), norm_A)
    B = scale_to_spectral_norm(rng.normal(size=(n, n)), norm_B)
    D = scale_to_spectral_norm(rng.normal(size=(n, n)), norm_D)
    R = _draw_R_with_min_eig(n, min_eig_R, rng)
    return A, B, D, R


def generate_system_commuting(n, norm_A, norm_B, norm_D, min_eig_R,
                              alignment_constant, rng):
    """Commuting system: A,B,D,R diagonal in the eigenbasis of R, with alignment."""
    _validate_alignment_constant(alignment_constant)
    if min_eig_R <= 0.0:
        raise ValueError("commuting experiments require min_eig_R > 0")

    eigvals_R, V = _draw_aligned_R_eigendecomposition(n, min_eig_R, rng)

    first_A_min_abs = (norm_A + alignment_constant - 1.0) / alignment_constant
    first_B_min_abs = norm_B * alignment_constant

    diag_A = _draw_diagonal_with_norm(n, norm_A, rng, first_A_min_abs)
    diag_B = _draw_diagonal_with_norm(n, norm_B, rng, first_B_min_abs)
    diag_D = _draw_diagonal_with_norm(n, norm_D, rng)

    A = V @ np.diag(diag_A) @ V.T
    B = V @ np.diag(diag_B) @ V.T
    D = V @ np.diag(diag_D) @ V.T
    R = V @ np.diag(eigvals_R) @ V.T
    return sym(A), sym(B), sym(D), sym(R)


# ── Lipschitz-experiment task-matrix (Q) sampling ─────────────────────────────

def sample_psd_unitfro_min_eig_concentrated(n, rng, q0=0.0, norm=1.0,
                                             require_exact_norm=False):
    """PSD Q with min eigenvalue >= q0 and a controlled Frobenius norm.

    If q0 == 0: Q = M Mᵀ / ||M Mᵀ||_F scaled by a radius (uniform in [0, norm], or
    exactly ``norm``). If q0 > 0: eigenvalues [q0,...,q0, sqrt(r² - (n-1) q0²)] in a
    random orthogonal basis.
    """
    q0 = float(q0)
    norm = float(norm)

    if norm < 0.0:
        raise ValueError("norm must be nonnegative")
    if norm == 0.0:
        if q0 > 0.0:
            raise ValueError("cannot satisfy q0 > 0 with norm == 0")
        return np.zeros((n, n))

    if q0 == 0.0:
        M = rng.normal(size=(n, n))
        Q_psd = M.T @ M
        fro = np.linalg.norm(Q_psd, "fro")
        Q = np.eye(n) / np.sqrt(n) if fro == 0 else Q_psd / fro
        radius = norm if require_exact_norm else rng.uniform(0.0, norm)
        return sym(radius * Q)

    q0_max = norm / np.sqrt(n)
    if q0 >= q0_max:
        q0 = q0_max

    radius_min = np.sqrt(n) * q0
    radius = norm if require_exact_norm else rng.uniform(radius_min, norm)
    lam_last = np.sqrt(max(radius ** 2 - (n - 1) * (q0 ** 2), 1e-32))
    lam = np.concatenate([np.full(n - 1, q0), [lam_last]])

    Z = rng.normal(size=(n, n))
    Qq, _ = la.qr(Z)
    if np.linalg.det(Qq) < 0:
        Qq[:, 0] = -Qq[:, 0]
    return sym(Qq @ np.diag(lam) @ Qq.T)


def sample_Qs_general(n, n_Q, rng, q0=0.0, norm=1.0, require_exact_norm=False):
    """Sample a list of ``n_Q`` PSD task matrices (see the single-sample helper)."""
    return [
        sample_psd_unitfro_min_eig_concentrated(
            n, rng, q0=q0, norm=norm, require_exact_norm=require_exact_norm
        )
        for _ in range(n_Q)
    ]
