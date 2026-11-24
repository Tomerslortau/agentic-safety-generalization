"""System and task-matrix (Q) generation shared by both sub-experiments.

  * ``make_random_matrix`` / ``make_random_psd`` — the draw used by the imitation
    experiment's data generation (A, B rescaled to a target spectral norm; R = M Mᵀ
    rescaled to a target Frobenius norm; D = I).
  * ``scale_to_spectral_norm``, ``_draw_R_with_min_eig`` and ``sample_Qs_general``
    — building blocks of the Lipschitz-ratio experiment's draw (spectral-norm
    rescaling; PSD R with a target minimum eigenvalue; PSD task matrices Q with
    controlled Frobenius norm and minimum eigenvalue).

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
