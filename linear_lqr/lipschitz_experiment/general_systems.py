"""Generic random systems and theory constants for the general-systems experiment.

Systems control only norms: A and B are Gaussian matrices rescaled to a target
spectral norm, R is a rescaled Wishart matrix with a prescribed ||R^{-1}||_2, and D is
a Haar-random orthogonal matrix times ||D||_2, so kappa(D) = 1. There is no alignment
constraint and no commuting structure.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.stats import ortho_group

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.solvers import sym
from core.systems import _draw_R_with_min_eig, sample_Qs_general, scale_to_spectral_norm

NORM_TOL = 1e-9
PSD_TOL = 1e-12
MAX_REDRAWS = 100


class StructuralAssumptionError(ValueError):
    """A drawn system or task violates the structural assumptions."""


@dataclass
class System:
    A: np.ndarray
    B: np.ndarray
    D: np.ndarray
    R: np.ndarray


def system_seed_sequence(base_seed, m, grid_index, seed_index, attempt=0):
    """Independent stream per (dimension, grid point, seed, redraw attempt)."""
    return np.random.SeedSequence([base_seed, m, *grid_index, seed_index, attempt])


def draw_haar_orthogonal(m, rng):
    return ortho_group.rvs(dim=m, random_state=rng) if m > 1 else np.array([[rng.choice([-1.0, 1.0])]])


def draw_system(m, norm_A, norm_B, norm_Rinv, norm_D, rng):
    A = scale_to_spectral_norm(rng.normal(size=(m, m)), norm_A)
    B = scale_to_spectral_norm(rng.normal(size=(m, m)), norm_B)
    R = sym(_draw_R_with_min_eig(m, 1.0 / norm_Rinv, rng))
    U = draw_haar_orthogonal(m, rng)
    V = draw_haar_orthogonal(m, rng)
    D = norm_D * (U @ V.T)
    return System(A=A, B=B, D=D, R=R)


def assert_system(system, norm_A, norm_B, norm_Rinv, norm_D):
    """Raise StructuralAssumptionError unless the draw meets every assumption."""
    A, B, D, R = system.A, system.B, system.D, system.R
    if not all(np.all(np.isfinite(M)) for M in (A, B, D, R)):
        raise StructuralAssumptionError("non-finite system matrix")
    a = np.linalg.norm(A, 2)
    if not 0.0 < a < 1.0 or abs(a - norm_A) > NORM_TOL:
        raise StructuralAssumptionError(f"||A||_2={a}")
    b = np.linalg.norm(B, 2)
    if b <= 0.0 or abs(b - norm_B) > NORM_TOL:
        raise StructuralAssumptionError(f"||B||_2={b}")
    lam_min = np.linalg.eigvalsh(sym(R)).min()
    if lam_min <= 0.0 or abs(1.0 / lam_min - norm_Rinv) > NORM_TOL * max(1.0, norm_Rinv):
        raise StructuralAssumptionError(f"lambda_min(R)={lam_min}")
    sv_D = np.linalg.svd(D, compute_uv=False)
    if sv_D[-1] <= 0.0 or abs(sv_D[0] - norm_D) > NORM_TOL:
        raise StructuralAssumptionError(f"singular values of D: {sv_D[[0, -1]]}")


def theory_constants(system):
    """Per-system constants of Assumption 1, Lemma 1 and Theorem 1(ii)."""
    A, B, D, R = system.A, system.B, system.D, system.R
    a = float(np.linalg.norm(A, 2))
    norm_B = float(np.linalg.norm(B, 2))
    norm_Rinv = float(np.linalg.norm(np.linalg.inv(R), 2))
    norm_D = float(np.linalg.norm(D, 2))
    norm_Dinv = float(np.linalg.norm(np.linalg.inv(D), 2))
    kappa_D = norm_D * norm_Dinv
    alpha = float(np.linalg.norm(np.linalg.solve(R, B.T), 2) / (norm_B * norm_Rinv))
    beta = 1.0 + 0.5 * np.sqrt(norm_B ** 2 * norm_Rinv / (1.0 - a ** 2))
    a_beta = a * beta
    holds = bool(a_beta < 1.0)
    coef = alpha * (1.0 - a) ** 2 * (1.0 - a_beta ** 2) / (kappa_D ** 3 * a_beta)
    c_unsafe = norm_B * norm_Rinv * a_beta / (1.0 - a_beta ** 2) if holds else float("nan")
    return dict(
        measured_norm_A=a, measured_norm_B=norm_B, measured_norm_Rinv=norm_Rinv,
        measured_norm_D=norm_D, measured_norm_Dinv=norm_Dinv, alpha=alpha,
        kappa_D=kappa_D, beta=float(beta), a_beta=float(a_beta),
        assumption_1_holds=holds, coef=float(coef), c_unsafe=float(c_unsafe),
    )


def rank_one_tasks(system, m, rng, epsilons=(1e-3, 1e-2)):
    """Tasks eps * v v^T for 2m random unit v and the top right singular vector of R^{-1}B^T."""
    random_v = rng.normal(size=(2 * m, m))
    random_v /= np.linalg.norm(random_v, axis=1, keepdims=True)
    _, _, Vh = np.linalg.svd(np.linalg.solve(system.R, system.B.T))
    directions = np.vstack([random_v, Vh[0]])
    return [eps * np.outer(v, v) for eps in epsilons for v in directions]


def sanitize_tasks(tasks):
    """Symmetrize, clip round-off negative eigenvalues and enforce ||Q||_F <= 1.

    Returns (array of tasks, number of tasks clipped, number of tasks renormalized).
    Violations larger than round-off raise StructuralAssumptionError.
    """
    Qs = np.array([sym(Q) for Q in tasks])
    n_clipped = 0
    eigvals = np.linalg.eigvalsh(Qs)
    negative = np.flatnonzero(eigvals[:, 0] < 0.0)
    for i in negative:
        if eigvals[i, 0] < -1e-9:
            raise StructuralAssumptionError(f"task {i} has min eigenvalue {eigvals[i, 0]}")
        w, V = np.linalg.eigh(Qs[i])
        Qs[i] = sym((V * np.maximum(w, 0.0)) @ V.T)
        n_clipped += 1
    fro = np.linalg.norm(Qs, axis=(1, 2))
    over = np.flatnonzero(fro > 1.0)
    if over.size and fro[over].max() > 1.0 + 1e-9:
        raise StructuralAssumptionError(f"task Frobenius norm {fro[over].max()}")
    Qs[over] /= (fro[over] * (1.0 + 1e-15))[:, None, None]
    if np.linalg.eigvalsh(Qs)[:, 0].min() < -PSD_TOL:
        raise StructuralAssumptionError("task not PSD after clipping")
    if np.linalg.norm(Qs, axis=(1, 2)).max() > 1.0:
        raise StructuralAssumptionError("task outside the unit Frobenius ball")
    return Qs, n_clipped, int(over.size)


def sample_base_tasks(m, n_per_batch, norm_A, rng, q_min_eig_hinf=1e-1, q_min_eig_lqr=0.0):
    """The three existing task batches (see Appendix K.1.1)."""
    tasks = sample_Qs_general(m, n_per_batch, rng, q0=q_min_eig_hinf, norm=1.0,
                              require_exact_norm=True)
    tasks.extend(sample_Qs_general(m, n_per_batch, rng, q0=q_min_eig_lqr, norm=1.0))
    tasks.extend(sample_Qs_general(m, n_per_batch, rng, q0=q_min_eig_lqr, norm=norm_A))
    return tasks
