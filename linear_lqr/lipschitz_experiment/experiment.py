"""Random systems, Lipschitz estimators and the run for one system.

Systems control only norms: A and B are Gaussian rescaled to a spectral norm, R is a
rescaled Wishart matrix with prescribed ||R^{-1}||_2, and D is ||D||_2 times a
Haar-random orthogonal matrix, so kappa(D) = 1.
"""

import os
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import ortho_group

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.solvers import compute_K_hinf, solve_lqr, sym
from core.systems import _draw_R_with_min_eig, sample_Qs_general, scale_to_spectral_norm

import config as C

EPS = 1e-12
NORM_TOL = 1e-9
PSD_TOL = 1e-12
LEMMA_TOL = 1e-9
FD_TOL = 1e-3
MAX_REDRAWS = 100


class InvalidDraw(ValueError):
    """A drawn system or task violates the structural assumptions."""


class HardCheckError(RuntimeError):
    """A check that can only fail through a bug; the run must stop."""


# ── Systems and tasks ─────────────────────────────────────────────────────────

def build_jobs(cfg):
    return [dict(system_id=f"m={m}_A={a:g}_B={b:g}_Rinv={r:g}_D={cfg.norm_D:g}_seed={s}",
                 m=m, norm_A=a, norm_B=b, norm_Rinv=r, norm_D=cfg.norm_D, seed_index=s,
                 grid_index=(i_A, i_B, i_R))
            for m in cfg.dims
            for i_R, r in enumerate(cfg.norm_Rinv)
            for i_B, b in enumerate(cfg.norm_B)
            for i_A, a in enumerate(cfg.norm_A)
            for s in range(cfg.n_seeds)]


def draw_system(job, rng):
    m = job["m"]
    A = scale_to_spectral_norm(rng.normal(size=(m, m)), job["norm_A"])
    B = scale_to_spectral_norm(rng.normal(size=(m, m)), job["norm_B"])
    R = sym(_draw_R_with_min_eig(m, 1.0 / job["norm_Rinv"], rng))
    U = ortho_group.rvs(dim=m, random_state=rng)
    V = ortho_group.rvs(dim=m, random_state=rng)
    return A, B, job["norm_D"] * (U @ V.T), R


def check_system(A, B, D, R, job):
    if not all(np.all(np.isfinite(M)) for M in (A, B, D, R)):
        raise InvalidDraw("non-finite system matrix")
    a = np.linalg.norm(A, 2)
    if not 0.0 < a < 1.0 or abs(a - job["norm_A"]) > NORM_TOL:
        raise InvalidDraw(f"||A||_2={a}")
    b = np.linalg.norm(B, 2)
    if b <= 0.0 or abs(b - job["norm_B"]) > NORM_TOL:
        raise InvalidDraw(f"||B||_2={b}")
    lam_min = np.linalg.eigvalsh(sym(R)).min()
    if lam_min <= 0.0 or abs(1.0 / lam_min - job["norm_Rinv"]) > NORM_TOL * max(1.0, job["norm_Rinv"]):
        raise InvalidDraw(f"lambda_min(R)={lam_min}")
    sv_D = np.linalg.svd(D, compute_uv=False)
    if sv_D[-1] <= 0.0 or abs(sv_D[0] - job["norm_D"]) > NORM_TOL:
        raise InvalidDraw(f"singular values of D: {sv_D[[0, -1]]}")


def theory_constants(A, B, D, R):
    """Constants of Assumption 1, Lemma 1 and Theorem 1(ii)."""
    a = float(np.linalg.norm(A, 2))
    norm_B = float(np.linalg.norm(B, 2))
    norm_Rinv = float(np.linalg.norm(np.linalg.inv(R), 2))
    kappa_D = float(np.linalg.norm(D, 2) * np.linalg.norm(np.linalg.inv(D), 2))
    alpha = float(np.linalg.norm(np.linalg.solve(R, B.T), 2) / (norm_B * norm_Rinv))
    beta = float(1.0 + 0.5 * np.sqrt(norm_B ** 2 * norm_Rinv / (1.0 - a ** 2)))
    a_beta = a * beta
    holds = a_beta < 1.0
    coef = alpha * (1.0 - a) ** 2 * (1.0 - a_beta ** 2) / (kappa_D ** 3 * a_beta)
    c_unsafe = norm_B * norm_Rinv * a_beta / (1.0 - a_beta ** 2) if holds else float("nan")
    return dict(alpha=alpha, kappa_D=kappa_D, beta=beta, assumption_1_holds=holds,
                coef=float(coef), c_unsafe=float(c_unsafe))


def draw_tasks(job, cfg, B, R, rng):
    """The three task batches followed by the rank-one batch eps * v v^T."""
    m, n = job["m"], cfg.n_per_batch(job["m"])
    base = sample_Qs_general(m, n, rng, q0=cfg.q_min_eig, norm=1.0, require_exact_norm=True)
    base += sample_Qs_general(m, n, rng, q0=0.0, norm=1.0)
    base += sample_Qs_general(m, n, rng, q0=0.0, norm=job["norm_A"])
    v = rng.normal(size=(2 * m, m))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    top_right = np.linalg.svd(np.linalg.solve(R, B.T))[2][0]
    directions = np.vstack([v, top_right])
    return base + [eps * np.outer(u, u) for eps in cfg.rank_one_eps for u in directions], len(base)


def sanitize_tasks(tasks):
    """Symmetrize, clip round-off negative eigenvalues and enforce ||Q||_F <= 1."""
    Qs = np.array([sym(Q) for Q in tasks])
    eigvals = np.linalg.eigvalsh(Qs)
    negative = np.flatnonzero(eigvals[:, 0] < 0.0)
    for i in negative:
        if eigvals[i, 0] < -1e-9:
            raise InvalidDraw(f"task {i} has min eigenvalue {eigvals[i, 0]}")
        w, V = np.linalg.eigh(Qs[i])
        Qs[i] = sym((V * np.maximum(w, 0.0)) @ V.T)
    fro = np.linalg.norm(Qs, axis=(1, 2))
    over = np.flatnonzero(fro > 1.0)
    if over.size and fro[over].max() > 1.0 + 1e-9:
        raise InvalidDraw(f"task Frobenius norm {fro[over].max()}")
    Qs[over] /= (fro[over] * (1.0 + 1e-15))[:, None, None]
    if np.linalg.eigvalsh(Qs)[:, 0].min() < -PSD_TOL:
        raise InvalidDraw("task not PSD after clipping")
    if np.linalg.norm(Qs, axis=(1, 2)).max() > 1.0:
        raise InvalidDraw("task outside the unit Frobenius ball")
    return Qs, int(negative.size), int(over.size)


def draw_valid(job, cfg):
    """Draw a system and tasks, moving to the next seed offset if a draw is invalid."""
    for attempt in range(MAX_REDRAWS):
        seq = np.random.SeedSequence([cfg.base_seed, job["m"], *job["grid_index"],
                                      job["seed_index"], attempt])
        system_seq, task_seq = seq.spawn(2)
        try:
            A, B, D, R = draw_system(job, np.random.default_rng(system_seq))
            check_system(A, B, D, R, job)
            rng = np.random.default_rng(task_seq)
            tasks, n_base = draw_tasks(job, cfg, B, R, rng)
            Qs, n_clipped, n_renormalized = sanitize_tasks(tasks)
        except InvalidDraw:
            continue
        return (A, B, D, R), Qs, n_base, attempt, n_clipped, n_renormalized, rng
    raise RuntimeError(f"{job['system_id']}: no valid draw in {MAX_REDRAWS} attempts")


# ── Lipschitz estimators ──────────────────────────────────────────────────────

def _max_ratio(Qa, Ka, Qb=None, Kb=None):
    """Largest ||K_i-K_j||_F / (||Q_i-Q_j||_F + eps) within one block or between two."""
    if Qb is None:
        if len(Qa) < 2:
            return -np.inf
        iu = np.triu_indices(len(Qa), k=1)
        return float((cdist(Ka, Ka)[iu] / (cdist(Qa, Qa)[iu] + EPS)).max())
    if len(Qa) == 0 or len(Qb) == 0:
        return -np.inf
    return float((cdist(Ka, Kb) / (cdist(Qa, Qb) + EPS)).max())


def pairwise_lipschitz(Qs, Ks, n_base, block_size):
    """Pairwise estimate over the first n_base tasks, and over all tasks."""
    Q, K = Qs.reshape(len(Qs), -1), Ks.reshape(len(Ks), -1)
    blocks = [slice(i, min(i + block_size, n_base)) for i in range(0, n_base, block_size)]
    base = -np.inf
    for index, a in enumerate(blocks):
        base = max(base, _max_ratio(Q[a], K[a]))
        for b in blocks[index + 1:]:
            base = max(base, _max_ratio(Q[a], K[a], Q[b], K[b]))
    extra = _max_ratio(Q[n_base:], K[n_base:])
    for a in blocks:
        extra = max(extra, _max_ratio(Q[a], K[a], Q[n_base:], K[n_base:]))
    return base, max(base, extra)


def symmetric_basis(m):
    """Frobenius-orthonormal basis of the m x m symmetric matrices."""
    basis = []
    for i in range(m):
        for j in range(i, m):
            E = np.zeros((m, m))
            E[i, j] = E[j, i] = 1.0 if i == j else 1.0 / np.sqrt(2.0)
            basis.append(E)
    return np.array(basis)


def lqr_jacobian(A, B, R, P, K, basis):
    """Jacobian of Q -> K_lqr(Q) on ``basis``.

    With X = R + B'PB and A_cl = A + BK:
        dP = dQ + A_cl' dP A_cl,    dK = -X^{-1} B' dP A_cl.
    """
    m = A.shape[0]
    A_cl = A + B @ K
    X = R + B.T @ P @ B
    lyapunov = np.eye(m * m) - np.kron(A_cl.T, A_cl.T)
    rhs = basis.reshape(len(basis), -1, order="F").T
    dPs = np.linalg.solve(lyapunov, rhs).T.reshape(len(basis), m, m, order="F")
    dKs = -np.linalg.solve(X, B.T @ dPs @ A_cl)
    return dKs.reshape(len(basis), -1, order="F").T


def jacobian_fd_error(A, B, R, Qs, rng, n_tasks=3, n_directions=3, step=1e-6):
    """Largest relative error of the Jacobian against central finite differences."""
    basis = symmetric_basis(A.shape[0])
    worst = 0.0
    for index in rng.choice(len(Qs), size=min(n_tasks, len(Qs)), replace=False):
        Q = Qs[index]
        P, K = solve_lqr(A, B, Q, R)
        J = lqr_jacobian(A, B, R, P, K, basis)
        directions = [np.linalg.svd(J, full_matrices=False)[2][0]]
        for _ in range(n_directions - 1):
            c = rng.normal(size=len(basis))
            directions.append(c / np.linalg.norm(c))
        for c in directions:
            dQ = np.tensordot(c, basis, axes=1)
            dK_fd = (solve_lqr(A, B, Q + step * dQ, R)[1] - solve_lqr(A, B, Q - step * dQ, R)[1]) / (2.0 * step)
            dK = (J @ c).reshape(K.shape, order="F")
            worst = max(worst, float(np.linalg.norm(dK_fd - dK) / max(np.linalg.norm(dK), 1e-12)))
    return worst


# ── One system ────────────────────────────────────────────────────────────────

def _ratio(numerator, denominator):
    ok = np.isfinite(numerator) and np.isfinite(denominator) and denominator > 0
    return float(numerator / denominator) if ok else float("nan")


def run_system(job, cfg):
    """Run one system and return its flat result record."""
    start = time.time()
    (A, B, D, R), Qs, n_base, attempt, n_clipped, n_renormalized, rng = draw_valid(job, cfg)
    constants = theory_constants(A, B, D, R)

    solved = [solve_lqr(A, B, Q, R) for Q in Qs]
    Ps = np.array([s[0] for s in solved])
    K_unsafe = np.array([s[1] for s in solved])

    K_safe = np.full_like(K_unsafe, np.nan)
    gammas = np.full(len(Qs), np.nan)
    for i, (Q, P, K) in enumerate(zip(Qs, Ps, K_unsafe)):
        K_i, gamma = compute_K_hinf(A, B, D, Q, R, P_lqr=P, K_lqr=K, on_infeasible="fallback_lqr",
                                    **C.HINF_SEARCH)
        if not np.isnan(gamma):
            K_safe[i], gammas[i] = K_i, gamma
    feasible = np.isfinite(gammas)
    at_floor = feasible & (gammas <= C.HINF_SEARCH["gamma_floor"])

    pair_base, pair_all = pairwise_lipschitz(Qs, K_unsafe, n_base, cfg.block_size)
    L_safe_base, L_safe = pairwise_lipschitz(Qs[feasible], K_safe[feasible],
                                             int(feasible[:n_base].sum()), cfg.block_size)
    L_safe_no_floor = L_safe
    if at_floor.any():
        keep = feasible & ~at_floor
        _, L_safe_no_floor = pairwise_lipschitz(Qs[keep], K_safe[keep], int(keep[:n_base].sum()),
                                                cfg.block_size)

    basis = symmetric_basis(job["m"])
    jacobian = np.array([np.linalg.norm(lqr_jacobian(A, B, R, P, K, basis), 2)
                         for P, K in zip(Ps, K_unsafe)])
    jac_base, jac_all = float(jacobian[:n_base].max()), float(jacobian.max())
    fd_error = jacobian_fd_error(A, B, R, Qs[:cfg.n_per_batch(job["m"])], rng)
    L_unsafe_base, L_unsafe = max(pair_base, jac_base), max(pair_all, jac_all)

    if fd_error > FD_TOL:
        raise HardCheckError(f"{job['system_id']}: Jacobian validation failed ({fd_error:.2e})")
    if constants["assumption_1_holds"] and L_unsafe > constants["c_unsafe"] * (1.0 + LEMMA_TOL):
        raise HardCheckError(f"{job['system_id']}: Lemma 1 violated: L_unsafe={L_unsafe:.6g} "
                             f"> c_unsafe={constants['c_unsafe']:.6g}")

    found = gammas[feasible]
    record = {k: v for k, v in job.items() if k != "grid_index"}
    record.update(constants)
    record.update(
        n_redraws=attempt,
        n_tasks=len(Qs),
        n_tasks_clipped=n_clipped,
        n_tasks_renormalized=n_renormalized,
        n_infeasible=int((~feasible).sum()),
        n_gamma_at_floor=int(at_floor.sum()),
        gamma_min=float(found.min()) if found.size else float("nan"),
        gamma_median=float(np.median(found)) if found.size else float("nan"),
        gamma_max=float(found.max()) if found.size else float("nan"),
        L_safe=L_safe,
        L_safe_no_rank_one=L_safe_base,
        L_unsafe=L_unsafe,
        L_unsafe_no_rank_one=L_unsafe_base,
        L_unsafe_pairwise=pair_all,
        L_unsafe_jacobian=jac_all,
        ratio=_ratio(L_safe, L_unsafe),
        ratio_pairwise_only=_ratio(L_safe, pair_all),
        ratio_no_rank_one=_ratio(L_safe_base, L_unsafe_base),
        ratio_without_floor_tasks=_ratio(L_safe_no_floor, L_unsafe),
        jacobian_fd_rel_error=fd_error,
        seconds=time.time() - start,
    )
    return record
