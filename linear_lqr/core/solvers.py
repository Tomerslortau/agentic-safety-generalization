"""Unified LQR (DARE) and H-infinity (DGARE) controller solvers.

This module merges the two previously-duplicated solver implementations from the
linear-imitation experiment and the Lipschitz-ratio experiment. Both computed the
exact same maps K*(Q):

    unsafe teacher  K_lqr(Q)  : discrete-time LQR via the DARE
    safe   teacher  K_hinf(Q) : discrete-time H-infinity state feedback via the
                                DGARE, found by searching for the smallest feasible
                                disturbance-attenuation level gamma.

Sign convention (both maps):  u = K x   (the sign is baked into K)
    K_lqr  = -(R + B'PB)^{-1} B'PA
    K_hinf = -[H^{-1} F]_{:n_u, :}
    closed loop: A + B K

H-infinity joint block-H fixed point:
    P = Q + A'PA - F'H^{-1}F
    H = [[R + B'PB,  B'PD        ],
         [D'PB,      D'PD - g^2 I]]
    F = [[B'PA], [D'PA]]

The two original experiments differed only in two behaviours, preserved here as
explicit flags rather than silently picking one:

  * ``on_infeasible`` — when no feasible gamma is found for a task:
        "raise"        (imitation experiment) so the task is dropped from the dataset;
        "fallback_lqr" (Lipschitz experiment) returns K_lqr with gamma = NaN.
  * ``require_closed_loop_stable`` — the imitation experiment rejected non-stabilizing
        DGARE fixed points (``A + B K`` spectral radius >= 1); the Lipschitz experiment
        did not. Default True (the stricter, more correct behaviour); the Lipschitz
        config passes False to reproduce the published figure exactly.

Tolerances are parameters too (imitation used HINF_TOL_P=1e-10, Lipschitz 1e-6).
NumPy + SciPy only.
"""

import numpy as np
from scipy.linalg import solve_discrete_are


# ── Tunable defaults ───────────────────────────────────────────────────────────
HINF_TOL_P          = 1e-10
HINF_MAX_ITER_P     = 500
HINF_TOL_GAMMA      = 1e-3
HINF_MAX_ITER_GAMMA = 30
HINF_GAMMA_INIT     = 1e-3
HINF_GAMMA_MAX      = 3.0
HINF_GAMMA_BRACKET  = 2.0
PD_TOL              = 1e-8
ND_TOL              = 1e-12


# ── Helpers ─────────────────────────────────────────────────────────────────────

def sym(M):
    return 0.5 * (M + M.T)


def is_posdef(M, tol=PD_TOL):
    """Return (ok, lambda_min) for the symmetric part of M."""
    w = np.linalg.eigvalsh(sym(M))
    return bool(np.all(w > tol)), float(w.min())


# ── LQR ───────────────────────────────────────────────────────────────────────

def solve_lqr(A, B, Q, R):
    """Discrete-time LQR via the DARE. Returns (P, K) with u = K x."""
    P = sym(solve_discrete_are(A, B, Q, R))
    K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return P, K


def compute_K_lqr(A, B, Q, R):
    """Convenience wrapper returning only the LQR gain K."""
    return solve_lqr(A, B, Q, R)[1]


# ── H-infinity joint block-H fixed-point iteration ──────────────────────────────

def solve_dgare(A, B, D, Q, R, gamma, P_init=None,
                tol=HINF_TOL_P, maxit=HINF_MAX_ITER_P,
                require_closed_loop_stable=True):
    """One DGARE solve at a fixed attenuation level gamma.

    Returns (ok, P, K). On any failure (invalid gamma, non-PD Suu, non-ND
    w-Schur complement, non-finite update, non-convergence, or -- when
    ``require_closed_loop_stable`` -- a non-stabilizing fixed point) returns
    (False, None, None).
    """
    if gamma <= 0.0 or not np.isfinite(gamma):
        return False, None, None

    n_u = B.shape[1]
    n_w = D.shape[1]
    Iw = np.eye(n_w)

    if P_init is None:
        P_init = solve_discrete_are(A, B, Q, R)
    P = sym(P_init.copy())

    def _build_HF(P):
        Suu = R + B.T @ P @ B
        Suw = B.T @ P @ D
        Swu = D.T @ P @ B
        Sww = D.T @ P @ D - (gamma ** 2) * Iw
        H = np.block([[Suu, Suw], [Swu, Sww]])
        F = np.vstack([B.T @ P @ A, D.T @ P @ A])
        return Suu, Suw, Swu, Sww, H, F

    def _feasible_step(P):
        Suu, Suw, Swu, Sww, H, F = _build_HF(P)
        ok_Suu, _ = is_posdef(Suu)
        if not ok_Suu:
            return None
        try:
            Suu_inv_Suw = np.linalg.solve(Suu, Suw)
        except np.linalg.LinAlgError:
            return None
        Schur_w = sym(Sww - Swu @ Suu_inv_Suw)
        if np.linalg.eigvalsh(Schur_w).max() >= -ND_TOL:
            return None
        try:
            Z = np.linalg.solve(H, F)
        except np.linalg.LinAlgError:
            return None
        return F, Z

    for _ in range(maxit):
        step = _feasible_step(P)
        if step is None:
            return False, None, None
        F, Z = step
        P_new = sym(Q + A.T @ P @ A - F.T @ Z)
        if not np.all(np.isfinite(P_new)):
            return False, None, None
        rel = np.linalg.norm(P_new - P, 'fro') / max(1.0, np.linalg.norm(P, 'fro'))
        P = P_new
        if rel < tol:
            break
    else:
        return False, None, None

    step = _feasible_step(P)
    if step is None:
        return False, None, None
    _, Z = step
    K = -Z[:n_u, :]
    if not np.all(np.isfinite(K)):
        return False, None, None

    if require_closed_loop_stable and \
            np.max(np.abs(np.linalg.eigvals(A + B @ K))) >= 1.0:
        return False, None, None

    return True, sym(P), K


# ── gamma search: bracket then bisect ───────────────────────────────────────────

def find_min_gamma_for_Q(A, B, D, Q, R,
                         gamma_floor=HINF_GAMMA_INIT,
                         gamma_cap=HINF_GAMMA_MAX,
                         bracket_factor=HINF_GAMMA_BRACKET,
                         tol_rel=HINF_TOL_GAMMA,
                         max_bisect=HINF_MAX_ITER_GAMMA,
                         tol_p=HINF_TOL_P,
                         maxit_p=HINF_MAX_ITER_P,
                         P_init=None,
                         require_closed_loop_stable=True):
    """Bracket-then-bisect search for the smallest feasible gamma.

    Returns (found, gamma_min, K) or (False, None, None) if no feasible
    gamma <= gamma_cap.
    """
    if P_init is None:
        P_init, _ = solve_lqr(A, B, Q, R)

    def _dgare(g):
        return solve_dgare(A, B, D, Q, R, g, P_init=P_init, tol=tol_p,
                           maxit=maxit_p,
                           require_closed_loop_stable=require_closed_loop_stable)

    lo = max(gamma_floor, 1e-12)
    hi = lo
    ok, _, K = _dgare(hi)
    if ok:
        return True, hi, K

    while hi < gamma_cap:
        lo = hi
        hi = hi * bracket_factor
        ok, _, K = _dgare(hi)
        if ok:
            break

    if not ok:
        return False, None, None

    if hi == lo:
        lo = max(hi / bracket_factor, 1e-12)

    best_K = K
    for _ in range(max_bisect):
        if (hi - lo) / max(1.0, hi) <= tol_rel:
            return True, hi, best_K
        mid = 0.5 * (lo + hi)
        ok_mid, _, K_mid = _dgare(mid)
        if ok_mid:
            hi = mid
            best_K = K_mid
        else:
            lo = mid

    return True, hi, best_K


# ── Public API ──────────────────────────────────────────────────────────────────

def compute_K_hinf(A, B, D, Q, R, P_lqr=None, K_lqr=None,
                   on_infeasible="fallback_lqr", **search_kwargs):
    """Return (K_hinf, gamma_min) for the tightest feasible gamma.

    on_infeasible:
        "fallback_lqr" -> return (K_lqr, NaN) when the gamma search fails
                          (Lipschitz-experiment behaviour).
        "raise"        -> raise ValueError when the gamma search fails
                          (imitation-experiment behaviour; task is dropped).
    """
    if P_lqr is None or K_lqr is None:
        P_lqr, K_lqr = solve_lqr(A, B, Q, R)

    found, gamma_min, K_hinf = find_min_gamma_for_Q(
        A, B, D, Q, R, P_init=P_lqr, **search_kwargs
    )
    if not found:
        if on_infeasible == "raise":
            raise ValueError("H-inf: no feasible gamma in [floor, cap]")
        return K_lqr, np.nan
    return K_hinf, gamma_min


def find_hinf_gain(A, B, D, Q, R, **search_kwargs):
    """Return K_hinf for the tightest feasible gamma; raise on failure.

    No LQR fallback: tasks for which no feasible gamma is found are rejected so
    the data-generation step can drop them. Used by the imitation experiment.
    """
    K, _ = compute_K_hinf(A, B, D, Q, R, on_infeasible="raise", **search_kwargs)
    return K


# Backward-compatible alias (same signature as the original).
lqr_dare = solve_lqr
