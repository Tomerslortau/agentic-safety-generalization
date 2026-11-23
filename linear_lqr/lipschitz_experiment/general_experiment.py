"""One system of the general-systems Lipschitz experiment, and its job grid."""

import os
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.solvers import compute_K_hinf, find_min_gamma_for_Q, solve_lqr

import config as C
from general_systems import (
    MAX_REDRAWS,
    StructuralAssumptionError,
    assert_system,
    draw_system,
    rank_one_tasks,
    sample_base_tasks,
    sanitize_tasks,
    system_seed_sequence,
    theory_constants,
)
from lipschitz_estimators import jacobian_max, pairwise_max_with_extra, validate_jacobian

LEMMA_TOL = 1e-9
FD_TOL = 1e-3


class HardCheckError(RuntimeError):
    """A check that can only fail through a bug; the whole run must stop."""


def build_jobs(cfg):
    jobs = []
    for m in cfg.dims:
        for i_R, norm_Rinv in enumerate(cfg.norm_Rinv_list):
            for i_B, norm_B in enumerate(cfg.norm_B_list):
                for i_A, norm_A in enumerate(cfg.norm_A_list):
                    for seed_index in range(cfg.n_seeds):
                        jobs.append(dict(
                            system_id=f"m={m}_A={norm_A:g}_B={norm_B:g}_Rinv={norm_Rinv:g}"
                                      f"_D={cfg.norm_D:g}_seed={seed_index}",
                            m=m, norm_A=norm_A, norm_B=norm_B, norm_Rinv=norm_Rinv,
                            norm_D=cfg.norm_D, seed_index=seed_index,
                            grid_index=(i_A, i_B, i_R),
                        ))
    return jobs


def git_commit():
    """Commit of the deployed code (GIT_COMMIT file) or of the local checkout."""
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        marker = os.path.join(path, "GIT_COMMIT")
        if os.path.exists(marker):
            with open(marker) as f:
                return f.read().strip()
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def draw_valid_system(job, cfg):
    """Draw system and tasks, redrawing with the next seed offset on a failed assertion."""
    failures = []
    for attempt in range(MAX_REDRAWS):
        seq = system_seed_sequence(cfg.base_seed, job["m"], job["grid_index"],
                                   job["seed_index"], attempt)
        system_seq, task_seq = seq.spawn(2)
        try:
            system = draw_system(job["m"], job["norm_A"], job["norm_B"], job["norm_Rinv"],
                                 job["norm_D"], np.random.default_rng(system_seq))
            assert_system(system, job["norm_A"], job["norm_B"], job["norm_Rinv"], job["norm_D"])
            task_rng = np.random.default_rng(task_seq)
            base = sample_base_tasks(job["m"], cfg.n_per_batch(job["m"]), job["norm_A"], task_rng,
                                     cfg.q_min_eig_hinf, cfg.q_min_eig_lqr)
            extra = rank_one_tasks(system, job["m"], task_rng, cfg.rank_one_epsilons)
            Qs, n_clipped, n_renorm = sanitize_tasks(base + extra)
        except StructuralAssumptionError as exc:
            failures.append(f"attempt {attempt}: {exc}")
            continue
        return system, Qs, len(base), attempt, n_clipped, n_renorm, failures, task_rng
    raise RuntimeError(f"{job['system_id']}: no valid draw in {MAX_REDRAWS} attempts: {failures[-3:]}")


def solve_safe(system, Qs, Ps, Ks, search_kwargs):
    """K_safe on every task; returns gains (NaN where infeasible) and gammas."""
    K_safe = np.full_like(Ks, np.nan)
    gammas = np.full(len(Qs), np.nan)
    for i, (Q, P, K) in enumerate(zip(Qs, Ps, Ks)):
        K_i, gamma = compute_K_hinf(system.A, system.B, system.D, Q, system.R, P_lqr=P, K_lqr=K,
                                    on_infeasible="fallback_lqr", **search_kwargs)
        if not np.isnan(gamma):
            K_safe[i] = K_i
            gammas[i] = gamma
    return K_safe, gammas


def _ratio(numerator, denominator):
    if np.isfinite(numerator) and np.isfinite(denominator) and denominator > 0:
        return float(numerator / denominator)
    return float("nan")


def run_system(job, cfg, search_kwargs=None):
    """Run one system and return its flat result record."""
    t_start = time.time()
    search_kwargs = C.general_systems_search_kwargs() if search_kwargs is None else search_kwargs
    system, Qs, n_base, attempt, n_clipped, n_renorm, failures, rng = draw_valid_system(job, cfg)
    constants = theory_constants(system)

    t0 = time.time()
    solved = [solve_lqr(system.A, system.B, Q, system.R) for Q in Qs]
    Ps = np.array([s[0] for s in solved])
    K_unsafe = np.array([s[1] for s in solved])
    t_lqr = time.time() - t0

    t0 = time.time()
    K_safe, gammas = solve_safe(system, Qs, Ps, K_unsafe, search_kwargs)
    t_hinf = time.time() - t0
    feasible = np.isfinite(gammas)

    infeasible = np.flatnonzero(~feasible)
    n_rejected_nonstabilizing = 0
    if search_kwargs.get("require_closed_loop_stable", False):
        relaxed = dict(search_kwargs, require_closed_loop_stable=False)
        for i in infeasible:
            found, _, _ = find_min_gamma_for_Q(system.A, system.B, system.D, Qs[i], system.R,
                                               P_init=Ps[i], **relaxed)
            n_rejected_nonstabilizing += int(found)

    t0 = time.time()
    L_unsafe_pair_base, L_unsafe_pair_all = pairwise_max_with_extra(
        Qs, K_unsafe, n_base, block_size=cfg.block_size)
    n_base_feasible = int(feasible[:n_base].sum())
    L_safe_base, L_safe_all = pairwise_max_with_extra(
        Qs[feasible], K_safe[feasible], n_base_feasible, block_size=cfg.block_size)
    at_floor = feasible & (gammas <= search_kwargs["gamma_floor"])
    L_safe_no_floor = L_safe_all
    if at_floor.any():
        keep = feasible & ~at_floor
        _, L_safe_no_floor = pairwise_max_with_extra(
            Qs[keep], K_safe[keep], int(keep[:n_base].sum()), block_size=cfg.block_size)
    t_pairwise = time.time() - t0

    t0 = time.time()
    L_unsafe_jac_base, L_unsafe_jac_all = jacobian_max(system.A, system.B, system.R, Qs, Ps,
                                                       K_unsafe, n_base)
    fd_error, lyap_error = validate_jacobian(system.A, system.B, system.R,
                                             Qs[:cfg.n_per_batch(job["m"])], rng)
    t_jacobian = time.time() - t0

    L_unsafe_base = max(L_unsafe_pair_base, L_unsafe_jac_base)
    L_unsafe_all = max(L_unsafe_pair_all, L_unsafe_jac_all)

    if fd_error > FD_TOL or lyap_error > FD_TOL:
        raise HardCheckError(f"{job['system_id']}: Jacobian validation failed "
                             f"(fd={fd_error:.2e}, lyapunov={lyap_error:.2e})")
    lemma_ok = True
    if constants["assumption_1_holds"]:
        lemma_ok = bool(L_unsafe_all <= constants["c_unsafe"] * (1.0 + LEMMA_TOL))
        if not lemma_ok:
            raise HardCheckError(f"{job['system_id']}: Lemma 1 violated: L_unsafe={L_unsafe_all:.6g} "
                                 f"> c_unsafe={constants['c_unsafe']:.6g}")

    finite_gammas = gammas[feasible]
    floor = search_kwargs["gamma_floor"]
    record = {k: v for k, v in job.items() if k != "grid_index"}
    record.update(constants)
    record.update(
        git_commit=git_commit(),
        base_seed=cfg.base_seed,
        seed_attempt=attempt,
        n_redraws=len(failures),
        redraw_reasons=" | ".join(failures),
        n_tasks=len(Qs), n_tasks_base=n_base, n_tasks_rank_one=len(Qs) - n_base,
        n_tasks_clipped=n_clipped, n_tasks_renormalized=n_renorm,
        n_safe_feasible=int(feasible.sum()), n_safe_feasible_base=n_base_feasible,
        n_infeasible=int((~feasible).sum()),
        n_infeasible_base=int((~feasible[:n_base]).sum()),
        n_infeasible_rank_one=int((~feasible[n_base:]).sum()),
        n_rejected_nonstabilizing=n_rejected_nonstabilizing,
        n_gamma_at_floor=int((finite_gammas <= floor).sum()),
        gamma_min=float(finite_gammas.min()) if finite_gammas.size else float("nan"),
        gamma_median=float(np.median(finite_gammas)) if finite_gammas.size else float("nan"),
        gamma_mean=float(finite_gammas.mean()) if finite_gammas.size else float("nan"),
        gamma_max=float(finite_gammas.max()) if finite_gammas.size else float("nan"),
        L_safe=L_safe_all, L_safe_no_rank_one=L_safe_base,
        L_safe_without_floor_tasks=L_safe_no_floor,
        ratio_without_floor_tasks=_ratio(L_safe_no_floor, L_unsafe_all),
        L_unsafe=L_unsafe_all, L_unsafe_no_rank_one=L_unsafe_base,
        L_unsafe_pairwise=L_unsafe_pair_all, L_unsafe_pairwise_no_rank_one=L_unsafe_pair_base,
        L_unsafe_jacobian=L_unsafe_jac_all, L_unsafe_jacobian_no_rank_one=L_unsafe_jac_base,
        ratio=_ratio(L_safe_all, L_unsafe_all),
        ratio_pairwise_only=_ratio(L_safe_all, L_unsafe_pair_all),
        ratio_no_rank_one=_ratio(L_safe_base, L_unsafe_base),
        ratio_no_rank_one_pairwise_only=_ratio(L_safe_base, L_unsafe_pair_base),
        lemma_1_ok=lemma_ok,
        jacobian_fd_rel_error=fd_error, jacobian_lyapunov_rel_error=lyap_error,
        tol_gamma=search_kwargs["tol_rel"],
        require_closed_loop_stable=bool(search_kwargs["require_closed_loop_stable"]),
        seconds_lqr=t_lqr, seconds_hinf=t_hinf, seconds_pairwise=t_pairwise,
        seconds_jacobian=t_jacobian, seconds_total=time.time() - t_start,
    )
    return record


def config_record(cfg):
    return asdict(cfg)
