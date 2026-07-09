"""Lipschitz-ratio experiment: estimate L(K_hinf) / L(K_lqr) over a grid.

For each grid point (||A||, ||B||, min-eig(R), ||D||) and each seed, draw a system,
sample task matrices Q, compute the LQR gains K_lqr(Q) and the H-infinity gains
K_hinf(Q) (on the subset for which a feasible gamma exists), and estimate each map's
empirical Lipschitz constant. The reported statistic is the ratio.

Uses the shared solver/system core (linear_lqr/core) and an explicit config object
(no dynamic module aliasing).
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.solvers import lqr_dare, compute_K_hinf
from core.systems import generate_system_general, generate_system_commuting, sample_Qs_general
from core.lipschitz import lipschitz_max_vectorized

import config as C


def commuting_bound_value(norm_A, norm_B, min_eig_R, alignment_constant):
    """The plotted commuting lower-bound for one grid point."""
    return (alignment_constant ** 3) / (norm_A * (2.0 + 4.0 * (norm_B ** 2) / min_eig_R))


def run_experiment(n, cfg):
    """Run the Lipschitz experiment for a fixed dimension n under config ``cfg``.

    Returns a pandas DataFrame with one row per (grid point, seed).
    """
    if cfg.experiment_type not in {"general", "commuting"}:
        raise ValueError(f"Unknown experiment type: {cfg.experiment_type}")
    if cfg.experiment_type == "commuting" and not 0.0 < cfg.alignment_constant <= 1.0:
        raise ValueError("alignment_constant must be in (0, 1] for commuting experiments")

    t_overall_start = time.time()
    seeds = [cfg.base_seed + i for i in range(cfg.n_seeds)]
    search_kwargs = C.hinf_search_kwargs()
    rows = []

    for norm_D in cfg.NORM_D_LIST:
        for min_eig_R in cfg.MIN_EIG_R_LIST:
            for norm_B in cfg.NORM_B_LIST:
                for norm_A in cfg.NORM_A_LIST:
                    ratio_values = []
                    bound_value = (
                        commuting_bound_value(norm_A, norm_B, min_eig_R, cfg.alignment_constant)
                        if cfg.experiment_type == "commuting" else np.nan
                    )
                    for t, seed in enumerate(seeds):
                        t_start = time.time()
                        rng = np.random.default_rng(seed)
                        if cfg.experiment_type == "commuting":
                            A, B, D, R = generate_system_commuting(
                                n, norm_A, norm_B, norm_D, min_eig_R,
                                cfg.alignment_constant, rng)
                        else:
                            A, B, D, R = generate_system_general(
                                n, norm_A, norm_B, norm_D, min_eig_R, rng)

                        # Sample Q matrices (three batches, as in the paper).
                        n_Q = C.num_Q_for_dim(n)
                        Q_list = sample_Qs_general(
                            n, n_Q, rng, q0=cfg.q_min_eig_hinf, norm=1.0, require_exact_norm=True)
                        Q_list.extend(sample_Qs_general(
                            n, n_Q, rng, q0=cfg.q_min_eig_lqr, norm=1.0))
                        Q_list.extend(sample_Qs_general(
                            n, n_Q, rng, q0=cfg.q_min_eig_lqr, norm=norm_A))

                        # LQR: same Qs; H-inf: only the feasible subset.
                        K_lqr_list, P_lqr_list = [], []
                        for Q in Q_list:
                            P_lqr, K_lqr = lqr_dare(A, B, Q, R)
                            P_lqr_list.append(P_lqr)
                            K_lqr_list.append(K_lqr)

                        K_hinf_list, Q_hinf_feasible, gamma_list = [], [], []
                        for Q, P_lqr, K_lqr in zip(Q_list, P_lqr_list, K_lqr_list):
                            K_hinf, gamma = compute_K_hinf(
                                A, B, D, Q, R, P_lqr=P_lqr, K_lqr=K_lqr,
                                on_infeasible="fallback_lqr", **search_kwargs)
                            if not np.isnan(gamma):
                                K_hinf_list.append(K_hinf)
                                Q_hinf_feasible.append(Q)
                                gamma_list.append(gamma)

                        L_lqr = lipschitz_max_vectorized(Q_list, K_lqr_list) if len(K_lqr_list) >= 2 else np.nan
                        L_hinf = lipschitz_max_vectorized(Q_hinf_feasible, K_hinf_list) if len(K_hinf_list) >= 2 else np.nan
                        if np.isfinite(L_lqr) and np.isfinite(L_hinf) and abs(L_lqr) > 0:
                            ratio_values.append(L_hinf / L_lqr)

                        dt = time.time() - t_start
                        rows.append({
                            "n": n,
                            "experiment_type": cfg.experiment_type,
                            "norm_A": norm_A, "norm_B": norm_B, "norm_D": norm_D,
                            "min_eig_R": min_eig_R,
                            "alignment_constant": (cfg.alignment_constant
                                                   if cfg.experiment_type == "commuting" else np.nan),
                            "bound": bound_value,
                            "seed": seed,
                            "n_Q": len(Q_list),
                            "n_Q_lqr": len(Q_list),
                            "n_Q_hinf_feasible": len(Q_hinf_feasible),
                            "L_lqr_emp": L_lqr,
                            "L_hinf_emp": L_hinf,
                            "gamma_attenuation_mean": float(np.mean(gamma_list)) if gamma_list else np.nan,
                        })

                        if cfg.verbose_outer:
                            print(f"[|A|={norm_A:.3g} |B|={norm_B:.3g} min_eig(R)={min_eig_R:.3g} "
                                  f"|D|={norm_D:.3g}] seed {t+1}/{len(seeds)} | "
                                  f"LQR={L_lqr:.3g} Hinf={L_hinf:.3g} "
                                  f"({len(Q_hinf_feasible)}/{len(Q_list)} feasible) | {dt:.2f}s", flush=True)

                    if cfg.verbose_outer:
                        ratio_str = f"{np.mean(ratio_values):.3g}" if ratio_values else "nan"
                        print(f"[|A|={norm_A:.3g} |B|={norm_B:.3g} min_eig(R)={min_eig_R:.3g} "
                              f"|D|={norm_D:.3g}] mean Hinf/LQR ratio: {ratio_str}", flush=True)

    if cfg.verbose_outer:
        print(f"\n[TIMING] completed in {time.time() - t_overall_start:.2f}s", flush=True)
    return pd.DataFrame(rows)


def save_results(df, n, cfg):
    """Append the DataFrame to the configured CSV; return its path."""
    filepath = C.resolve_theory_path(C.results_csv_template(cfg.experiment_type).format(n))
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    file_exists = os.path.exists(filepath)
    df.to_csv(filepath, mode="a", header=not file_exists, index=False)
    return filepath
