"""Configuration for the Lipschitz-ratio experiment.

This replaces the original ``common.config`` dynamic-module-alias mechanism (where
``select_config`` swapped ``sys.modules["common.config"]`` at runtime) with plain
module-level constants and two explicit ``ExperimentConfig`` objects. Callers pass a
config object to ``run_experiment(n, cfg)`` and the plotting functions.
"""

import os
from dataclasses import dataclass, field
from typing import List


# ── Constants shared by all Lipschitz experiments ─────────────────────────────
Q_MIN_EIG_LQR = 0.0
Q_MIN_EIG_HINF = 1e-1

N_SEEDS = 5
BASE_SEED = 42

EPS = 1e-12
FIGURES_DIR = "figures"

# Solver hyperparameters (looser than the imitation experiment's, matching the
# original Lipschitz configuration).
LQR_TOL = 1e-6
LQR_MAX_ITER = 500
HINF_TOL_P = 1e-6
HINF_MAX_ITER_P = 500
HINF_TOL_GAMMA = 1e-2
HINF_MAX_ITER_GAMMA = 100
HINF_GAMMA_INIT = 1e-2
HINF_GAMMA_MAX = 1e5
HINF_GAMMA_BRACKET = 10.0

# The original Lipschitz experiment did NOT reject non-stabilizing DGARE fixed
# points; keep that behaviour to reproduce the published figure exactly.
REQUIRE_CLOSED_LOOP_STABLE = False

# Default alignment constant used when plotting legacy commuting files.
ALIGNMENT_CONSTANT = 0.9

RESULTS_GENERAL_CSV_TEMPLATE = "results/experiments_general_n={}.csv"
RESULTS_COMMUTING_CSV_TEMPLATE = "results/experiments_commuting_n={}.csv"
THEORY_DIR = os.path.dirname(os.path.abspath(__file__))


def num_Q_for_dim(n: int) -> int:
    return 100 * n ** 2


def results_csv_template(experiment_type: str) -> str:
    if experiment_type == "general":
        return RESULTS_GENERAL_CSV_TEMPLATE
    if experiment_type == "commuting":
        return RESULTS_COMMUTING_CSV_TEMPLATE
    raise ValueError(f"Unknown experiment type: {experiment_type}")


def resolve_theory_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(THEORY_DIR, path)


def hinf_search_kwargs() -> dict:
    """Solver kwargs passed to core.solvers.compute_K_hinf for this experiment."""
    return dict(
        gamma_floor=HINF_GAMMA_INIT,
        gamma_cap=HINF_GAMMA_MAX,
        bracket_factor=HINF_GAMMA_BRACKET,
        tol_rel=HINF_TOL_GAMMA,
        max_bisect=HINF_MAX_ITER_GAMMA,
        tol_p=HINF_TOL_P,
        maxit_p=HINF_MAX_ITER_P,
        require_closed_loop_stable=REQUIRE_CLOSED_LOOP_STABLE,
    )


# ── Per-experiment configuration objects (no dynamic module aliasing) ─────────

@dataclass
class ExperimentConfig:
    experiment_type: str
    NORM_A_LIST: List[float]
    NORM_B_LIST: List[float]
    MIN_EIG_R_LIST: List[float]
    NORM_D_LIST: List[float]
    alignment_constant: float = ALIGNMENT_CONSTANT
    n_seeds: int = N_SEEDS
    base_seed: int = BASE_SEED
    q_min_eig_lqr: float = Q_MIN_EIG_LQR
    q_min_eig_hinf: float = Q_MIN_EIG_HINF
    verbose_outer: bool = True
    verbose_inner: bool = False


def config_general() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_type="general",
        NORM_A_LIST=[0.1, 0.4, 0.7],
        NORM_B_LIST=[0.3, 0.6, 0.9],
        MIN_EIG_R_LIST=[1, 2],
        NORM_D_LIST=[1],
    )


def config_commuting() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_type="commuting",
        NORM_A_LIST=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        NORM_B_LIST=[0.5],
        MIN_EIG_R_LIST=[1],
        NORM_D_LIST=[1],
        alignment_constant=0.9,
    )


def select_config(experiment_type: str) -> ExperimentConfig:
    """Return the ExperimentConfig for the requested type (general | commuting)."""
    if experiment_type == "general":
        return config_general()
    if experiment_type == "commuting":
        return config_commuting()
    raise ValueError(f"Unknown experiment type: {experiment_type}")
