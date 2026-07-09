"""Shared numerics for the linear-quadratic experiments.

Both sub-experiments (Lipschitz-ratio estimation and MLP imitation) compute the
same maps K_lqr(Q) and K_hinf(Q); this package holds the single implementation.
"""

from .solvers import (
    solve_lqr,
    solve_dgare,
    find_min_gamma_for_Q,
    find_hinf_gain,
    compute_K_lqr,
    compute_K_hinf,
    lqr_dare,
    sym,
    is_posdef,
)
from .systems import (
    make_random_matrix,
    make_random_psd,
    generate_system_general,
    generate_system_commuting,
    sample_Qs_general,
    sample_psd_unitfro_min_eig_concentrated,
    scale_to_spectral_norm,
)
from .lipschitz import lipschitz_max_vectorized

__all__ = [
    "solve_lqr", "solve_dgare", "find_min_gamma_for_Q", "find_hinf_gain",
    "compute_K_lqr", "compute_K_hinf", "lqr_dare", "sym", "is_posdef",
    "make_random_matrix", "make_random_psd", "generate_system_general",
    "generate_system_commuting", "sample_Qs_general",
    "sample_psd_unitfro_min_eig_concentrated", "scale_to_spectral_norm",
    "lipschitz_max_vectorized",
]
