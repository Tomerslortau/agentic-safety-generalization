"""Configuration of the Lipschitz experiment (Section 4.1)."""

from dataclasses import dataclass, field
from typing import List

# Gamma search for K_safe (core.solvers). The gamma tolerance is absolute for gamma < 1.
HINF_SEARCH = dict(
    gamma_floor=1e-2,
    gamma_cap=1e5,
    bracket_factor=10.0,
    tol_rel=1e-4,
    max_bisect=100,
    tol_p=1e-6,
    maxit_p=500,
    require_closed_loop_stable=True,
)


@dataclass
class Config:
    dims: List[int] = field(default_factory=lambda: [4, 8])
    norm_A: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7])
    norm_B: List[float] = field(default_factory=lambda: [0.3, 0.6, 0.9])
    norm_Rinv: List[float] = field(default_factory=lambda: [1.0, 2.0])
    norm_D: float = 1.0
    n_seeds: int = 5
    base_seed: int = 42
    rank_one_eps: List[float] = field(default_factory=lambda: [1e-3, 1e-2])
    q_min_eig: float = 0.1  # minimum eigenvalue of the first task batch
    tasks_per_batch: int = 0  # 0 means 100 m^2
    block_size: int = 2000  # pairwise distances are computed in blocks of this size

    def n_per_batch(self, m: int) -> int:
        return self.tasks_per_batch or 100 * m ** 2
