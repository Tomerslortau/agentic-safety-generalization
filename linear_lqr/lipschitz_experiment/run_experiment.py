#!/usr/bin/env python3
"""Run the Lipschitz-ratio experiment (produces the paper's main figure data).

Examples:
    python run_experiment.py --n 4 --experiment-type general
    python run_experiment.py --n 4 --experiment-type commuting --alignment-constant 0.9
    python run_experiment.py --n 4 --experiment-type general --norm-A 0.1,0.2 --n-seeds 5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import select_config


def _float_list(value):
    return [float(x.strip()) for x in value.split(',')]


def main():
    p = argparse.ArgumentParser(
        description="Estimate the Lipschitz ratio L(K_hinf)/L(K_lqr) over a grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--n', type=int, required=True, help='System dimension')
    p.add_argument('--experiment-type', choices=['general', 'commuting'], default='general')
    p.add_argument('--alignment-constant', type=float, default=None)
    p.add_argument('--n-seeds', type=int, default=None)
    p.add_argument('--base-seed', type=int, default=None)
    p.add_argument('--norm-A', type=_float_list, default=None, help='e.g. "0.1,0.2,0.3"')
    p.add_argument('--norm-B', type=_float_list, default=None)
    p.add_argument('--norm-D', type=_float_list, default=None)
    p.add_argument('--min-eig-R', type=_float_list, default=None)
    p.add_argument('--quiet', action='store_true', help='Disable per-seed progress output')
    args = p.parse_args()

    cfg = select_config(args.experiment_type)
    if args.alignment_constant is not None:
        cfg.alignment_constant = args.alignment_constant
    if args.n_seeds is not None:
        cfg.n_seeds = args.n_seeds
    if args.base_seed is not None:
        cfg.base_seed = args.base_seed
    if args.norm_A is not None:
        cfg.NORM_A_LIST = args.norm_A
    if args.norm_B is not None:
        cfg.NORM_B_LIST = args.norm_B
    if args.norm_D is not None:
        cfg.NORM_D_LIST = args.norm_D
    if args.min_eig_R is not None:
        cfg.MIN_EIG_R_LIST = args.min_eig_R
    if args.quiet:
        cfg.verbose_outer = False

    # Import after config so any module-level constants are already set.
    from experiments import run_experiment, save_results

    df = run_experiment(args.n, cfg)
    path = save_results(df, args.n, cfg)
    print(f"\nResults saved to: {path}")


if __name__ == '__main__':
    main()
