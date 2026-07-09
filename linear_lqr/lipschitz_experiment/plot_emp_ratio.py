#!/usr/bin/env python3
"""Plot the Lipschitz-ratio figures from saved result CSVs.

Examples:
    python plot_emp_ratio.py --n 4 --experiment-type general     # per-config plot
    python plot_emp_ratio.py --n 4 --comparison                  # paper main figure
                                                                 # (commuting vs general)
Figures are written to ./figures/.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(
        description="Plot empirical H-infinity / LQR Lipschitz-ratio figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--n', type=int, required=True, help='System dimension')
    p.add_argument('--experiment-type', choices=['general', 'commuting'], default='general')
    p.add_argument('--comparison', action='store_true',
                   help='Side-by-side commuting-vs-general comparison (the paper main figure). '
                        'Requires both result CSVs to exist.')
    p.add_argument('--ylabel-frac', action='store_true',
                   help='Use the fraction-style y-axis label')
    args = p.parse_args()

    from plotting import plot_emp_ratio, plot_emp_ratio_comparison

    if args.comparison:
        paths = plot_emp_ratio_comparison(args.n, ylabel_frac=args.ylabel_frac)
    else:
        paths = plot_emp_ratio(args.n, experiment_type=args.experiment_type,
                               ylabel_frac=args.ylabel_frac)
    for path in paths:
        print(f"Saved: {path}")


if __name__ == '__main__':
    main()
