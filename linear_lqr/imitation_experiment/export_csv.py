"""Aggregate infinite-sample training logs into paper-format CSVs.

Run the infinite-sample experiment redirecting each run's stdout to a log file:

    mkdir -p runs/value
    for n in 10 15 20; do for s in 14 48 99; do for v in lqr hinf; do
      python train_value.py --dim $n --seed $s --variant $v \
        --num-layers 3 --activation gelu --use-layernorm \
        > runs/value/n${n}_seed${s}_${v}.log
    done; done; done

then:

    python export_csv.py --runs-dir runs/value --out-dir results

Produces:
  results/experiment_comparison_summary.csv     (one row per seed x variant x dim)
  results/experiment_comparison_aggregated.csv  (mean/std over seeds; the paper format)

Parses the "Done. Train=<x>, Test=<y>" line emitted by train_value.py. Pure stdlib
plus numpy. (hinf -> "robust" naming is preserved for the aggregated file.)
"""

import argparse
import csv
import glob
import os
import re
import numpy as np


def collect(runs_dir):
    rows = []
    missing = []
    for f in sorted(glob.glob(os.path.join(runs_dir, 'n*_seed*_*.log'))):
        name = os.path.basename(f).replace('.log', '')
        m = re.match(r'n(\d+)_seed(\d+)_(\w+)', name)
        if not m:
            continue
        n, seed, variant = int(m.group(1)), int(m.group(2)), m.group(3)
        with open(f) as fh:
            content = fh.read()
        done = re.search(r'Done\. Train=(\S+), Test=(\S+)', content)
        if not done:
            missing.append(f)
            continue
        train = float(done.group(1).rstrip(','))
        test = float(done.group(2).rstrip(','))
        rows.append(dict(dimension=n, seed=seed, variant=variant,
                         train_NMSE=train, test_NMSE=test))
    return rows, missing


def write_summary(rows, out_path):
    fields = ['dimension', 'seed', 'variant', 'train_NMSE', 'test_NMSE']
    with open(out_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r['dimension'], r['variant'], r['seed'])))


def write_aggregated(rows, out_path):
    groups = {}
    for r in rows:
        groups.setdefault((r['dimension'], r['variant']), []).append(r)
    fields = ['dimension', 'variant', 'train_NMSE_mean', 'train_NMSE_std',
              'test_NMSE_mean', 'test_NMSE_std', 'n_seeds']
    with open(out_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for (dim, variant) in sorted(groups):
            g = groups[(dim, variant)]
            tr = np.array([x['train_NMSE'] for x in g])
            te = np.array([x['test_NMSE'] for x in g])
            w.writerow(dict(
                dimension=dim, variant=variant,
                train_NMSE_mean=f"{tr.mean():.6e}", train_NMSE_std=f"{tr.std():.6e}",
                test_NMSE_mean=f"{te.mean():.6e}", test_NMSE_std=f"{te.std():.6e}",
                n_seeds=len(g)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs-dir', default='runs/value')
    p.add_argument('--out-dir', default='results')
    args = p.parse_args()

    rows, missing = collect(args.runs_dir)
    if not rows:
        print(f"No parseable logs found in {args.runs_dir}. "
              f"Did you redirect train_value.py output there?")
        return
    os.makedirs(args.out_dir, exist_ok=True)
    write_summary(rows, os.path.join(args.out_dir, 'experiment_comparison_summary.csv'))
    write_aggregated(rows, os.path.join(args.out_dir, 'experiment_comparison_aggregated.csv'))
    print(f"Wrote {len(rows)} rows to {args.out_dir}/")
    if missing:
        print(f"WARNING: {len(missing)} logs had no 'Done.' line:")
        for f in missing:
            print(f"  {f}")


if __name__ == '__main__':
    main()
