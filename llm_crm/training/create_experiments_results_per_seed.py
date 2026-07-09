#!/usr/bin/env python3
"""Aggregate per-seed training metrics into a summary CSV.

Reads, for each seed directory under ``--results-dir``, the files
``metrics_<experiment>_vanilla.csv`` and ``metrics_<experiment>_safe.csv``, takes the
row for ``--epoch``, and writes one summary row per (seed, model_type) with train /
val / test accuracy. Pure standard library.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional


def load_epoch_metrics(csv_path: Path, epoch: int) -> Optional[Dict[str, float]]:
    """Load metrics for a specific epoch from a CSV file."""
    if not csv_path.exists():
        return None
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['epoch']) == epoch:
                return {
                    'train_accuracy': float(row.get('train_accuracy', 0.0)),
                    'val_accuracy': float(row.get('val_accuracy', 0.0)),
                    'test_accuracy': float(row.get('test_accuracy', 0.0)),
                }
    return None


def collect_results(results_dir: Path, experiment_name: str, epoch: int) -> List[Dict]:
    """Collect per-seed vanilla/safe metrics at the given epoch."""
    seeds = []
    for seed_dir in sorted(results_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        try:
            seed = int(seed_dir.name)
        except ValueError:
            continue
        vanilla_path = seed_dir / f'metrics_{experiment_name}_vanilla.csv'
        safe_path = seed_dir / f'metrics_{experiment_name}_safe.csv'
        if vanilla_path.exists() or safe_path.exists():
            seeds.append(seed)

    results = []
    for seed in sorted(seeds):
        seed_dir = results_dir / str(seed)
        for model_type in ('vanilla', 'safe'):
            path = seed_dir / f'metrics_{experiment_name}_{model_type}.csv'
            metrics = load_epoch_metrics(path, epoch)
            if metrics:
                results.append({
                    'seed': seed,
                    'model_type': model_type,
                    'train_accuracy': metrics['train_accuracy'],
                    'val_accuracy': metrics['val_accuracy'],
                    'test_accuracy': metrics['test_accuracy'],
                })
            else:
                print(f"Warning: no epoch {epoch} data for seed {seed} {model_type}")
    return results


def write_csv(results: List[Dict], output_path: Path) -> None:
    fieldnames = ['seed', 'model_type', 'train_accuracy', 'val_accuracy', 'test_accuracy']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main(results_dir=None, experiment_name='7_trajectories_train', epoch=19, out_path=None):
    base_dir = Path(__file__).parent
    if results_dir is None:
        results_dir = base_dir / 'results'
    results_dir = Path(results_dir)
    if out_path is None:
        out_path = base_dir / 'experiments_results_per_seed.csv'
    out_path = Path(out_path)

    if not results_dir.exists():
        print(f"Results dir not found: {results_dir}")
        return []

    results = collect_results(results_dir, experiment_name, epoch)
    write_csv(results, out_path)
    print(f"\nResults saved to: {out_path}  ({len(results)} rows)")
    print(f"\n{'Seed':<6} {'Model':<10} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-" * 50)
    for row in results:
        print(f"{row['seed']:<6} {row['model_type']:<10} "
              f"{row['train_accuracy']:<10.4f} {row['val_accuracy']:<10.4f} {row['test_accuracy']:<10.4f}")
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', default=None,
                   help='Directory of per-seed metrics (default: ./results)')
    p.add_argument('--experiment-name', default='7_trajectories_train')
    p.add_argument('--epoch', type=int, default=19)
    p.add_argument('--out', default=None, help='Output CSV path')
    args = p.parse_args()
    main(results_dir=args.results_dir, experiment_name=args.experiment_name,
         epoch=args.epoch, out_path=args.out)
