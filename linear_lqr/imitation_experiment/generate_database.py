"""Generate the hierarchical database of LQR / H-infinity gains.

For each (seed, dimension) pair, draw a fixed system (A, B rescaled to spectral
norm 0.5; R = M Mᵀ rescaled to a Frobenius norm in [0.25, 1]; D = I), then draw
``systems_per_n2 * n^2`` PSD task matrices Q and solve for K_lqr(Q) and K_hinf(Q).
Tasks with no feasible H-infinity controller are dropped. Deterministic per seed.

Writes data/seed_<S>/n<N>/{system.npz, tasks.npz, meta.json}.

Uses the shared solver/system core (linear_lqr/core).
"""

import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.solvers import solve_lqr, find_hinf_gain
from core.systems import make_random_matrix, make_random_psd


def generate(seed, n, data_root, systems_per_n2=800, A_norm=0.5, B_norm=0.5, sigma=5.0):
    """Generate and save gains for one (seed, dimension) pair."""
    data_dir = os.path.join(data_root, f'seed_{seed}', f'n{n}')
    meta_path = os.path.join(data_dir, 'meta.json')

    if os.path.exists(meta_path):
        print(f"[seed={seed} n={n}] Already exists at {data_dir}, skipping.", flush=True)
        return

    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    num_tasks = systems_per_n2 * n * n

    # Fixed system matrices
    A = make_random_matrix(n, n, A_norm, rng, sigma)
    B = make_random_matrix(n, n, B_norm, rng, sigma)
    R = make_random_psd(n, rng)
    R += 0.01 * np.eye(n)
    D = np.eye(n)

    np.savez(os.path.join(data_dir, 'system.npz'), A=A, B=B, R=R, D=D)

    Qs, K_lqrs, K_hinfs = [], [], []
    skipped = 0
    log_interval = max(1, num_tasks // 20)

    print(f"[seed={seed} n={n}] Generating {num_tasks} tasks...", flush=True)
    t0 = time.time()

    for i in range(num_tasks):
        Q = make_random_psd(n, rng)
        try:
            _, K_lqr = solve_lqr(A, B, Q, R)
            K_hinf = find_hinf_gain(A, B, D, Q, R)
        except (ValueError, np.linalg.LinAlgError):
            skipped += 1
            continue

        Qs.append(Q.flatten())
        K_lqrs.append(K_lqr)
        K_hinfs.append(K_hinf)

        done = i + 1
        if done % log_interval == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (num_tasks - done) / rate
            print(f"[seed={seed} n={n}] {done}/{num_tasks} ({100*done/num_tasks:.0f}%) | "
                  f"{rate:.0f} sys/s | ETA {eta:.0f}s | feasible {len(Qs)} | skipped {skipped}",
                  flush=True)

    elapsed = time.time() - t0
    print(f"[seed={seed} n={n}] Done: {len(Qs)} feasible / {num_tasks} total, "
          f"skipped {skipped}, took {elapsed:.1f}s", flush=True)

    np.savez(os.path.join(data_dir, 'tasks.npz'),
             Q=np.array(Qs),
             K_lqr=np.array(K_lqrs),
             K_hinf=np.array(K_hinfs))

    with open(meta_path, 'w') as f:
        json.dump({
            'seed': seed, 'n': n,
            'num_tasks': len(Qs),
            'num_attempted': num_tasks,
            'skipped': skipped,
            'elapsed_s': round(elapsed, 1),
        }, f, indent=2)

    print(f"[seed={seed} n={n}] Saved to {data_dir}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[14, 48, 99])
    parser.add_argument('--dimensions', nargs='+', type=int, default=[10, 15, 20])
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--systems-per-n2', type=int, default=800)
    args = parser.parse_args()

    for seed in args.seeds:
        for n in args.dimensions:
            generate(seed, n, args.data_root, args.systems_per_n2)
