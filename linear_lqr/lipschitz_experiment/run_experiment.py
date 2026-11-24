#!/usr/bin/env python3
"""Run the Lipschitz experiment. Finished systems are skipped on restart.

    python run_experiment.py --out-dir <dir> --workers <n>
"""

import argparse
import json
import multiprocessing as mp
import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import config as C
from experiment import build_jobs, run_system


def result_path(out_dir, job):
    return os.path.join(out_dir, f"{job['system_id']}.json")


def run_and_save(args):
    job, cfg, out_dir = args
    record = run_system(job, cfg)
    path = result_path(out_dir, job)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1)
    os.replace(tmp, path)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--dims", type=int, nargs="+", default=None)
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--tasks-per-batch", type=int, default=0, help="0 means the paper's 100 m^2")
    args = p.parse_args()

    cfg = C.Config(tasks_per_batch=args.tasks_per_batch)
    if args.dims:
        cfg.dims = args.dims
    if args.n_seeds:
        cfg.n_seeds = args.n_seeds
    os.makedirs(args.out_dir, exist_ok=True)

    jobs = build_jobs(cfg)
    pending = [job for job in jobs if not os.path.exists(result_path(args.out_dir, job))]
    print(f"{len(jobs)} systems in grid, {len(jobs) - len(pending)} already complete", flush=True)
    with mp.Pool(args.workers) as pool:
        tasks = [(job, cfg, args.out_dir) for job in pending]
        for done, record in enumerate(pool.imap_unordered(run_and_save, tasks), 1):
            print(f"[{done}/{len(pending)}] {record['system_id']} ratio={record['ratio']:.4g} "
                  f"infeasible={record['n_infeasible']} {record['seconds']:.0f}s", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
