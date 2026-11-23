#!/usr/bin/env python3
"""Run the general-systems Lipschitz experiment. Resumable across restarts and hosts.

Each finished system is one atomically written JSON file. Workers on any machine that
shares ``--out-dir`` claim systems through lock files, so the full run is split simply
by launching this command on each machine.

Examples:
    python run_general_systems.py --out-dir OUT --workers 32
    python run_general_systems.py --out-dir OUT --workers 2 --dims 3 --n-seeds 1 \
        --norm-A 0.3 --norm-B 0.6 --norm-Rinv 1 --tasks-per-batch 20
"""

import argparse
import json
import multiprocessing as mp
import os
import socket
import sys
import time
import traceback

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def _float_list(value):
    return [float(x) for x in value.split(",")]


def _int_list(value):
    return [int(x) for x in value.split(",")]


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def claim(lock_path):
    """Create the lock atomically; reclaim it if its owner on this host has died."""
    host = socket.gethostname()
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                with open(lock_path) as f:
                    owner_host, owner_pid = f.read().split()
            except (OSError, ValueError):
                return False
            if owner_host == host and os.name == "posix" and not _pid_alive(int(owner_pid)):
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass
                continue
            return False
        with os.fdopen(fd, "w") as f:
            f.write(f"{host} {os.getpid()}\n")
        return True
    return False


def write_json_atomic(path, payload):
    tmp = f"{path}.{socket.gethostname()}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def worker(worker_id, jobs, cfg, out_dir):
    from general_experiment import HardCheckError, config_record, run_system

    systems_dir = os.path.join(out_dir, "systems")
    locks_dir = os.path.join(out_dir, "locks")
    stop_path = os.path.join(out_dir, "STOP")
    tag = f"{socket.gethostname()}:{worker_id}"
    for job in jobs:
        if os.path.exists(stop_path):
            print(f"[{tag}] STOP file present, exiting", flush=True)
            return
        result_path = os.path.join(systems_dir, f"{job['system_id']}.json")
        lock_path = os.path.join(locks_dir, f"{job['system_id']}.lock")
        if os.path.exists(result_path) or not claim(lock_path):
            continue
        try:
            if os.path.exists(result_path):
                continue
            record = run_system(job, cfg)
            write_json_atomic(result_path, {"record": record, "config": config_record(cfg)})
            print(f"[{tag}] done {job['system_id']} ratio={record['ratio']:.4g} "
                  f"infeasible={record['n_infeasible']} {record['seconds_total']:.0f}s", flush=True)
        except HardCheckError as exc:
            write_json_atomic(stop_path, {"system_id": job["system_id"], "error": str(exc)})
            print(f"[{tag}] HARD CHECK FAILED: {exc}", flush=True)
            return
        except Exception:
            error_path = os.path.join(out_dir, "errors", f"{job['system_id']}.txt")
            with open(error_path, "w") as f:
                f.write(traceback.format_exc())
            print(f"[{tag}] ERROR {job['system_id']} (see {error_path})", flush=True)
        finally:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--dims", type=_int_list, default=None)
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--norm-A", type=_float_list, default=None)
    p.add_argument("--norm-B", type=_float_list, default=None)
    p.add_argument("--norm-Rinv", type=_float_list, default=None)
    p.add_argument("--tasks-per-batch", type=int, default=0,
                   help="Tasks in each base batch (0 means the paper's 100 m^2)")
    args = p.parse_args()

    cfg = C.GeneralSystemsConfig(tasks_per_batch=args.tasks_per_batch)
    for attr, value in (("dims", args.dims), ("n_seeds", args.n_seeds),
                        ("norm_A_list", args.norm_A), ("norm_B_list", args.norm_B),
                        ("norm_Rinv_list", args.norm_Rinv)):
        if value is not None:
            setattr(cfg, attr, value)

    from general_experiment import build_jobs

    for sub in ("systems", "locks", "errors"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
    stop_path = os.path.join(args.out_dir, "STOP")
    if os.path.exists(stop_path):
        sys.exit(f"Refusing to start: {stop_path} exists (a hard check failed).")

    jobs = build_jobs(cfg)
    done = sum(os.path.exists(os.path.join(args.out_dir, "systems", f"{j['system_id']}.json"))
               for j in jobs)
    print(f"{len(jobs)} systems in grid, {done} already complete, {args.workers} workers", flush=True)
    t0 = time.time()
    procs = [mp.Process(target=worker, args=(i, jobs, cfg, args.out_dir))
             for i in range(args.workers)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

    done = sum(os.path.exists(os.path.join(args.out_dir, "systems", f"{j['system_id']}.json"))
               for j in jobs)
    print(f"{done}/{len(jobs)} systems complete after {time.time() - t0:.0f}s", flush=True)
    if os.path.exists(stop_path):
        with open(stop_path) as f:
            print("HARD CHECK FAILED:", f.read(), flush=True)
        sys.exit(2)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
