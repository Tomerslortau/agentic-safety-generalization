#!/usr/bin/env python3
"""Merge per-system JSON results into one CSV per dimension (one row per system)."""

import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def collect(out_dir, dest_dir):
    paths = sorted(glob.glob(os.path.join(out_dir, "systems", "*.json")))
    if not paths:
        raise SystemExit(f"No result files in {out_dir}")
    records = []
    for path in paths:
        with open(path) as f:
            records.append(json.load(f)["record"])
    df = pd.DataFrame(records).sort_values(
        ["m", "norm_Rinv", "norm_B", "norm_A", "seed_index"]).reset_index(drop=True)
    os.makedirs(dest_dir, exist_ok=True)
    written = []
    for m, group in df.groupby("m"):
        path = os.path.join(dest_dir, f"general_systems_m={m}.csv")
        tmp = f"{path}.tmp"
        group.to_csv(tmp, index=False)
        os.replace(tmp, path)
        written.append((path, len(group)))
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True, help="Directory given to run_general_systems.py")
    p.add_argument("--dest-dir", default=C.resolve_theory_path(C.GENERAL_SYSTEMS_RESULTS_DIR))
    args = p.parse_args()
    for path, n_rows in collect(args.out_dir, args.dest_dir):
        print(f"{path}: {n_rows} systems")


if __name__ == "__main__":
    main()
