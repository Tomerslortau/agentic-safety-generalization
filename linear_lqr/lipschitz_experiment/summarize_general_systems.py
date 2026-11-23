#!/usr/bin/env python3
"""Checks and headline numbers of the general-systems experiment, from the result CSVs."""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

REL_TOL = 1e-12


def _stats(values):
    values = np.asarray(values, dtype=float)
    return dict(n=int(values.size), min=float(values.min()), median=float(np.median(values)),
                max=float(values.max()))


def _changed(a, b):
    return np.abs(np.asarray(a) / np.asarray(b) - 1.0) > REL_TOL


def summarize(df):
    holds = df["assumption_1_holds"].astype(bool)
    out = dict(
        n_systems=int(len(df)),
        n_tasks_per_system=sorted(int(x) for x in df["n_tasks"].unique()),
        kappa_D=dict(min=float(df["kappa_D"].min()), max=float(df["kappa_D"].max())),
        alpha=_stats(df["alpha"]),
        ratio=_stats(df["ratio"]),
        n_ratio_le_1=int((df["ratio"] <= 1.0).sum()),
        fraction_ratio_gt_1=float((df["ratio"] > 1.0).mean()),
        n_ratio_nonfinite=int((~np.isfinite(df["ratio"])).sum()),
        n_assumption_1_holds=int(holds.sum()),
        n_assumption_1_violated=int((~holds).sum()),
        violated_grid_points=sorted({(float(a), float(b), float(r)) for a, b, r in
                                     zip(df.loc[~holds, "norm_A"], df.loc[~holds, "norm_B"],
                                         df.loc[~holds, "norm_Rinv"])}),
        assumption_1_uniform_per_grid_point=bool(
            (df.groupby(["norm_A", "norm_B", "norm_Rinv"])["assumption_1_holds"].nunique() == 1).all()),
        n_coef_ge_1=int((df["coef"] >= 1.0).sum()),
        n_coef_nonpositive=int((df["coef"] <= 0.0).sum()),
        n_coef_positive_without_assumption=int(((df["coef"] > 0.0) & ~holds).sum()),
        median_ratio_over_coef=float((df.loc[holds, "ratio"] / df.loc[holds, "coef"]).median()),
        n_ratio_below_coef_under_assumption=int((df.loc[holds, "ratio"] < df.loc[holds, "coef"]).sum()),
        lemma_1_all_ok=bool(df["lemma_1_ok"].astype(bool).all()),
        max_L_unsafe_over_c_unsafe=float((df.loc[holds, "L_unsafe"] / df.loc[holds, "c_unsafe"]).max()),
        n_infeasible=int(df["n_infeasible"].sum()),
        n_infeasible_rank_one=int(df["n_infeasible_rank_one"].sum()),
        n_rejected_nonstabilizing=int(df["n_rejected_nonstabilizing"].sum()),
        n_systems_with_infeasible=int((df["n_infeasible"] > 0).sum()),
        n_redraws=int(df["n_redraws"].sum()),
        n_tasks_clipped=int(df["n_tasks_clipped"].sum()),
        n_tasks_renormalized=int(df["n_tasks_renormalized"].sum()),
        n_tasks_total=int(df["n_tasks"].sum()),
        gamma_min=float(df["gamma_min"].min()), gamma_max=float(df["gamma_max"].max()),
        max_jacobian_fd_rel_error=float(df["jacobian_fd_rel_error"].max()),
        max_jacobian_lyapunov_rel_error=float(df["jacobian_lyapunov_rel_error"].max()),
        jacobian_over_pairwise_L_unsafe=_stats(df["L_unsafe_jacobian"] / df["L_unsafe_pairwise"]),
        n_jacobian_exceeds_pairwise=int((df["L_unsafe_jacobian"] > df["L_unsafe_pairwise"]).sum()),
        ratio_over_pairwise_only_ratio=_stats(df["ratio"] / df["ratio_pairwise_only"]),
        ratio_over_no_rank_one_ratio=_stats(df["ratio"] / df["ratio_no_rank_one"]),
        n_rank_one_changes_ratio=int(_changed(df["ratio"], df["ratio_no_rank_one"]).sum()),
        n_rank_one_changes_L_safe=int(_changed(df["L_safe"], df["L_safe_no_rank_one"]).sum()),
        n_rank_one_changes_L_unsafe=int(_changed(df["L_unsafe"], df["L_unsafe_no_rank_one"]).sum()),
        n_gamma_at_floor=int(df["n_gamma_at_floor"].sum()),
        n_systems_with_floor_tasks=int((df["n_gamma_at_floor"] > 0).sum()),
        n_floor_tasks_change_ratio=int(_changed(df["ratio"], df["ratio_without_floor_tasks"]).sum()),
        max_abs_rel_change_without_floor_tasks=float(
            (df["ratio_without_floor_tasks"] / df["ratio"] - 1.0).abs().max()),
        core_hours=float(df["seconds_total"].sum() / 3600.0),
        max_seconds_per_system=float(df["seconds_total"].max()),
    )
    per_combo = (df.assign(gt1=df["ratio"] > 1.0)
                 .groupby(["norm_B", "norm_Rinv"])
                 .agg(n=("gt1", "size"), fraction_ratio_gt_1=("gt1", "mean"),
                      min_ratio=("ratio", "min"), n_coef_ge_1=("coef", lambda c: int((c >= 1).sum())))
                 .reset_index())
    out["per_B_Rinv"] = per_combo.to_dict(orient="records")
    log_ratio = df.assign(log_ratio=np.log(df["ratio"])).groupby("norm_A")["log_ratio"]
    out["per_norm_A"] = [dict(norm_A=float(a), n=int(g.size), geo_mean=float(np.exp(g.mean())),
                              log_std=float(g.std()), min=float(np.exp(g.min())),
                              max=float(np.exp(g.max())))
                         for a, g in log_ratio]
    flagged = df.loc[holds & (df["ratio"] < df["coef"]), ["system_id", "ratio", "coef"]]
    out["systems_ratio_below_coef"] = flagged.to_dict(orient="records")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=C.resolve_theory_path(C.GENERAL_SYSTEMS_RESULTS_DIR))
    args = p.parse_args()
    frames = {}
    for path in glob.glob(os.path.join(args.results_dir, "general_systems_m=*.csv")):
        frames[int(re.search(r"m=(\d+)\.csv$", path).group(1))] = pd.read_csv(path)
    if not frames:
        raise SystemExit(f"No result CSVs in {args.results_dir}")
    summary = {f"m={m}": summarize(df) for m, df in sorted(frames.items())}
    summary["all"] = summarize(pd.concat(frames.values(), ignore_index=True))
    path = os.path.join(args.results_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
