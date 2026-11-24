#!/usr/bin/env python3
"""Merge per-system results, compute the checks and numbers, and produce the figures.

    python analyze.py --out-dir <dir>

Writes results/lipschitz_m=<m>.csv, results/summary.json and, in PDF and PNG,
figures/general_systems_ratio (Figure 1) and figures/general_systems_breakdown_m=<m>.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
MARKERS = ["o", "s", "^"]
WIDTH = 5.2
HEIGHT = 1.9
WSPACE = 0.2
X_TICK_STEP = 0.1
Y_LABEL = r"$\mathrm{Lip}(K_{\mathrm{safe}})\; / \;\mathrm{Lip}(K_{\mathrm{unsafe}})$"


# ── Results and checks ────────────────────────────────────────────────────────

def load_results(out_dir, results_dir):
    """Read the per-system JSON files and write one CSV per dimension."""
    records = []
    for path in glob.glob(os.path.join(out_dir, "*.json")):
        with open(path) as f:
            records.append(json.load(f))
    if not records:
        raise SystemExit(f"No result files in {out_dir}")
    df = pd.DataFrame(records).sort_values(["m", "norm_Rinv", "norm_B", "norm_A", "seed_index"])
    os.makedirs(results_dir, exist_ok=True)
    frames = {int(m): group.reset_index(drop=True) for m, group in df.groupby("m")}
    for m, group in frames.items():
        group.to_csv(os.path.join(results_dir, f"lipschitz_m={m}.csv"), index=False)
    return frames


def _range(values):
    return dict(min=float(values.min()), median=float(values.median()), max=float(values.max()))


def _n_changed(a, b):
    return int(((a / b - 1.0).abs() > 1e-12).sum())


def summarize(df):
    holds = df["assumption_1_holds"].astype(bool)
    violated = df.loc[~holds, ["norm_A", "norm_B", "norm_Rinv"]].drop_duplicates()
    log_ratio = np.log(df["ratio"]).groupby(df["norm_A"])
    return dict(
        n_systems=len(df),
        ratio=_range(df["ratio"]),
        n_ratio_gt_1=int((df["ratio"] > 1.0).sum()),
        n_assumption_1_holds=int(holds.sum()),
        n_assumption_1_violated=int((~holds).sum()),
        violated_grid_points=sorted(map(tuple, violated.to_numpy().tolist())),
        n_coef_ge_1=int((df["coef"] >= 1.0).sum()),
        median_ratio_over_coef=float((df.loc[holds, "ratio"] / df.loc[holds, "coef"]).median()),
        n_ratio_below_coef_under_assumption=int((df.loc[holds, "ratio"] < df.loc[holds, "coef"]).sum()),
        max_L_unsafe_over_c_unsafe=float((df.loc[holds, "L_unsafe"] / df.loc[holds, "c_unsafe"]).max()),
        n_tasks=int(df["n_tasks"].sum()),
        n_infeasible=int(df["n_infeasible"].sum()),
        n_redraws=int(df["n_redraws"].sum()),
        n_tasks_clipped=int(df["n_tasks_clipped"].sum()),
        n_tasks_renormalized=int(df["n_tasks_renormalized"].sum()),
        n_gamma_at_floor=int(df["n_gamma_at_floor"].sum()),
        n_floor_tasks_change_ratio=_n_changed(df["ratio"], df["ratio_without_floor_tasks"]),
        max_jacobian_fd_rel_error=float(df["jacobian_fd_rel_error"].max()),
        jacobian_over_pairwise_L_unsafe=_range(df["L_unsafe_jacobian"] / df["L_unsafe_pairwise"]),
        ratio_over_pairwise_only_ratio=_range(df["ratio"] / df["ratio_pairwise_only"]),
        ratio_over_no_rank_one_ratio=_range(df["ratio"] / df["ratio_no_rank_one"]),
        n_rank_one_changes_L_safe=_n_changed(df["L_safe"], df["L_safe_no_rank_one"]),
        n_rank_one_changes_L_unsafe=_n_changed(df["L_unsafe"], df["L_unsafe_no_rank_one"]),
        per_norm_A=[dict(norm_A=float(a), geo_mean=float(np.exp(g.mean())), log_std=float(g.std()))
                    for a, g in log_ratio],
        per_B_Rinv=[dict(norm_B=float(b), norm_Rinv=float(r), n=len(g),
                         n_ratio_gt_1=int((g["ratio"] > 1.0).sum()), min_ratio=float(g["ratio"].min()))
                    for (b, r), g in df.groupby(["norm_B", "norm_Rinv"])],
    )


# ── Figures ───────────────────────────────────────────────────────────────────

def log_stats(df, column):
    """Geometric mean and +/- one std of the log, per ||A||_2."""
    df = df[np.isfinite(df[column]) & (df[column] > 0.0)]
    logs = np.log(df[column]).groupby(df["norm_A"])
    mean, std = logs.mean(), logs.std().fillna(0.0)
    return pd.DataFrame({"norm_A": mean.index.to_numpy(), "mean": np.exp(mean.to_numpy()),
                         "lower": np.exp((mean - std).to_numpy()),
                         "upper": np.exp((mean + std).to_numpy())})


def _norm_label(symbol):
    return rf"$\|{symbol}\|_2$"


def _errorbar(ax, stats, color, marker):
    y = stats["mean"].to_numpy()
    yerr = np.vstack([np.maximum(y - stats["lower"].to_numpy(), 0.0),
                      np.maximum(stats["upper"].to_numpy() - y, 0.0)])
    ax.errorbar(stats["norm_A"].to_numpy(), y, yerr=yerr, capsize=3, linestyle="-", color=color,
                linewidth=2, elinewidth=1.2, marker=marker, markersize=6, markeredgecolor=color,
                markeredgewidth=1.5, markerfacecolor="white", zorder=3)


def _finish_axis(ax, x_values, y_min, y_max):
    ax.axhline(1.0, color="red", linestyle=":", linewidth=1.5, zorder=1)
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, axis="y", which="major", alpha=0.35)
    x_low = np.round(np.floor(np.min(x_values) / X_TICK_STEP + 1e-12) * X_TICK_STEP, 10)
    x_high = np.round(np.ceil(np.max(x_values) / X_TICK_STEP - 1e-12) * X_TICK_STEP, 10)
    ax.xaxis.set_major_locator(MultipleLocator(X_TICK_STEP))
    ax.grid(True, axis="x", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=12, labelbottom=True, labelleft=True)
    ax.set_xlabel(_norm_label("A"), fontsize=14)
    ax.set_xlim(x_low - 0.3 * X_TICK_STEP, x_high + 0.3 * X_TICK_STEP)
    y_low, y_high = min(y_min, 1.0) / 1.5, 1.5 * max(y_max, 1.0)
    ax.set_ylim(y_low, y_high)
    low, high = int(np.ceil(np.log10(y_low))), int(np.floor(np.log10(y_high)))
    step = 1 if high - low <= 4 else 2  # decade ticks that always include y = 1
    ax.yaxis.set_major_locator(FixedLocator([10.0 ** k for k in range(low, high + 1) if k % step == 0]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))


def _save(fig, figures_dir, stem):
    for extension in ("png", "pdf"):
        fig.savefig(os.path.join(figures_dir, f"{stem}.{extension}"), dpi=200, bbox_inches="tight",
                    pad_inches=0.1)
    plt.close(fig)


def plot_main(frames, figures_dir):
    """Figure 1: one panel per dimension, no bound."""
    fig, axes = plt.subplots(1, len(frames), figsize=(WIDTH * len(frames) + 0.8, HEIGHT), sharey=True)
    axes = np.atleast_1d(axes)
    stats = {m: log_stats(df, "ratio") for m, df in frames.items()}
    y_min = min(float(s["lower"].min()) for s in stats.values())
    y_max = max(float(s["upper"].max()) for s in stats.values())
    for ax, (m, s) in zip(axes, stats.items()):
        _errorbar(ax, s, plt.get_cmap("tab10")(0), MARKERS[0])
        ax.set_title(rf"$m={m}$", fontsize=13)
        _finish_axis(ax, s["norm_A"].to_numpy(), y_min, y_max)
    axes[0].set_ylabel(Y_LABEL, fontsize=14)
    fig.subplots_adjust(wspace=WSPACE)
    _save(fig, figures_dir, "general_systems_ratio")


def plot_breakdown(df, m, figures_dir):
    """Appendix: panels by ||R^{-1}||_2, curves by ||B||_2, bound where Assumption 1 holds."""
    norm_Rinv_values, norm_B_values = sorted(df["norm_Rinv"].unique()), sorted(df["norm_B"].unique())
    colors = [plt.get_cmap("tab10")(i) for i in range(len(norm_B_values))]
    fig, axes = plt.subplots(1, len(norm_Rinv_values), figsize=(WIDTH * len(norm_Rinv_values) + 0.8, HEIGHT),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    y_min, y_max = np.inf, 0.0
    for ax, norm_Rinv in zip(axes, norm_Rinv_values):
        for index, norm_B in enumerate(norm_B_values):
            curve = df[(df["norm_Rinv"] == norm_Rinv) & (df["norm_B"] == norm_B)]
            stats = log_stats(curve, "ratio")
            _errorbar(ax, stats, colors[index], MARKERS[index % len(MARKERS)])
            bound = log_stats(curve[curve["assumption_1_holds"].astype(bool)], "coef")
            ax.plot(bound["norm_A"], bound["mean"], linestyle="--", color=colors[index], linewidth=1.6, zorder=2)
            ax.fill_between(bound["norm_A"], bound["lower"], bound["upper"], color=colors[index], alpha=0.15,
                            linewidth=0, zorder=1)
            y_min = min(y_min, float(stats["lower"].min()), float(bound["lower"].min()))
            y_max = max(y_max, float(stats["upper"].max()), float(bound["upper"].max()))
        ax.set_title(rf"{_norm_label('R^{-1}')}={norm_Rinv:g}", fontsize=13)
    for ax in axes:
        _finish_axis(ax, df["norm_A"].to_numpy(), y_min, y_max)
    axes[0].set_ylabel(Y_LABEL, fontsize=14)
    handles = [Line2D([0], [0], linestyle="-", color=color, linewidth=1.8, marker=MARKERS[i % len(MARKERS)],
                      markersize=5.5, markeredgecolor=color, markeredgewidth=1.4, markerfacecolor="white")
               for i, color in enumerate(colors)]
    handles.append(Line2D([0], [0], linestyle="--", color="black", linewidth=1.6))
    labels = [rf"{_norm_label('B')}={b:g}" for b in norm_B_values] + ["Bound"]
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.885, 0.5), fontsize=12,
               framealpha=0.9, ncol=1, handlelength=1.8, markerscale=1.0)
    fig.subplots_adjust(wspace=WSPACE, right=0.75)
    _save(fig, figures_dir, f"general_systems_breakdown_m={m}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, help="Directory given to run_experiment.py")
    p.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    p.add_argument("--figures-dir", default=os.path.join(HERE, "figures"))
    args = p.parse_args()

    frames = load_results(args.out_dir, args.results_dir)
    summary = {f"m={m}": summarize(df) for m, df in frames.items()}
    summary["all"] = summarize(pd.concat(frames.values(), ignore_index=True))
    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary["all"], indent=1))

    os.makedirs(args.figures_dir, exist_ok=True)
    plot_main(frames, args.figures_dir)
    for m, df in frames.items():
        plot_breakdown(df, m, args.figures_dir)
    print(f"Wrote results to {args.results_dir} and figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
