#!/usr/bin/env python3
"""Figures of the general-systems experiment, in the style of plotting.py.

Writes the main-text figure (ratio vs ||A||_2, one panel per dimension, no bound) and
one appendix breakdown per dimension (panels by ||R^{-1}||_2, curves by ||B||_2, with
the Theorem 1 coefficient where Assumption 1 holds). Each is saved as PDF and PNG.
"""

import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from plotting import (
    HEIGHT,
    MARKERS,
    WIDTH,
    WSPACE,
    Y_LABEL,
    _build_legend_handles,
    _endpoint_yerr,
    _format_log_y_axis,
    _format_value,
    _log_ratio_stats,
    _norm_label,
    _save_figure,
    _x_ticks_for_range,
)

X_TICK_STEP = 0.1


def load_results(results_dir):
    frames = {}
    for path in glob.glob(os.path.join(results_dir, "general_systems_m=*.csv")):
        m = int(re.search(r"m=(\d+)\.csv$", path).group(1))
        frames[m] = pd.read_csv(path)
    if not frames:
        raise SystemExit(f"No general_systems_m=*.csv files in {results_dir}")
    return dict(sorted(frames.items()))


def ratio_stats(df, column="ratio"):
    return _log_ratio_stats(df.assign(ratio_emp_hinf_over_lqr=df[column]), ["norm_A"])


def coef_stats(df):
    """Mean and std of log(coef) per ||A||_2, only where Assumption 1 holds."""
    held = df[df["assumption_1_holds"].astype(bool) & (df["coef"] > 0.0)]
    if held.empty:
        return pd.DataFrame(columns=["norm_A", "mean", "lower", "upper"])
    stats = (held.assign(log_coef=np.log(held["coef"]))
             .groupby("norm_A", as_index=False)
             .agg(log_mean=("log_coef", "mean"), log_std=("log_coef", "std")))
    log_std = stats["log_std"].fillna(0.0)
    stats["mean"] = np.exp(stats["log_mean"])
    stats["lower"] = np.exp(stats["log_mean"] - log_std)
    stats["upper"] = np.exp(stats["log_mean"] + log_std)
    return stats


def _errorbar(ax, stats, color, marker):
    x, y = stats["norm_A"].to_numpy(), stats["mean"].to_numpy()
    yerr = _endpoint_yerr(y, stats["lower"].to_numpy(), stats["upper"].to_numpy())
    ax.errorbar(x, y, yerr=yerr, capsize=3, linestyle="-", color=color, linewidth=2,
                elinewidth=1.2, marker=marker, markersize=6, markeredgecolor=color,
                markeredgewidth=1.5, markerfacecolor="white", zorder=3)


def _set_decade_ticks(ax, y_low, y_high):
    """Decade ticks that always include y = 1 and label values below one correctly."""
    low, high = int(np.ceil(np.log10(y_low))), int(np.floor(np.log10(y_high)))
    step = 1 if high - low <= 4 else 2
    ticks = [10.0 ** k for k in range(low, high + 1) if k % step == 0]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))


def _finish_axis(ax, x_values, y_min, y_max):
    ax.axhline(1.0, color="red", linestyle=":", linewidth=1.5, zorder=1)
    _format_log_y_axis(ax)
    x_ticks = _x_ticks_for_range(x_values, X_TICK_STEP)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_locator(MultipleLocator(X_TICK_STEP))
    ax.grid(True, axis="x", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=12, labelbottom=True, labelleft=True)
    ax.set_xlabel(_norm_label("A"), fontsize=14)
    ax.set_xlim(x_ticks[0] - 0.3 * X_TICK_STEP, x_ticks[-1] + 0.3 * X_TICK_STEP)
    y_low, y_high = min(y_min, 1.0) / 1.5, 1.5 * max(y_max, 1.0)
    ax.set_ylim(y_low, y_high)
    _set_decade_ticks(ax, y_low, y_high)


def plot_main(frames, figures_dir):
    """Main-text figure: one panel per dimension, 30 systems per point, no bound."""
    n_cols = len(frames)
    fig, axes = plt.subplots(1, n_cols, figsize=(WIDTH * n_cols + 0.8, HEIGHT), sharey=True)
    axes = np.atleast_1d(axes)
    color = plt.get_cmap("tab10")(0)
    all_stats = {m: ratio_stats(df) for m, df in frames.items()}
    y_min = min(float(s["lower"].min()) for s in all_stats.values())
    y_max = max(float(s["upper"].max()) for s in all_stats.values())
    for ax, (m, stats) in zip(axes, all_stats.items()):
        _errorbar(ax, stats, color, MARKERS[0])
        ax.set_title(rf"$m={m}$", fontsize=13)
        _finish_axis(ax, stats["norm_A"].to_numpy(), y_min, y_max)
    axes[0].set_ylabel(Y_LABEL, fontsize=14)
    fig.subplots_adjust(wspace=WSPACE)
    paths = _save_figure(fig, figures_dir, "general_systems_ratio")
    plt.close(fig)
    return paths


def plot_breakdown(df, m, figures_dir):
    """Appendix breakdown for one dimension."""
    norm_Rinv_values = sorted(df["norm_Rinv"].unique())
    norm_B_values = sorted(df["norm_B"].unique())
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(len(norm_B_values))]
    n_cols = len(norm_Rinv_values)
    fig, axes = plt.subplots(1, n_cols, figsize=(WIDTH * n_cols + 0.8, HEIGHT),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    y_min, y_max = np.inf, 0.0
    for ax, norm_Rinv in zip(axes, norm_Rinv_values):
        for b_idx, norm_B in enumerate(norm_B_values):
            curve = df[(df["norm_Rinv"] == norm_Rinv) & (df["norm_B"] == norm_B)]
            stats = ratio_stats(curve)
            if stats.empty:
                continue
            _errorbar(ax, stats, colors[b_idx], MARKERS[b_idx % len(MARKERS)])
            y_min = min(y_min, float(stats["lower"].min()))
            y_max = max(y_max, float(stats["upper"].max()))
            bound = coef_stats(curve)
            if not bound.empty:
                x = bound["norm_A"].to_numpy(dtype=float)
                ax.plot(x, bound["mean"].to_numpy(dtype=float), linestyle="--",
                        color=colors[b_idx], linewidth=1.6, zorder=2)
                ax.fill_between(x, bound["lower"].to_numpy(dtype=float),
                                bound["upper"].to_numpy(dtype=float),
                                color=colors[b_idx], alpha=0.15, linewidth=0, zorder=1)
                y_min = min(y_min, float(bound["lower"].min()))
                y_max = max(y_max, float(bound["upper"].max()))
        ax.set_title(rf"{_norm_label('R^{-1}')}={_format_value(norm_Rinv)}", fontsize=13)
    for ax in axes:
        _finish_axis(ax, df["norm_A"].to_numpy(), y_min, y_max)
    axes[0].set_ylabel(Y_LABEL, fontsize=14)
    handles, labels = _build_legend_handles(norm_B_values, colors)
    handles.append(Line2D([0], [0], linestyle="--", color="black", linewidth=1.6))
    labels.append("Bound")
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.885, 0.5), fontsize=12,
               framealpha=0.9, ncol=1, handlelength=1.8, markerscale=1.0)
    fig.subplots_adjust(wspace=WSPACE, right=0.75)
    paths = _save_figure(fig, figures_dir, f"general_systems_breakdown_m={m}")
    plt.close(fig)
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=C.resolve_theory_path(C.GENERAL_SYSTEMS_RESULTS_DIR))
    p.add_argument("--figures-dir", default=C.resolve_theory_path(C.FIGURES_DIR))
    args = p.parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)
    frames = load_results(args.results_dir)
    paths = plot_main(frames, args.figures_dir)
    for m, df in frames.items():
        paths.extend(plot_breakdown(df, m, args.figures_dir))
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
