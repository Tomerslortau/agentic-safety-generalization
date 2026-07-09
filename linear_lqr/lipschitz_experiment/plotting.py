import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MultipleLocator, NullLocator, ScalarFormatter
from typing import List

from config import (
    ALIGNMENT_CONSTANT,
    EPS,
    FIGURES_DIR,
    resolve_theory_path,
    results_csv_template,
)


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
WIDTH = 5.2
Y_LABEL = r"$\mathrm{Lip}(K_{\mathrm{safe}})\; / \;\mathrm{Lip}(K_{\mathrm{unsafe}})$"
Y_LABEL_FRACTION = (
    r"$\frac{\mathrm{Lip}(\mathbf{K}_{\mathrm{safe}})}{\mathrm{Lip}(\mathbf{K}_{\mathrm{unsafe}})}$"
)
HEIGHT = 1.9
WSPACE = 0.2


def _format_value(value: float) -> str:
    """Compact numeric formatting for titles and filenames."""
    return f"{value:g}"


def _norm_label(symbol: str) -> str:
    """Return a mathtext 2-norm label."""
    return rf"$\|{symbol}\|_2$"


def _resolve_y_label(ylabel_frac: bool = False) -> str:
    """Return the preferred y-axis label format."""
    return Y_LABEL_FRACTION if ylabel_frac else Y_LABEL


def _find_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    """Return the first matching column name from a list of aliases."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not find a column for {label}. Tried: {', '.join(candidates)}"
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Support both current and legacy result-file schemas."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    rename_map = {
        _find_column(df, ["norm_A", "A_norm"], "||A||"): "norm_A",
        _find_column(df, ["norm_B", "B_norm"], "||B||"): "norm_B",
        _find_column(df, ["norm_D", "norm_E", "D_norm", "E_norm"], "||D||"): "norm_D",
        _find_column(df, ["min_eig_R", "norm_R", "R_norm"], "lambda_min(R)"): "min_eig_R",
        _find_column(df, ["L_lqr_emp"], "empirical LQR Lipschitz"): "L_lqr_emp",
        _find_column(df, ["L_hinf_emp"], "empirical Hinf Lipschitz"): "L_hinf_emp",
    }
    return df.rename(columns=rename_map)


def _build_legend_handles(norm_B_values, colors):
    """Create consistent legend entries for B-norm curves."""
    handles = []
    labels = []
    for idx, norm_B in enumerate(norm_B_values):
        color = colors[idx]
        marker = MARKERS[idx % len(MARKERS)]
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="-",
                color=color,
                linewidth=1.8,
                marker=marker,
                markersize=5.5,
                markeredgecolor=color,
                markeredgewidth=1.4,
                markerfacecolor="white",
            )
        )
        labels.append(rf"{_norm_label('B')}={_format_value(norm_B)}")
    return handles, labels


def _bound_curve(
    norm_A_values: np.ndarray,
    norm_B: float,
    min_eig_R: float,
    alignment_constant: float,
) -> np.ndarray:
    """Compute the commuting-experiment bound curve."""
    return (alignment_constant ** 3) / (
        norm_A_values * (2.0 + 4.0 * (norm_B ** 2) / min_eig_R)
    )


def _endpoint_yerr(mean: np.ndarray, lower_endpoint: np.ndarray, upper_endpoint: np.ndarray) -> np.ndarray:
    """Return errorbar distances from absolute lower/upper endpoints."""
    lower_error = np.maximum(mean - lower_endpoint, 0.0)
    upper_error = np.maximum(upper_endpoint - mean, 0.0)
    return np.vstack([lower_error, upper_error])


def _log_ratio_stats(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Aggregate positive empirical ratios in log space for log-y plots."""
    df = df[np.isfinite(df["ratio_emp_hinf_over_lqr"]) & (df["ratio_emp_hinf_over_lqr"] > 0.0)].copy()
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["mean", "std", "count", "lower", "upper"])

    df["log_ratio_emp_hinf_over_lqr"] = np.log(df["ratio_emp_hinf_over_lqr"])
    stats = (
        df.groupby(group_cols, as_index=False)
        .agg(
            log_mean=("log_ratio_emp_hinf_over_lqr", "mean"),
            log_std=("log_ratio_emp_hinf_over_lqr", "std"),
            count=("log_ratio_emp_hinf_over_lqr", "count"),
        )
        .sort_values(group_cols)
    )
    log_std = stats["log_std"].fillna(0.0).to_numpy()
    log_mean = stats["log_mean"].to_numpy()
    stats["mean"] = np.exp(log_mean)
    stats["std"] = log_std
    stats["lower"] = np.exp(log_mean - log_std)
    stats["upper"] = np.exp(log_mean + log_std)
    return stats


def _log_bound_stats(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Aggregate commuting bounds in log space for aggregated plots."""
    df = df.copy()
    if "alignment_constant" in df.columns:
        alignment = df["alignment_constant"].fillna(ALIGNMENT_CONSTANT)
    else:
        alignment = ALIGNMENT_CONSTANT

    df["bound_for_plot"] = (alignment ** 3) / (
        df["norm_A"] * (2.0 + 4.0 * (df["norm_B"] ** 2) / df["min_eig_R"])
    )
    df = df[np.isfinite(df["bound_for_plot"]) & (df["bound_for_plot"] > 0.0)].copy()
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["bound"])

    df["log_bound_for_plot"] = np.log(df["bound_for_plot"])
    stats = (
        df.groupby(group_cols, as_index=False)
        .agg(log_bound=("log_bound_for_plot", "mean"))
        .sort_values(group_cols)
    )
    stats["bound"] = np.exp(stats["log_bound"].to_numpy())
    return stats


def _format_log_y_axis(ax) -> None:
    """Use decade ticks only, avoiding dense minor horizontal gridlines."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.grid(True, axis="y", which="major", alpha=0.35)


def _x_ticks_for_range(values, step: float) -> np.ndarray:
    """Return linear ticks spanning the available x-values."""
    x_min = float(np.min(values))
    x_max = float(np.max(values))
    x_tick_start = np.floor((x_min / step) + EPS) * step
    x_tick_end = np.ceil((x_max / step) - EPS) * step
    return np.round(np.arange(x_tick_start, x_tick_end + 0.5 * step, step), 10)


def _save_figure(fig, figures_dir: str, stem: str) -> List[str]:
    """Save a figure in raster and vector formats."""
    saved_paths = []
    for extension in ("png", "pdf"):
        output_path = os.path.join(figures_dir, f"{stem}.{extension}")
        fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
        saved_paths.append(output_path)
    return saved_paths


def plot_emp_ratio(n: int, experiment_type: str = None, ylabel_frac: bool = False) -> List[str]:
    """Create empirical H-infinity/LQR ratio figures for an experiment."""
    experiment_type = "general" if experiment_type is None else experiment_type
    csv_path = resolve_theory_path(results_csv_template(experiment_type).format(n))
    df = _normalize_columns(pd.read_csv(csv_path))
    y_label = _resolve_y_label(ylabel_frac)

    df["ratio_emp_hinf_over_lqr"] = np.where(
        np.abs(df["L_lqr_emp"]) > EPS,
        df["L_hinf_emp"] / df["L_lqr_emp"],
        np.nan,
    )

    figures_dir = resolve_theory_path(FIGURES_DIR)
    os.makedirs(figures_dir, exist_ok=True)

    available_norm_D = sorted(df["norm_D"].dropna().unique())
    available_min_eig_R = sorted(df["min_eig_R"].dropna().unique())
    available_norm_B = sorted(df["norm_B"].dropna().unique())
    available_norm_A = sorted(df["norm_A"].dropna().unique())

    if len(available_norm_D) == 0:
        raise ValueError(f"No {_norm_label('D')} values found in {csv_path}")
    if len(available_min_eig_R) == 0:
        raise ValueError(f"No {_norm_label('R^{-1}')} values found in {csv_path}")
    if len(available_norm_B) == 0:
        raise ValueError(f"No {_norm_label('B')} values found in {csv_path}")
    alignment_constant = ALIGNMENT_CONSTANT
    if experiment_type == "commuting" and "alignment_constant" in df.columns:
        alignment_values = df["alignment_constant"].dropna().unique()
        if len(alignment_values) > 0:
            alignment_constant = float(alignment_values[0])

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(len(available_norm_B))]
    if experiment_type == "commuting":
        legend_handles = [
            Line2D(
                [0],
                [0],
                linestyle="-",
                color=colors[0],
                linewidth=2,
                marker=MARKERS[0],
                markersize=6,
                markeredgecolor=colors[0],
                markeredgewidth=1.5,
                markerfacecolor="white",
            ),
            Line2D([0], [0], linestyle="--", color="black", linewidth=1.8),
        ]
        legend_labels = ["Empirical", "Bound"]
    else:
        legend_handles, legend_labels = _build_legend_handles(available_norm_B, colors)

    saved_paths = []

    for norm_D in available_norm_D:
        df_d = df[df["norm_D"] == norm_D]
        if df_d.empty:
            continue

        n_cols = len(available_min_eig_R)
        fig, axes = plt.subplots(
            1,
            n_cols,
            figsize=(WIDTH * n_cols + 0.8, HEIGHT),
            sharey=True,
            sharex=True,
        )
        if n_cols == 1:
            axes = [axes]

        x_tick_step = 0.05 if experiment_type == "commuting" else 0.1
        x_ticks = _x_ticks_for_range(available_norm_A, x_tick_step)

        for col_idx, min_eig_R in enumerate(available_min_eig_R):
            ax = axes[col_idx]
            df_sub = df_d[df_d["min_eig_R"] == min_eig_R]
            subplot_ymin = np.inf
            subplot_ymax = 0.0

            for b_idx, norm_B in enumerate(available_norm_B):
                df_curve = df_sub[df_sub["norm_B"] == norm_B]
                if df_curve.empty:
                    continue

                stats = _log_ratio_stats(df_curve, ["norm_A"])
                if stats.empty:
                    continue

                x = stats["norm_A"].to_numpy()
                y = stats["mean"].to_numpy()
                lower_endpoint = stats["lower"].to_numpy()
                upper_endpoint = stats["upper"].to_numpy()
                yerr = _endpoint_yerr(y, lower_endpoint, upper_endpoint)
                subplot_ymin = min(subplot_ymin, float(np.nanmin(lower_endpoint)))
                subplot_ymax = max(subplot_ymax, float(np.nanmax(upper_endpoint)))
                color = colors[b_idx]
                marker = MARKERS[b_idx % len(MARKERS)]

                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    capsize=3,
                    linestyle="-",
                    color=color,
                    linewidth=2,
                    elinewidth=1.2,
                    marker=marker,
                    markersize=6,
                    markeredgecolor=color,
                    markeredgewidth=1.5,
                    markerfacecolor="white",
                    zorder=3,
                )

                if experiment_type == "commuting":
                    bound = _bound_curve(x, norm_B, min_eig_R, alignment_constant)
                    positive_bound = bound[bound > 0.0]
                    if len(positive_bound) > 0:
                        subplot_ymin = min(subplot_ymin, float(np.nanmin(positive_bound)))
                    subplot_ymax = max(subplot_ymax, float(np.nanmax(bound)))
                    ax.plot(
                        x,
                        bound,
                        linestyle="--",
                        color="black",
                        linewidth=1.8,
                        zorder=2,
                    )

            if experiment_type == "commuting":
                ax.set_title("")
            else:
                ax.set_title(rf"{_norm_label('R^{-1}')}={_format_value(min_eig_R)}", fontsize=13)
            ax.axhline(1.0, color="red", linestyle=":", linewidth=1.5, zorder=1)
            _format_log_y_axis(ax)
            ax.set_xticks(x_ticks)
            ax.xaxis.set_major_locator(MultipleLocator(x_tick_step))
            ax.grid(True, axis="x", alpha=0.5)
            ax.tick_params(axis="both", which="major", labelsize=12, labelbottom=True, labelleft=True)
            ax.set_xlabel(_norm_label("A"), fontsize=14)
            ax.set_xlim(x_ticks[0] - 0.3 * x_tick_step, x_ticks[-1] + 0.3 * x_tick_step)
            subplot_ymin = min(subplot_ymin, 1.0)
            subplot_ymax = max(subplot_ymax, 1.0)
            if np.isfinite(subplot_ymin) and subplot_ymax > 0:
                ax.set_ylim(subplot_ymin / 1.5, 1.5 * subplot_ymax)
            else:
                ax.set_ylim(0.5, 1.0)
            if col_idx == 0:
                ax.set_ylabel(y_label, fontsize=14)

        if experiment_type == "commuting":
            fig.legend(
                legend_handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(0.85, 0.5),
                fontsize=12,
                framealpha=0.9,
                ncol=1,
                handlelength=1.8,
                markerscale=1.0,
            )
            fig.subplots_adjust(wspace=WSPACE, right=0.82)
        else:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="center right",
                bbox_to_anchor=(0.885, 0.5),
                fontsize=12,
                framealpha=0.9,
                ncol=1,
                handlelength=1.8,
                markerscale=1.0,
            )
            fig.subplots_adjust(wspace=WSPACE, right=0.75)
        filename_stem = f"{experiment_type}_emp_ratio_n={n}_D={_format_value(norm_D)}"
        saved_paths.extend(_save_figure(fig, figures_dir, filename_stem))
        plt.close(fig)

    return saved_paths


def _aggregated_ratio_stats(n: int, experiment_type: str) -> pd.DataFrame:
    """Load result data and aggregate empirical ratios by A norm."""
    experiment_type = "general" if experiment_type is None else experiment_type
    csv_path = resolve_theory_path(results_csv_template(experiment_type).format(n))
    df = _normalize_columns(pd.read_csv(csv_path))

    df["ratio_emp_hinf_over_lqr"] = np.where(
        np.abs(df["L_lqr_emp"]) > EPS,
        df["L_hinf_emp"] / df["L_lqr_emp"],
        np.nan,
    )

    stats = _log_ratio_stats(df, ["norm_A"])
    if stats.empty:
        raise ValueError(f"No positive finite empirical ratios found in {csv_path}")
    if experiment_type == "commuting":
        bound_stats = _log_bound_stats(df, ["norm_A"])
        stats = stats.merge(bound_stats[["norm_A", "bound"]], on="norm_A", how="left")
    return stats

def plot_emp_ratio_comparison(n: int, ylabel_frac: bool = False) -> List[str]:
    """Create a side-by-side commuting/general aggregated comparison plot."""
    stats_by_experiment = {
        "commuting": _aggregated_ratio_stats(n, "commuting"),
        "general": _aggregated_ratio_stats(n, "general"),
    }
    y_label = _resolve_y_label(ylabel_frac)
    figures_dir = resolve_theory_path(FIGURES_DIR)
    os.makedirs(figures_dir, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH * 2 + 0.8, HEIGHT),
        sharey=True,
        sharex=False,
    )

    color = plt.get_cmap("tab10")(0)
    all_lower = np.concatenate(
        [stats["lower"].to_numpy() for stats in stats_by_experiment.values()]
    )
    all_upper = np.concatenate(
        [stats["upper"].to_numpy() for stats in stats_by_experiment.values()]
    )
    all_bound = stats_by_experiment["commuting"]["bound"].dropna().to_numpy()
    y_min = float(np.nanmin(all_lower))
    y_max = float(np.nanmax(all_upper))
    if len(all_bound) > 0:
        y_min = min(y_min, float(np.nanmin(all_bound)))
        y_max = max(y_max, float(np.nanmax(all_bound)))

    for ax, experiment_type, title in zip(
        axes,
        ["commuting", "general"],
        ["Assumptions Met", "Assumptions Can Be Violated"],
    ):
        stats = stats_by_experiment[experiment_type]
        x_tick_step = 0.05 if experiment_type == "commuting" else 0.1
        x = stats["norm_A"].to_numpy()
        y = stats["mean"].to_numpy()
        lower_endpoint = stats["lower"].to_numpy()
        upper_endpoint = stats["upper"].to_numpy()
        yerr = _endpoint_yerr(y, lower_endpoint, upper_endpoint)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            capsize=3,
            linestyle="-",
            color=color,
            linewidth=2,
            elinewidth=1.2,
            marker=MARKERS[0],
            markersize=6,
            markeredgecolor=color,
            markeredgewidth=1.5,
            markerfacecolor="white",
            zorder=3,
        )

        if experiment_type == "commuting" and "bound" in stats.columns:
            bound_stats = stats.dropna(subset=["bound"])
            if not bound_stats.empty:
                ax.plot(
                    bound_stats["norm_A"].to_numpy(),
                    bound_stats["bound"].to_numpy(),
                    linestyle="--",
                    color="black",
                    linewidth=1.8,
                    zorder=2,
                )

        ax.set_title(title, fontsize=13)
        ax.axhline(1.0, color="red", linestyle=":", linewidth=1.5, zorder=1)
        _format_log_y_axis(ax)
        x_ticks = _x_ticks_for_range(x, x_tick_step)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_locator(MultipleLocator(x_tick_step))
        ax.grid(True, axis="x", alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=12, labelbottom=True, labelleft=True)
        ax.set_xlabel(_norm_label("A"), fontsize=14)
        ax.set_xlim(x_ticks[0] - 0.3 * x_tick_step, x_ticks[-1] + 0.3 * x_tick_step)

    axes[0].set_ylabel(y_label, fontsize=14)
    y_min = min(y_min, 1.0)
    y_max = max(y_max, 1.0)
    if y_max > 0:
        axes[0].set_ylim(y_min / 1.5, 1.5 * y_max)
    else:
        axes[0].set_ylim(0.5, 1.0)

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="-",
            color=color,
            linewidth=2,
            marker=MARKERS[0],
            markersize=6,
            markeredgecolor=color,
            markeredgewidth=1.5,
            markerfacecolor="white",
        ),
        Line2D([0], [0], linestyle="--", color="black", linewidth=1.8),
    ]
    axes[0].legend(
        legend_handles,
        ["Empirical", "Bound"],
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
        ncol=1,
        handlelength=1.5,
        handletextpad=0.5,
        borderpad=0.35,
        labelspacing=0.3,
        markerscale=0.85,
    )
    fig.subplots_adjust(wspace=WSPACE)

    filename_stem = f"comparison_emp_ratio_n={n}"
    saved_paths = _save_figure(fig, figures_dir, filename_stem)
    plt.close(fig)
    return saved_paths


def plot_general_emp_ratio(n: int, ylabel_frac: bool = False) -> List[str]:
    """Backward-compatible wrapper for the general empirical-ratio plot."""
    return plot_emp_ratio(n, experiment_type="general", ylabel_frac=ylabel_frac)