#!/usr/bin/env python3

from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

DATASETS = [
    ("data-laptop.csv", "plot-laptop"),
]
ALG_PATTERN = re.compile(r"Mode::(?P<mode>\w+),\s*(?P<k>\d+)>")
METRICS = [
    ("bumped_frac", "Bumped (%)"),
    ("actual_alpha", "Actual $\\alpha$"),
    ("build_ns", "Build time (ns/key)"),
    ("loop_ns", "Query time (ns/query)"),
    # ("throughput_ns", "Throughput (ns)"),
]
LINESTYLES = {
    "SortBump": "-",
    "SortBumpGreedy": "--",
}
LINEWIDTHS = {
    "SortBump": 2.0,
    "SortBumpGreedy": 1.0,
}
BUMP_TICKS = [0, 1, 2, 5, 10]
BUMP_TICK_LABELS = ["0%", "1%", "2%", "5%", "10%"]

HIGHLIGHT = 1.5


def parse_algorithm(alg: str) -> tuple[str, int]:
    match = ALG_PATTERN.search(alg)
    if match is None:
        raise ValueError(f"Could not parse algorithm label: {alg}")
    return match.group("mode"), int(match.group("k"))


def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_path)
    parsed = df["alg"].map(parse_algorithm)
    df["mode"] = parsed.map(lambda item: item[0])
    df["k"] = parsed.map(lambda item: item[1])

    numeric_cols = [
        "n",
        "alpha",
        "factor",
        "target_bits_per_key",
        "actual_bits_per_key",
        "actual_alpha",
        "bumped_frac",
        "build_ns",
        "loop_ns",
        "throughput_ns",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    # df["loop_ns"] = df["loop_ns"] / df["n"]
    # df["throughput_ns"] = df["build_ns"] / df["n"]

    lower_bounds = (
        df[df["factor"] == 1]
        .groupby(["k", "alpha"], as_index=False)["target_bits_per_key"]
        .min()
        .rename(columns={"target_bits_per_key": "lower_bound"})
    )

    median_df = (
        df.groupby(["n", "k", "mode", "alpha", "target_bits_per_key"], as_index=False)
        .agg(
            actual_bits_per_key=("actual_bits_per_key", "median"),
            bumped_frac=("bumped_frac", "median"),
            build_ns=("build_ns", "median"),
            actual_alpha=("actual_alpha", "median"),
            loop_ns=("loop_ns", "median"),
            throughput_ns=("throughput_ns", "median"),
            factor=("factor", "first"),
        )
        .sort_values(["k", "mode", "alpha", "target_bits_per_key"])
        .assign(bumped_frac=lambda frame: frame["bumped_frac"] * 100.0)
    )

    return median_df, lower_bounds


def plot_dataset_ns(data_path: Path, output_name) -> None:
    df, lower_bounds = load_data(data_path)

    for n, data in df.groupby("n"):
        plot_dataset(data, lower_bounds, Path(".") / (output_name + f"-n{n:,}.png"))


def plot_dataset(df, lower_bounds, output_path: Path) -> None:
    k_values = sorted(df["k"].unique())
    alpha_values = sorted(df["alpha"].unique())
    alpha_colors = {
        alpha: color
        for alpha, color in zip(
            alpha_values, plt.cm.viridis(range(0, 256, 256 // len(alpha_values)))
        )
    }
    fig, axes = plt.subplots(
        len(METRICS),
        len(k_values),
        figsize=(6 * len(k_values), 4 * len(METRICS)),
        sharey="row",
        squeeze=False,
    )

    fig.suptitle(f"n={df['n'].iloc[0]:,}", fontsize=16)

    for col, k in enumerate(k_values):
        subset = df[df["k"] == k]
        lb_subset = lower_bounds[lower_bounds["k"] == k]
        x_min = min(subset["actual_bits_per_key"].min(), lb_subset["lower_bound"].min())
        x_max = subset["actual_bits_per_key"].max()
        x_limits = (x_min / 1.1, x_max * 1.1)
        lb_ticks = lb_subset["lower_bound"].tolist()
        xticks = sorted(set(lb_ticks))
        for row, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row][col]
            for (mode, alpha), group in subset.groupby(["mode", "alpha"], sort=True):
                # group = group.sort_values("target_bits_per_key")
                ax.plot(
                    group["actual_bits_per_key"],
                    group[metric],
                    marker=None,
                    linewidth=LINEWIDTHS.get(mode, 2.0),
                    linestyle=LINESTYLES.get(mode, "-"),
                    color=alpha_colors[alpha],
                )
                point_colors = [
                    "red" if f == HIGHLIGHT else alpha_colors[alpha]
                    for f in group["factor"]
                ]
                point_sizes = [
                    (16 if factor == HIGHLIGHT else 14)
                    for factor, value in zip(group["factor"], group[metric])
                ]
                ax.scatter(
                    group["actual_bits_per_key"],
                    group[metric],
                    s=point_sizes,
                    color=point_colors,
                    zorder=3,
                )

                zero_group = group[group["bumped_frac"] == 0]
                zero_colors = [
                    "red" if f == HIGHLIGHT else alpha_colors[alpha]
                    for f in zero_group["factor"]
                ]
                ax.scatter(
                    zero_group["actual_bits_per_key"],
                    zero_group[metric],
                    s=18,
                    facecolors="white",
                    edgecolors=zero_colors,
                    linewidths=1.5,
                    zorder=4,
                )

            ax.set_title(f"k={k}" if row == 0 else "")
            ax.set_xlabel("bits / key" if row == len(METRICS) - 1 else "")
            ax.set_ylabel(ylabel if col == 0 else "")
            ax.set_xscale("log")
            ax.set_xlim(*x_limits)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{tick:.2g}" for tick in xticks])
            for tick_value, tick_label in zip(xticks, ax.get_xticklabels()):
                for _, lb_row in lb_subset.iterrows():
                    if abs(tick_value - lb_row["lower_bound"]) < 1e-12:
                        tick_label.set_color(alpha_colors[lb_row["alpha"]])
                        break
            if metric == "bumped_frac":
                # ax.set_yscale("symlog", linthresh=0.1)
                ax.set_ylim(bottom=-0.4)
                ax.set_yticks(BUMP_TICKS)
                ax.set_yticklabels(BUMP_TICK_LABELS)
            elif metric == "build_ns":
                ax.set_yscale("log")
                ax.set_ylim(bottom=20, top=1000)
                ax.set_yticks([25, 50, 100, 200, 400, 800])
                ax.set_yticklabels(["25", "50", "100", "200", "400", "800"])
            elif metric == "actual_alpha":
                ax.set_ylim(bottom=0.56)
                ax.set_ylim(top=1.0)
            else:
                ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)

            lb_subset = lower_bounds[lower_bounds["k"] == k]
            for _, lb_row in lb_subset.iterrows():
                ax.axvline(
                    lb_row["lower_bound"],
                    color=alpha_colors[lb_row["alpha"]],
                    linewidth=0.8,
                    linestyle="--",
                    alpha=0.7,
                )

            if row == 0 and col == 0:
                mode_handles = [
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        linestyle=LINESTYLES[mode],
                        linewidth=LINEWIDTHS[mode],
                        label=mode,
                    )
                    for mode in LINESTYLES
                ]
                factor_range_handle = Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="black",
                    markersize=6,
                    label="factor$= 1.0/1.25/1.75/2.0/2.25/2.5$",
                    linestyle="none",
                )
                factor_handle = Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="red",
                    markersize=6,
                    label=f"factor$={HIGHLIGHT}$",
                    linestyle="none",
                )
                alpha_handles = [
                    Line2D(
                        [0],
                        [0],
                        color=alpha_colors[a],
                        linewidth=2,
                        label=f"Target $\\alpha={a:g}$",
                    )
                    for a in alpha_values
                ]
                bumped_handle = Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="black",
                    markerfacecolor="white",
                    markersize=5,
                    label="None bumped",
                    linestyle="none",
                    markeredgewidth=1.5,
                )
                ax.legend(
                    handles=mode_handles
                    + [factor_range_handle, factor_handle, bumped_handle]
                    + alpha_handles,
                    ncol=2,
                    loc="upper left",
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def default_output_path(data_path: Path) -> Path:
    if data_path.name.startswith("data"):
        return data_path.with_name(data_path.name.replace("data", "plot", 1)).name
    return data_path.name


def main() -> None:
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1] + ".csv")
        plot_dataset_ns(data_path, sys.argv[1].replace("data", "plot"))
        return

    for data_name, output_name in DATASETS:
        plot_dataset_ns(Path(".") / data_name, output_name)


main()
