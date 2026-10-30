#!/usr/bin/env python3
# Run ./plot.py laptop slides && ./plot.py laptop slides mt
# Output:
# - kphf-set-laptop-slides.pdf
# - kphf-set-laptop-slides-mt.pdf

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import math
import sys
from pathlib import Path

w = 64


def read_data(name):
    """Read and preprocess benchmark data for a given benchmark name."""
    path = Path(name)
    data = pd.read_csv(path if path.is_file() else f"data-{name}.csv")

    data = data[data.n < 500_000_000]

    def cleanup_name(h):
        if "Function2" in h:
            return "PhfSet<PHast+minimal>"
        if "Perfect" in h:
            return "PhfSet<PHast+>"
        if "PtrHash" in h:
            return "PhfSet<PtrHash>"
        if "KptrHash" in h:
            return "kPhfSet<kPtrHash>"
        if "FxHashSet" in h:
            return "HashSet"
        return h

    # Older CSVs call the same measured allocation/input ratio "overhead".
    if "memory_ratio" not in data.columns:
        data["memory_ratio"] = data["overhead"]

    # Preserve support for older CSVs without inventing placement statistics.
    if "primary_hit_ratio" not in data.columns:
        data["primary_hit_ratio"] = float("nan")

    data["h"] = data["h"].map(cleanup_name)
    data = data[~(data["h"] == "FphMetaSet")]
    data = data[~(data["h"] == "MapEmbed")]
    data = data[~(data["h"] == "Hd8Set")]
    data = data[~(data["h"] == "U64HashSet")]
    data = data[~(data["h"] == "TPHT")]
    data = data[~(data["metric"] == "prefetch")]

    queries = ["q01", "q50", "q99"]
    group_columns = [
        "h",
        "pf",
        "threads",
        "metric",
        "n",
        "alpha",
        "kphf_target_bits_per_key",
    ]

    data = data.groupby(group_columns, as_index=False, sort=False)[
        ["build", "load_factor", "memory_ratio", "primary_hit_ratio", *queries]
    ].min()

    return data


metric_name = {
    "loop": "for loop",
    "prefetch2": "with prefetching",
}

cpu_names = {
    "laptop": "Intel i7-10750H laptop",
    "diffie": "AMD EPYC Zen 4 9684X server",
    "floyd": "Intel Xeon Gold 6530 server",
}

queries = ["q50"]

label_color = {
    "HashSet": "red",
    "CuckooSet<Lazy>": "orange",
    "CuckooSet<Eager>": "brown",
    "CuckooQuotientSet<Lazy>": "teal",
    "CuckooQuotientSet<Eager>": "navy",
    "CuckooQuotientSet<Lazy,8,SIMD>": "lightseagreen",
    "CuckooQuotientSet<Eager,8,SIMD>": "royalblue",
    "CuckooQuotientSet<Lazy,16,SIMD>": "teal",
    "CuckooQuotientSet<Eager,16,SIMD>": "navy",
    "CuckooQuotientSet<Lazy,32,SIMD>": "darkgreen",
    "CuckooQuotientSet<Eager,32,SIMD>": "steelblue",
    "BraidedCuckooSet<Lazy>": "purple",
    "BraidedCuckooSet<Eager>": "magenta",
    "TPHT": "gray",
    "TinyPtr": "olive",
    "FphDynSet": "pink",
    "PhfSet<PHast+>": "lime",
    "PhfSet<PtrHash>": "cyan",
    "kPhfSet<kPtrHash>": "blue",
    "MockHashSet": "black",
    # "U64HashSet": "magenta",
    # "Hd8Set": "black",
    # "MapEmbed": "brown",
}

display_name = {
    "CuckooQuotientSet<Lazy,8,SIMD>": "Quotient<Lazy, 8-bit SIMD>",
    "CuckooQuotientSet<Eager,8,SIMD>": "Quotient<Eager, 8-bit SIMD>",
    "CuckooQuotientSet<Lazy,16,SIMD>": "Quotient<Lazy, 16-bit SIMD>",
    "CuckooQuotientSet<Eager,16,SIMD>": "Quotient<Eager, 16-bit SIMD>",
    "CuckooQuotientSet<Lazy,32,SIMD>": "Quotient<Lazy, 32-bit SIMD>",
    "CuckooQuotientSet<Eager,32,SIMD>": "Quotient<Eager, 32-bit SIMD>",
    "FphDynSet": "FPH",
    "PhfSet<PHast+>": "PHF-set<PHast+>",
    "PhfSet<PtrHash>": "PHF-set<PtrHash>",
    "kPhfSet<kPtrHash>": "$k$-PHF-set<$k$-PtrHash>",
    # "U64HashSet": "magenta",
    # "Hd8Set": "black",
    # "MapEmbed": "brown",
}


def get_display_name(h):
    if h in display_name:
        return display_name[h]
    return h


def series_label(h, subset):
    """Show measured occupancy and allocation/input ratio across plotted sizes."""

    def value_range(column):
        lo, hi = subset[column].min(), subset[column].max()
        if round(lo, 2) == round(hi, 2):
            return f"{lo:.2f}"
        return f"{lo:.2f}–{hi:.2f}"

    primary_label = ""
    if "Cuckoo" in h:
        ratios = subset["primary_hit_ratio"].dropna() * 100
        if not ratios.empty:
            lo, hi = ratios.min(), ratios.max()
            percent = f"{lo:.1f}%" if round(lo, 1) == round(hi, 1) else f"{lo:.1f}–{hi:.1f}%"
            primary_label = f"; primary {percent}"
        else:
            primary_label = "; primary n/a"
    return (
        get_display_name(h)
        + f" ($\\alpha={value_range('load_factor')}$; "
        + f"{value_range('memory_ratio')}×{primary_label})"
    )


def width(name, alpha):
    if name != "kPhfSet<kPtrHash>":
        return 1.5
    return {
        0.7: 1.5,
        0.9: 1.1,
        0.95: 0.8,
    }[alpha]


def line_style(name):
    if name.startswith("CuckooQuotientSet<"):
        if ",8,SIMD>" in name:
            return "--"
        if ",32,SIMD>" in name:
            return ":"
    return "-"


def build_legend(fig, axes, bbox_y_anchor):
    """Build and add a custom legend to the figure with category headers."""
    handles, labels = axes[0][0].get_legend_handles_labels()
    groups = {
        "Probing": [],
        "1-PHF-set": [],
        "$k$-PHF-set": [],
        "Lower bound": [],
    }
    for handle, label in zip(handles, labels):
        if label.startswith("MockHashSet"):
            category = "Lower bound"
        elif label.startswith("$k$-PHF-set"):
            category = "$k$-PHF-set"
        elif label.startswith("PHF-set"):
            category = "1-PHF-set"
        else:
            category = "Probing"
        groups[category].append((handle, label))
    groups = {title: entries for title, entries in groups.items() if entries}
    rows = 1 + max(map(len, groups.values()))
    new_handles, new_labels = [], []
    for title, entries in groups.items():
        column = [(mpatches.Patch(visible=False), title), *entries]
        column += [(mpatches.Patch(visible=False), "")] * (rows - len(column))
        for handle, label in column:
            new_handles.append(handle)
            new_labels.append(label)
    legend = fig.legend(
        new_handles,
        new_labels,
        loc="upper center",
        ncols=len(groups),
        fontsize=9,
        bbox_to_anchor=(0.5, bbox_y_anchor),
    )
    for i, text in enumerate(legend.get_texts()):
        if i % rows == 0:
            text.set_fontweight("bold")


titles = ["50% positive"]
names = sys.argv[1:]
cpu_override = None
if "--cpu" in names:
    idx = names.index("--cpu")
    cpu_override = names[idx + 1]
    del names[idx : idx + 2]
slides = "slides" in names
if slides:
    names.remove("slides")
prefetch = "prefetch" in names
if prefetch:
    names.remove("prefetch")
mt = "mt" in names
if mt:
    names.remove("mt")

if "single" in names:
    names.remove("single")

    plt.close()

    metric = "prefetch2"
    threads = 1

    nrows = 3  # One row for each benchmark: laptop, floyd, diffie
    ncols = len(queries)
    figsize = (10, 3 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharey=True,
        sharex=True,
        squeeze=False,
    )

    # Load data for all three benchmarks
    benchmark_names = ["laptop", "floyd", "diffie"]
    benchmark_data = {bname: read_data(bname) for bname in benchmark_names}

    for row_idx, bname in enumerate(benchmark_names):
        bdata = benchmark_data[bname]
        row_data = bdata[(bdata["threads"] == threads) & (bdata["metric"] == metric)]
        groups = row_data.groupby(["h", "alpha"], sort=False)

        for ci, (q, title) in enumerate(zip(queries, titles)):
            ax = axes[row_idx][ci]
            for (h, alpha), subset in sorted(
                groups,
                key=lambda x: (
                    (
                        list(label_color).index(x[0][0])
                        if x[0][0] in label_color
                        else len(label_color)
                    ),
                    1 - x[0][1],
                ),
            ):
                sns.lineplot(
                    data=subset,
                    x="n",
                    y=q,
                    ax=ax,
                    estimator=None,
                    errorbar=None,
                    color=label_color[h],
                    lw=width(h, alpha),
                    linestyle=line_style(h),
                    marker="o",
                    markersize=3,
                    label=series_label(h, subset),
                )
            if row_idx == 0:
                ax.set_title(title)
            ax.set_xlabel("n" if row_idx == nrows - 1 else "")
            ax.set_xscale("log", base=2)
            ax.grid(True, which="both", ls="--", lw=0.5)
            ax.set_ylabel(f"{cpu_names[bname]}\nns / query" if ci == 0 else "")
            ax.set_ylim(0)
            ax.legend().remove() if ax.get_legend() else None

    build_legend(fig, axes, -0.02)

    fig.suptitle(f"StaticHashset<u64> query throughput (1 thread, loop with prefetch)")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    # fig.savefig(f"kphf-set.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"kphf-set.pdf", bbox_inches="tight")
    # fig.savefig(f"kphf-set.svg", bbox_inches="tight")


for name in names:
    data = read_data(name)
    cpu_name = cpu_names.get(cpu_override or name, cpu_override or Path(name).stem)
    max_threads = max(data["threads"].unique())

    print(name)
    plt.close()

    thread_specs = sorted(data["threads"].unique()) if mt else [1]
    metric_specs = ["loop", "prefetch2"]
    if slides:
        thread_specs = [12 if mt else 1]

    nrows = len(thread_specs)
    ncols = len(metric_specs)
    figsize = (4 * ncols, 3 * nrows if not slides else 4.5)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharey="row",
        sharex=True,
        squeeze=False,
    )

    for ri, threads in enumerate(thread_specs):
        for ci, metric in enumerate(metric_specs):
            row_data = data[(data["threads"] == threads) & (data["metric"] == metric)]
            groups = row_data.groupby(["h", "alpha"], sort=False)
            ax = axes[ri][ci]
            for (h, alpha), subset in sorted(
                groups,
                key=lambda x: (
                    (
                        list(label_color).index(x[0][0])
                        if x[0][0] in label_color
                        else len(label_color)
                    ),
                    1 - x[0][1],
                ),
            ):
                sns.lineplot(
                    data=subset,
                    x="n",
                    y=queries[0],
                    ax=ax,
                    estimator=None,
                    errorbar=None,
                    color=label_color[h],
                    lw=width(h, alpha),
                    linestyle=line_style(h),
                    marker="o",
                    markersize=3,
                    label=series_label(h, subset),
                )
            if ri == 0:
                ax.set_title(metric_name[metric])
            ax.set_xlabel("n" if ri == nrows - 1 else "")
            ax.set_xscale("log", base=2)
            ax.grid(True, which="both", ls="--", lw=0.5)
            ax.set_ylabel(f"ns / query" if ci == 0 else "")
            ax.set_ylim(0)
            ax.legend().remove() if ax.get_legend() else None
            if slides:
                ax.set_ylim(0, 6.5 if mt else 30)

    build_legend(fig, axes, -0.02)

    if mt:
        fig.suptitle(
            f"Hashset<u64> query throughput (50% positive queries, {cpu_name}, 12 threads)"
        )
    else:
        fig.suptitle(
            f"Hashset<u64> query throughput (50% positive queries, {cpu_name}, 1 thread)"
        )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = Path(name).stem
    if slides:
        out += "-slides"
        if mt:
            out += "-mt"
    fig.savefig(f"kphf-set-{out}.png", bbox_inches="tight", dpi=180)
    fig.savefig(f"kphf-set-{out}.pdf", bbox_inches="tight")
    # fig.savefig(f"kphf-set-{name}.svg", bbox_inches="tight")
