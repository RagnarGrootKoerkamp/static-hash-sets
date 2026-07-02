#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import math
import sys

w = 64


def read_data(name):
    """Read and preprocess benchmark data for a given benchmark name."""
    data = pd.read_csv(f"data-{name}.csv")

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

    data["h"] = data["h"].map(cleanup_name)
    data = data[~(data["h"] == "FphMetaSet")]
    data = data[~(data["h"] == "MapEmbed")]
    data = data[~(data["h"] == "Hd8Set")]
    data = data[~(data["h"] == "U64HashSet")]
    data = data[~(data["metric"] == "prefetch")]

    queries = ["q01", "q50", "q99"]
    group_columns = [
        "h",
        "pf",
        "threads",
        "metric",
        "n",
        "alpha",
        "label",
        "kphf_target_bits_per_key",
    ]

    data["label"] = data.apply(
        lambda row: row["h"] + " (" + str(row["alpha"]) + "x)", axis=1
    )
    data = data.groupby(group_columns, as_index=False, sort=False)[
        ["build", *queries]
    ].min()

    return data


metric_name = {
    "loop": "for loop",
    "prefetch2": "loop with prefetching",
}

cpu_names = {
    "laptop": "Intel i7-10750H laptop",
    "diffie": "AMD EPYC Zen 4 9684X server",
    "floyd": "Intel Xeon Gold 6530 server",
}

queries = ["q01", "q50", "q99"]

label_color = {
    "HashSet": "red",
    "CuckooSet<Lazy>": "orange",
    "CuckooSet<Eager>": "brown",
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


def width(name, alpha):
    if name != "kPhfSet<kPtrHash>":
        return 1.5
    return {
        0.7: 1.5,
        0.9: 1.1,
        0.95: 0.8,
    }[alpha]


def build_legend(fig, axes, bbox_y_anchor):
    """Build and add a custom legend to the figure with category headers."""
    handles, labels = axes[0][0].get_legend_handles_labels()
    col_titles = ["Probing", "1-PHF-set", "$k$-PHF-set", "Lower bound"]
    new_handles, new_labels = [], []
    for i, (h, l) in enumerate(zip(handles, labels)):
        if i % 3 == 0:
            new_handles.append(mpatches.Patch(visible=False))
            new_labels.append(col_titles[i // 3])
        new_handles.append(h)
        new_labels.append(l)
    while len(new_handles) % 4 != 0:
        new_handles.append(mpatches.Patch(visible=False))
        new_labels.append("")
    leg = fig.legend(
        new_handles,
        new_labels,
        loc="lower center",
        ncols=len(col_titles),
        fontsize=10,
        bbox_to_anchor=(0.5, bbox_y_anchor),
    )
    for i, text in enumerate(leg.get_texts()):
        if i % 4 == 0:
            text.set_fontweight("bold")


titles = ["p=0.01", "p=0.50", "p=0.99"]
names = sys.argv[1:]

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
                    label=get_display_name(h)
                    + (
                        f" ($\\alpha={alpha}$)"
                        if alpha != 0.5
                        else " ($0.44\\leq \\alpha \\leq 0.88$)"
                    ),
                )
            if row_idx == 0:
                ax.set_title(title)
            ax.set_xlabel("n" if row_idx == nrows - 1 else "")
            ax.set_xscale("log", base=2)
            ax.grid(True, which="both", ls="--", lw=0.5)
            ax.set_ylabel(f"{cpu_names[bname]}\nns / query" if ci == 0 else "")
            ax.set_ylim(0)
            ax.legend().remove() if ax.get_legend() else None

    build_legend(fig, axes, -0.05)

    fig.suptitle(f"StaticHashset<u64> query throughput (1 thread, loop with prefetch)")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    # fig.savefig(f"kphf-set.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"kphf-set.pdf", bbox_inches="tight")
    # fig.savefig(f"kphf-set.svg", bbox_inches="tight")


for name in names:
    data = read_data(name)
    cpu_name = cpu_names[name]
    max_threads = max(data["threads"].unique())

    print(name)
    plt.close()

    row_specs = [
        ("loop", 1),
        ("loop", max_threads),
        ("prefetch2", 1),
        ("prefetch2", max_threads),
    ]

    nrows = len(row_specs)
    ncols = len(queries)
    figsize = (4 * ncols, 3 * nrows)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharey="row",
        sharex=True,
        squeeze=False,
    )

    for ri, (metric, threads) in enumerate(row_specs):
        row_data = data[(data["threads"] == threads) & (data["metric"] == metric)]
        groups = row_data.groupby(["h", "alpha"], sort=False)
        for ci, (q, title) in enumerate(zip(queries, titles)):
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
                    y=q,
                    ax=ax,
                    estimator=None,
                    errorbar=None,
                    color=label_color[h],
                    lw=width(h, alpha),
                    label=get_display_name(h)
                    + (
                        f" ($\\alpha={alpha}$)"
                        if alpha != 0.5
                        else " ($0.44\\leq \\alpha \\leq 0.88$)"
                    ),
                )
            if ri == 0:
                ax.set_title(title)
            ax.set_xlabel("n" if ri == nrows - 1 else "")
            ax.set_xscale("log", base=2)
            ax.grid(True, which="both", ls="--", lw=0.5)
            ax.set_ylabel(
                f"{metric_name[metric]}\nthreads={threads}\nns / query"
                if ci == 0
                else ""
            )
            ax.set_ylim(0)
            ax.legend().remove() if ax.get_legend() else None

    build_legend(fig, axes, -0.02)

    fig.suptitle(f"Hashset<u64> query throughput ({cpu_name})")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    # fig.savefig(f"kphf-set-{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"kphf-set-{name}.pdf", bbox_inches="tight")
    # fig.savefig(f"kphf-set-{name}.svg", bbox_inches="tight")
