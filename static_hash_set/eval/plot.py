#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

w = 64
name = "new2"

data = pd.read_csv(f"out-{name}.csv")

data = data[data["metric"] == "prefetch"]

# data = data[data["h"].str.contains("KphfSet")]


data["label"] = data.apply(
    lambda row: row["h"] + " (" + str(row["alpha"]) + "x)", axis=1
)
print(data.columns)

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
data = data.groupby(group_columns, as_index=False)[["build", *queries]].min()

labels = data["label"].unique()
print(labels)
palette = sns.color_palette(n_colors=len(labels))
label_color = {
    "FxHashSet": "red",
    "U64HashSet": "green",
    "CuckooSet<Lazy>": "orange",
    "CuckooSet<Eager>": "pink",
    "KphfSet<Sort>": "lime",
    "KphfSet<SortBump>": "blue",
    "KphfSet<SortBumpGreedy>": "cyan",
}
label_lw = {
    0.5: 2,
    0.7: 1.75,
    0.8: 1.5,
    0.9: 1,
    0.95: 0.75,
    0.99: 0.5,
}

plt.close()

titles = ["p=0.01", "p=0.50", "p=0.99"]
# thread_counts = sorted(data["threads"].unique())
thread_counts = data["threads"].unique()
target_latencies = {
    1: 7.5,
    6: 2.5,
    12: 2.5,
}
sizes = [12 * 1024 * 1024]
cache_labels = ["L3  ", "  RAM"]

nrows = len(thread_counts)
ncols = len(queries)
fig, axes = plt.subplots(
    nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False, sharex=True
)

for ri, threads in enumerate(thread_counts):
    thread_data = data[data["threads"] == threads]
    groups = thread_data.groupby(["h", "alpha", "kphf_target_bits_per_key"])
    for ci, (q, title) in enumerate(zip(queries, titles)):
        ax = axes[ri][ci]

        for (h, alpha, bpk), subset in groups:
            sns.lineplot(
                data=subset,
                x="n",
                y=q,
                ax=ax,
                estimator=None,
                errorbar=None,
                color=label_color[h],
                lw=label_lw[alpha],
                label=h + " " + str(alpha),
            )
        if ri == 0:
            ax.set_title(title)

        ax.set_xlabel("n" if ri == nrows - 1 else "")
        ax.set_xscale("log", base=2)

        ax.grid(True, which="both", ls="--", lw=0.5)

        ax.set_ylabel(f"threads={threads}\nns / query" if ci == 0 else "")
        ax.set_ylim(0, 30 / min(threads, 6))
        ax.axhline(y=target_latencies[threads], color="red", lw=1, ls="--", zorder=0)

        for s, l in zip(sizes, cache_labels):
            ax.axvline(x=s / 4, c="black", lw=1, ls="--")
            ax.text(s / 4, ax.get_ylim()[0], l, ha="right", va="bottom", fontsize=8)
        ax.text(
            sizes[-1] / 4,
            ax.get_ylim()[0],
            cache_labels[-1],
            ha="left",
            va="bottom",
            fontsize=8,
        )
        if ri == 0 and ci == ncols - 1:
            ax.legend(loc="upper left", fontsize=8)
        else:
            ax.legend().remove() if ax.get_legend() else None

fig.suptitle(f"u{w} hashset query throughput")
fig.tight_layout()
fig.savefig(f"plot-{name}.png", bbox_inches="tight", dpi=300)
