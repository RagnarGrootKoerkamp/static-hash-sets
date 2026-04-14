#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

w = 64
name = "64"

data = pd.read_csv(f"out-{name}.csv")

data = data[data["metric"] >= "prefetch"]

# data = data[data["h"].str.contains("KphfSet")]


def make_category(row):
    if row["h"] == "FxHashSet":
        return "1.4-2.8"
    return str(math.floor(row["overhead"] * 10) / 10)


data["target_overhead"] = data.apply(make_category, axis=1)
data["label"] = data.apply(
    lambda row: row["h"] + " (" + row["target_overhead"] + "x)", axis=1
)

queries = ["q01", "q10", "q50", "q90", "q99"]
group_columns = ["h", "pf", "threads", "metric", "n", "target_overhead", "label"]
data = data.groupby(group_columns, as_index=False)[["build", *queries]].median()

labels = data["label"].unique()
print(labels)
palette = sns.color_palette(n_colors=len(labels))
label_color = {
    "FxHashSet (1.4-2.8x)": "red",
    "U64HashSet (1.4x)": "green",
    "U64HashSet (1.3x)": "green",
    "U64HashSet (1.2x)": "green",
    "U64HashSet (1.1x)": "green",
    "CuckooSet<PrefetchOneLazy> (1.4x)": "orange",
    "CuckooSet<PrefetchOneLazy> (1.2x)": "orange",
    "CuckooSet<PrefetchOneLazy> (1.1x)": "orange",
    "CuckooSet<PrefetchOneEager> (1.4x)": "white",
    "CuckooSet<PrefetchOneEager> (1.2x)": "white",
    "CuckooSet<PrefetchOneEager> (1.1x)": "white",
    "CuckooSet<PrefetchBoth> (1.4x)": "white",
    "CuckooSet<PrefetchBoth> (1.2x)": "white",
    "CuckooSet<PrefetchBoth> (1.1x)": "white",
    "KphfSet<Sort> (1.4x)": "blue",
    "KphfSet<Sort> (1.2x)": "blue",
    "KphfSet<Sort> (1.1x)": "blue",
    "KphfSet<SortBump> (1.4x)": "lightblue",
    "KphfSet<SortBump> (1.2x)": "lightblue",
    "KphfSet<SortBump> (1.1x)": "lightblue",
}
label_lw = {
    "FxHashSet (1.4-2.8x)": 2.5,
    "U64HashSet (1.4x)": 2,
    "U64HashSet (1.3x)": 1.75,
    "U64HashSet (1.2x)": 1.5,
    "U64HashSet (1.1x)": 1,
    "CuckooSet<PrefetchOneLazy> (1.4x)": 2,
    "CuckooSet<PrefetchOneLazy> (1.2x)": 1.5,
    "CuckooSet<PrefetchOneLazy> (1.1x)": 1,
    "CuckooSet<PrefetchOneEager> (1.4x)": 2,
    "CuckooSet<PrefetchOneEager> (1.2x)": 1.5,
    "CuckooSet<PrefetchOneEager> (1.1x)": 1,
    "CuckooSet<PrefetchBoth> (1.4x)": 2,
    "CuckooSet<PrefetchBoth> (1.2x)": 1.5,
    "CuckooSet<PrefetchBoth> (1.1x)": 1,
    "KphfSet<Sort> (1.4x)": 2,
    "KphfSet<Sort> (1.2x)": 1.5,
    "KphfSet<Sort> (1.1x)": 1,
    "KphfSet<SortBump> (1.4x)": 2,
    "KphfSet<SortBump> (1.2x)": 1.5,
    "KphfSet<SortBump> (1.1x)": 1,
}

plt.close()

titles = ["p=0.01", "p=0.10", "p=0.50", "p=0.90", "p=0.99"]
# thread_counts = sorted(data["threads"].unique())
thread_counts = [1, 6, 12]
sizes = [12 * 1024 * 1024]
cache_labels = ["L3  ", "  RAM"]

nrows = len(thread_counts)
ncols = len(queries)
fig, axes = plt.subplots(
    nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey="row", sharex=True
)

for ri, threads in enumerate(thread_counts):
    thread_data = data[data["threads"] == threads]
    for ci, (q, title) in enumerate(zip(queries, titles)):
        ax = axes[ri][ci]
        for label in labels:
            subset = thread_data[thread_data["label"] == label]
            sns.lineplot(
                data=subset,
                x="n",
                y=q,
                ax=ax,
                estimator=None,
                errorbar=None,
                color=label_color[label],
                lw=label_lw[label],
                label=label,
            )
        if ri == 0:
            ax.set_title(title)
        ax.set_xlabel("n" if ri == nrows - 1 else "")
        ax.set_ylabel(f"threads={threads}\nns / query" if ci == 0 else "")
        ax.set_ylim(0, 60 / threads)
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", ls="--", lw=0.5)
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
