#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv("out.csv")


def make_category(row):
    if row["h"] == "FxHashSet":
        return "1.4-2.8"
    return str(round(row["overhead"], 1))


data["target_overhead"] = data.apply(make_category, axis=1)
data["label"] = data.apply(
    lambda row: row["h"] + " (" + row["target_overhead"] + "x)", axis=1
)

labels = data["label"].unique()
palette = sns.color_palette(n_colors=len(labels))
label_color = {
    "FxHashSet (1.4-2.8x)": "red",
    "U64HashSet (1.4x)": "green",
    "U64HashSet (1.3x)": "green",
    "U64HashSet (1.2x)": "green",
    "U64HashSet (1.1x)": "green",
    "StaticHashSet (1.4x)": "blue",
    "StaticHashSet (1.2x)": "blue",
    "StaticHashSet (1.1x)": "blue",
}
label_lw = {
    "FxHashSet (1.4-2.8x)": 2.5,
    "U64HashSet (1.4x)": 2,
    "U64HashSet (1.3x)": 1.75,
    "U64HashSet (1.2x)": 1.5,
    "U64HashSet (1.1x)": 1,
    "StaticHashSet (1.4x)": 2,
    "StaticHashSet (1.2x)": 1.5,
    "StaticHashSet (1.1x)": 1,
}

plt.close()

queries = ["q01", "q10", "q50", "q90", "q99"]
titles = ["p=0.01", "p=0.10", "p=0.50", "p=0.90", "p=0.99"]
thread_counts = sorted(data["threads"].unique())
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

fig.suptitle("u32 hashset query throughput")
fig.tight_layout()
fig.savefig("plot.png", bbox_inches="tight", dpi=300)
