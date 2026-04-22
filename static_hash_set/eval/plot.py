#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import sys

w = 64

for name in sys.argv[1:]:
    data = pd.read_csv(f"data-{name}.csv")

    def cleanup_name(h):
        if "Function2" in h:
            return "PhfSet<PHast+minimal>"
        if "Perfect" in h:
            return "PhfSet<PHast+>"
        if "PtrHash" in h:
            return "PhfSet<PtrHash>"
        return h

    data["h"] = data["h"].map(cleanup_name)
    data = data[~(data["h"] == "FphMetaSet")]

    for metric, data in data.groupby("metric"):
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
            "CuckooSet<PrefetchOneLazy>": "orange",
            "CuckooSet<Eager>": "pink",
            "CuckooSet<PrefetchBothEager>": "pink",
            "KphfSet<Sort>": "lime",
            "KphfSet<SortBump>": "blue",
            "KphfSet<SortBumpGreedy>": "cyan",
            "FphDynSet": "purple",
            "FphMetaSet": "black",
            "PhfSet<PHast+>": "cyan",
            "PhfSet<PHast+minimal>": "darkred",
            "PhfSet<PtrHash>": "lime",
        }
        plt.close()

        titles = ["p=0.01", "p=0.50", "p=0.99"]
        thread_counts = data["threads"].unique()
        thread_counts = [1, max(thread_counts)]
        sizes = [12 * 1024 * 1024]
        cache_labels = ["L3  ", "  RAM"]

        nrows = len(thread_counts)
        ncols = len(queries)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey="row", sharex=True
        )

        for ri, threads in enumerate(thread_counts):
            thread_data = data[data["threads"] == threads]
            groups = thread_data.groupby(["h", "alpha"])
            for ci, (q, title) in enumerate(zip(queries, titles)):
                ax = axes[ri][ci]

                for (h, alpha), subset in groups:
                    sns.lineplot(
                        data=subset,
                        x="n",
                        y=q,
                        ax=ax,
                        estimator=None,
                        errorbar=None,
                        color=label_color[h],
                        lw=1.5,
                        label=h + " " + str(alpha),
                    )
                if ri == 0:
                    ax.set_title(title)

                ax.set_xlabel("n" if ri == nrows - 1 else "")
                ax.set_xscale("log", base=2)

                ax.grid(True, which="both", ls="--", lw=0.5)

                ax.set_ylabel(f"threads={threads}\nns / query" if ci == 0 else "")
                ax.set_ylim(0)

                if ri == 0 and ci == ncols - 1:
                    ax.legend(loc="upper left", fontsize=8)
                else:
                    ax.legend().remove() if ax.get_legend() else None

        fig.suptitle(f"{name}: u{w} hashset query throughput ({metric})")
        fig.tight_layout()
        fig.savefig(f"plot-{name}-{metric}.png", bbox_inches="tight", dpi=300)
        fig.savefig(f"plot-{name}-{metric}.pdf", bbox_inches="tight")
        fig.savefig(f"plot-{name}-{metric}.svg", bbox_inches="tight")
