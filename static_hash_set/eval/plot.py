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
        if "KptrHash" in h:
            return "kPhfSet<kPtrHash>"
        return h

    data["h"] = data["h"].map(cleanup_name)
    data = data[~(data["h"] == "FphMetaSet")]
    data = data[~(data["h"] == "MapEmbed")]
    data = data[~(data["h"] == "Hd8Set")]
    data = data[~(data["metric"] == "prefetch")]

    metric_name = {
        "loop": "for loop",
        "prefetch2": "loop with prefetch",
    }

    cpu_name = {
        "laptop": "Intel i7-10750H laptop",
        "diffie": "AMD EPYC Zen 4 9684X server",
        "floyd": "Intel Xeon Gold 6530 server",
    }[name]

    modes = ["all"]
    if name == "laptop":
        modes = ["single", "all"]

    data["label"] = data.apply(
        lambda row: row["h"] + " (" + str(row["alpha"]) + "x)", axis=1
    )

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
    data = data.groupby(group_columns, as_index=False, sort=False)[
        ["build", *queries]
    ].min()

    label_color = {
        "FxHashSet": "red",
        "U64HashSet": "magenta",
        "CuckooSet<Lazy>": "orange",
        "CuckooSet<Eager>": "brown",
        "FphDynSet": "pink",
        "PhfSet<PHast+>": "lime",
        "PhfSet<PtrHash>": "cyan",
        "kPhfSet<kPtrHash>": "blue",
        # "Hd8Set": "black",
        # "MapEmbed": "brown",
    }

    def width(name, alpha):
        if name != "kPhfSet<kPtrHash>":
            return 1.5
        return {
            0.7: 1.5,
            0.9: 1.1,
            0.95: 0.8,
        }[alpha]

    titles = ["p=0.01", "p=0.50", "p=0.99"]
    max_threads = max(data["threads"].unique())

    for mode in modes:
        print(name, mode)
        plt.close()

        if mode == "single":
            row_specs = [("prefetch2", 1)]
        else:
            row_specs = [
                ("loop", 1),
                ("loop", max_threads),
                ("prefetch2", 1),
                ("prefetch2", max_threads),
            ]

        if mode == "single":
            nrows = 1
            ncols = len(queries)
            figsize = (10, 7)
        else:
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

        if mode == "single":
            metric, threads = row_specs[0]
            row_data = data[(data["threads"] == threads) & (data["metric"] == metric)]
            groups = row_data.groupby(["h", "alpha"], sort=False)
            for ri, (q, title) in enumerate(zip(queries, titles)):
                ax = axes[0][ri]
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
                        label=h
                        + (
                            f" ($\\alpha={alpha}$)"
                            if alpha != 0.5
                            else " ($0.44\\leq \\alpha \\leq 0.88$)"
                        ),
                    )
                ax.set_title(title)
                ax.set_xlabel("n")
                ax.set_xscale("log", base=2)
                ax.grid(True, which="both", ls="--", lw=0.5)
                ax.set_ylabel("ns / query")
                ax.set_ylim(0)
                ax.legend().remove() if ax.get_legend() else None
        else:
            for ri, (metric, threads) in enumerate(row_specs):
                row_data = data[
                    (data["threads"] == threads) & (data["metric"] == metric)
                ]
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
                            label=h
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

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncols=3,
            fontsize=10,
            bbox_to_anchor=(0.5, -0.03) if mode == "single" else (0.5, -0.02),
        )

        if mode == "single":
            fig.suptitle(
                f"StaticHashset<u64> query throughput ({cpu_name}, 1 thread, loop with prefetch)"
            )
            fig.tight_layout(rect=[0, 0.08, 1, 1])
            # fig.savefig(f"kphf-set.png", bbox_inches="tight", dpi=300)
            fig.savefig(f"kphf-set.pdf", bbox_inches="tight")
            # fig.savefig(f"kphf-set.svg", bbox_inches="tight")
        else:
            fig.suptitle(f"Hashset<u64> query throughput ({cpu_name})")
            fig.tight_layout(rect=[0, 0.05, 1, 1])
            # fig.savefig(f"kphf-set-{name}.png", bbox_inches="tight", dpi=300)
            fig.savefig(f"kphf-set-{name}.pdf", bbox_inches="tight")
            # fig.savefig(f"kphf-set-{name}.svg", bbox_inches="tight")
