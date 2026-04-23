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

    modes = ["all"]
    if name == "laptop":
        modes = ["single", "all"]

    for mode in modes:
        for metric, data in data.groupby("metric"):
            if mode == "single" and metric != "prefetch2":
                continue
            print(name, mode, metric)

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

            labels = data["label"].unique()
            palette = sns.color_palette(n_colors=len(labels))
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
            plt.close()

            def width(name, alpha):
                if name != "kPhfSet<kPtrHash>":
                    return 1.5
                return {
                    0.7: 1.5,
                    0.9: 1.1,
                    0.95: 0.8,
                }[alpha]

            titles = ["p=0.01", "p=0.50", "p=0.99"]
            # thread_counts = data["threads"].unique()
            # thread_counts = [1, max(thread_counts)]
            sizes = [12 * 1024 * 1024]
            cache_labels = ["L3  ", "  RAM"]

            thread_counts = data["threads"].unique()
            thread_counts = [1] if mode == "single" else [1, max(thread_counts)]
            nrows = len(thread_counts)
            ncols = len(queries)
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(5 * ncols, 4 * nrows),
                sharey="row",
                sharex=True,
                squeeze=False,
            )

            for ri, threads in enumerate(thread_counts):
                thread_data = data[data["threads"] == threads]
                groups = thread_data.groupby(["h", "alpha"], sort=False)
                for ci, (q, title) in enumerate(zip(queries, titles)):
                    ax = axes[ri][ci]

                    for (h, alpha), subset in sorted(
                        groups,
                        key=lambda x: (
                            list(label_color).index(x[0][0])
                            if x[0][0] in label_color
                            else len(label_color)
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
                            label=h + " " + str(alpha),
                        )
                    if ri == 0:
                        ax.set_title(title)

                    ax.set_xlabel("n" if ri == nrows - 1 else "")
                    ax.set_xscale("log", base=2)

                    ax.grid(True, which="both", ls="--", lw=0.5)

                    if mode == "single":
                        ax.set_ylabel(f"ns / query" if ci == 0 else "")
                    else:
                        ax.set_ylabel(
                            f"threads={threads}\nns / query" if ci == 0 else ""
                        )
                    ax.set_ylim(0)

                    if ri == 0 and ci == 0:
                        ax.legend(loc="upper left", fontsize=8)
                    else:
                        ax.legend().remove() if ax.get_legend() else None

            metric_name = {
                "loop": "for loop",
                "prefetch": "prefetch (old)",
                "prefetch2": "loop with prefetch",
            }

            cpu_name = {
                "laptop": "Intel i7-10750H laptop",
                "diffie": "AMD EPYC Zen 4 9684X server",
                "floyd": "Intel Xeon Gold 6530 server",
            }[name]

            if mode == "single":
                fig.suptitle(
                    f"StaticHashset<u64> query throughput ({cpu_name}, 1 thread, loop with prefetch)"
                )
                fig.tight_layout()
                fig.savefig(f"plot.png", bbox_inches="tight", dpi=300)
                fig.savefig(f"plot.pdf", bbox_inches="tight")
                fig.savefig(f"plot.svg", bbox_inches="tight")
            else:
                fig.suptitle(
                    f"Hashset<u64> query throughput ({cpu_name}, {metric_name[metric]})"
                )
                fig.tight_layout()
                fig.savefig(f"plot-{name}-{metric}.png", bbox_inches="tight", dpi=300)
                fig.savefig(f"plot-{name}-{metric}.pdf", bbox_inches="tight")
                fig.savefig(f"plot-{name}-{metric}.svg", bbox_inches="tight")
