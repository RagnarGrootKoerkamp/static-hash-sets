#!/usr/bin/env python3

# I want the following plots:
# A)
# x-ax: size (bits per key)
# y-ax (left): build throughput (ns/key)
# y-ax (right): query throughput (ns/key)
#
# plots:
# - separate subplots for k=8 and k=16
# - each algorithm has it's own colour
# - alpha corresponds to thickness of the line
# - Plot the lower bound in black.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

df = pd.read_csv("out2.csv")
df = df[df["actual_bpk"] != "actual_bpk"]  # drop duplicate header rows
numeric_cols = [
    "k",
    "n",
    "alpha",
    "lb",
    "bits_per_key",
    "actual_bpk",
    "pct_bumped",
    "build_ns",
    "throughput_ns",
    "latency_ns",
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
df = df[df["actual_bpk"] > 0]  # drop failed builds
df = df[df["n"] == 3_000_000]  # only largest n
df = df[df["alpha"] <= 0.98]

# Average over different n values for the same (mode, k, alpha, bits_per_key)
df = df.groupby(["mode", "k", "alpha", "bits_per_key"], as_index=False).agg(
    {
        "actual_bpk": "mean",
        "build_ns": "mean",
        "throughput_ns": "mean",
        "pct_bumped": "mean",
        "lb": "first",
    }
)

modes = sorted(df["mode"].unique())
colors = dict(zip(modes, plt.cm.tab10(np.linspace(0, 0.8, len(modes)))))

alpha_vals = sorted(df["alpha"].unique())
lw_min, lw_max = 0.6, 3.0
lw = lambda a: lw_min + (lw_max - lw_min) * (a - alpha_vals[0]) / (
    alpha_vals[-1] - alpha_vals[0]
)

k_vals = sorted(df["k"].unique())
metrics = [
    ("build_ns", "build time (ns/key)"),
    ("throughput_ns", "query throughput (ns/key)"),
    ("pct_bumped", "% bumped keys"),
]
ncols = len(metrics)
nrows = len(k_vals)
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

for row, k in enumerate(k_vals):
    subset = df[df["k"] == k]

    for col, (metric, ylabel) in enumerate(metrics):
        ax = axes[row][col]

        if subset.empty:
            ax.set_title(f"k={k} (no data)")
            ax.set_visible(False)
            continue

        for mode in modes:
            for alpha in alpha_vals:
                grp = subset[(subset["mode"] == mode) & (subset["alpha"] == alpha)]
                if grp.empty:
                    continue
                ax.plot(
                    grp["actual_bpk"], grp[metric], color=colors[mode], lw=lw(alpha)
                )

        # Lower bound verticals
        for alpha in alpha_vals:
            grp = subset[subset["alpha"] == alpha]
            if grp.empty:
                continue
            ax.axvline(grp["lb"].iloc[0], color="black", lw=lw(alpha), ls=":")

        ax.set_xlabel("bits per key")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_ylim(bottom=0)
        ax.set_title(f"k={k} — {ylabel}")

        # Lower bound values and their alphas, sorted by lb
        lb_alpha = subset.groupby("alpha")["lb"].first().reset_index().sort_values("lb")
        lb_vals = lb_alpha["lb"].tolist()
        lb_alphas = lb_alpha["alpha"].tolist()

        # Bottom axis: pow2 ticks (labeled) + lb ticks (unlabeled)
        xmin, xmax = ax.get_xlim()
        i_min = int(np.floor(np.log2(xmin)))
        i_max = int(np.ceil(np.log2(xmax)))
        pow2_ticks = [2.0**i for i in range(i_min, i_max + 1) if xmin <= 2.0**i <= xmax]
        pow2_set = set(pow2_ticks)
        all_ticks = sorted(pow2_set | set(lb_vals))
        ax.set_xticks(all_ticks)
        ax.set_xticklabels([f"{t:g}" if t in pow2_set else "" for t in all_ticks])

        # Top axis: alpha labels at lb positions
        ax_top = ax.twiny()
        ax_top.set_xscale("log")
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(lb_vals)
        ax_top.set_xticklabels([f"α={a:.2g}" for a in lb_alphas], fontsize=7)

        # Legend (only on first column)
        if col == 0:
            algo_handles = [
                mlines.Line2D([], [], color=colors[m], lw=2, label=m) for m in modes
            ]
            lb_handle = mlines.Line2D(
                [], [], color="black", lw=1.5, ls=":", label="lower bound"
            )
            ax.legend(handles=algo_handles + [lb_handle], fontsize=7, loc="upper left")

plt.suptitle("KptrHash: build time and query throughput vs. space")
plt.tight_layout()
# plt.savefig("plot.pdf", bbox_inches="tight")
plt.savefig("plot.png", dpi=150, bbox_inches="tight")
# print("Saved plot.pdf and plot.png")
