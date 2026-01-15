#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

with open("data/1.json") as f:
    data = pd.DataFrame(json.load(f))
data["h"] = data["h"].astype(str) + data["pf"].map({True: " (+pf)", False: ""})

plt.close()

# Plot 'n' vs 'build':
# plt.plot(data["n"], data["build"], label="Build", c="black", lw=0.6)
sns.lineplot(data=data, x="n", y="build", hue="h", lw=0.6)
sns.lineplot(data=data, x="n", y="q01", hue="h", ls="--", lw=1.0)
# sns.lineplot(data=data, x="n", y="q50", hue="h", lw=1.25)
sns.lineplot(data=data, x="n", y="q99", hue="h", lw=1)
plt.xlabel("n")
plt.ylabel("ns / element")
plt.ylim(0, 30)
plt.xscale("log", base=2)
# plt.yscale("log", base=2)
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend(loc="upper left")
plt.title("FxHashSet<u32> query throughput")

# Cache sizes: 32KiB, 256KiB, 12MiB
# axvline at each size
sizes = [32 * 1024, 256 * 1024, 12 * 1024 * 1024]
labels = ["L1  ", "L2  ", "L3  ", "  RAM"]
for s, l in zip(sizes, labels):
    plt.axvline(x=s / 4, c="black", lw=1, ls="--")
    plt.text(s / 4, plt.ylim()[0], l, ha="right", va="bottom", fontsize=16)
plt.text(sizes[-1] / 4, plt.ylim()[0], labels[-1], ha="left", va="bottom", fontsize=16)

plt.gcf().set_size_inches(10, 6)
plt.savefig("data/plot.png", bbox_inches="tight", dpi=300)
