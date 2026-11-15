#!/usr/bin/env python3
# Run ./plot.py benchmark.csv
# Output: kphf-set-benchmark.html (self-contained interactive Plotly figure)

import pandas as pd
import math
from html import escape
import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    if "max_primary_bucket_size" not in data.columns:
        data["max_primary_bucket_size"] = float("nan")
    if "negative_filter_pass_ratio" not in data.columns:
        data["negative_filter_pass_ratio"] = float("nan")

    data["h"] = data["h"].map(cleanup_name)
    data = data[~data["h"].str.contains("AllKeys", regex=False)]
    data = data[~data["h"].str.contains(r"^CuckooQuotientSet<[^>]*,EF(?:,|>)")]
    hashbrown = data["h"] == "HashbrownSet"
    # For these tables capacity() is 7/8 of the allocated slots. The benchmark
    # CSV reports len()/capacity(), while the legend reports physical occupancy.
    data.loc[hashbrown & (data["n"] >= 8), "load_factor"] *= 7 / 8
    # Hashbrown ignores the requested alpha and uses its own capacity policy.
    # Collapse duplicate requested-alpha traces into its single measured baseline.
    data.loc[hashbrown, "alpha"] = 0.5
    data = data[~(data["h"] == "FphMetaSet")]
    data = data[~(data["h"] == "MapEmbed")]
    data = data[~(data["h"] == "Hd8Set")]
    data = data[~(data["h"] == "U64HashSet")]
    data = data[~(data["h"] == "TPHT")]
    data = data[~(data["metric"] == "prefetch")]
    data = data[~data["metric"].isin(["batch8", "batch16", "batch32"])]

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

    aggregations = {
        column: "min"
        for column in ["build", "load_factor", "memory_ratio", "primary_hit_ratio", *queries]
    }
    aggregations["max_primary_bucket_size"] = "max"
    aggregations["negative_filter_pass_ratio"] = "mean"
    data = data.groupby(group_columns, as_index=False, sort=False).agg(aggregations)

    # Prefetching can help or hurt. Show the better measured time at each point.
    lookup_columns = ["h", "pf", "threads", "n", "alpha", "kphf_target_bits_per_key"]
    loop_costs = {
        tuple(row[column] for column in lookup_columns): row[queries].to_numpy()
        for _, row in data[data["metric"] == "loop"].iterrows()
    }
    for index, row in data[data["metric"] == "prefetch2"].iterrows():
        key = tuple(row[column] for column in lookup_columns)
        if key in loop_costs:
            data.loc[index, queries] = row[queries].where(
                row[queries] < loop_costs[key], loop_costs[key]
            )

    return data


metric_name = {
    "loop": "for loop",
    "prefetch2": "best of prefetching and loop",
    "batch8": "batch of 8",
    "batch16": "batch of 16",
    "batch32": "batch of 32",
}

batch_dash = {"batch8": "dash", "batch16": "dot", "batch32": "dashdot"}

cpu_names = {
    "laptop": "Intel i7-10750H laptop",
    "hellman": "AMD EPYC Zen 4 9684X server",
    "floyd": "Intel Xeon Gold 6530 server",
}

queries = ["q01", "q50", "q99"]

label_color = {
    "HashSet": "red",
    "CuckooSet<Lazy>": "orange",
    "CuckooSet<Eager>": "brown",
    "GrowableRightCuckooSet": "darkmagenta",
    "FilteredCuckooSet<Lazy>": "forestgreen",
    "InlineTwoHotCuckooSet<Lazy,BumpedOnly>": "steelblue",
    "InlineOneHotCuckooSet<Lazy,BumpedOnly>": "crimson",
    "InlineOneHotCuckooSet<Lazy,BumpedOnly> v2": "#9B00FF",
    "InlineOneHotCuckooSet<Lazy,BumpedOnly,Branchless>": "mediumpurple",
    "SpillCuckooSet<Lazy,BeforePrimary,u8>": "purple",
    "SpillCuckooSet<Lazy,AfterMiss,u8>": "teal",
    "SpillCuckooSet<Lazy,BeforePrimary,u16>": "darkviolet",
    "SpillCuckooSet<Lazy,AfterMiss,u16>": "darkcyan",
    "HashbrownSet": "red",
    "CuckooQuotientSet<Lazy>": "teal",
    "CuckooQuotientSet<Eager>": "navy",
    "CuckooQuotientSet<Lazy,8,SIMD>": "lightseagreen",
    "CuckooQuotientSet<Eager,8,SIMD>": "royalblue",
    "CuckooQuotientSet<Lazy,16,SIMD>": "teal",
    "CuckooQuotientSet<Eager,16,SIMD>": "navy",
    "CuckooQuotientSet<Lazy,32,SIMD>": "darkgreen",
    "CuckooQuotientSet<Lazy,32,SIMD,OneHot>": "#00A6A6",
    "CuckooQuotientSet<Lazy,32,SIMD,OneHot> v2": "#7B1FA2",
    "CuckooQuotientSet<Lazy,32,SIMD,OneHot> v3": "#E67E00",
    "CuckooQuotientSet<Lazy,32,SIMD,OneHot> v4": "#4169E1",
    "CuckooQuotientSet<Eager,32,SIMD>": "steelblue",
    "CuckooQuotientSet<Lazy,8,SIMD,EF>": "#00FFFF",
    "CuckooQuotientSet<Eager,8,SIMD,EF>": "#0080FF",
    "CuckooQuotientSet<Lazy,16,SIMD,EF>": "#00FF66",
    "CuckooQuotientSet<Eager,16,SIMD,EF>": "#CC00FF",
    "CuckooQuotientSet<Lazy,32,SIMD,EF>": "#FFAA00",
    "CuckooQuotientSet<Lazy,32,SIMD,EF,OneHot>": "#8A2BE2",
    "CuckooQuotientSet<Eager,32,SIMD,EF>": "#FF1744",
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

variant_load_colors = {
    "CuckooQuotientSet<Lazy,32,SIMD>": {0.9: "#E000A0"},
    "CuckooSet<Eager>": {0.7: "#0057FF", 0.9: "#5C91FF"},
    "CuckooSet<Lazy>": {0.7: "#FFA500", 0.9: "#FFC266"},
    "GrowableRightCuckooSet": {0.7: "#9C179E", 0.9: "#D18AD4"},
    "FilteredCuckooSet<Lazy>": {0.7: "#009E73", 0.9: "#66C2A5"},
    "InlineTwoHotCuckooSet<Lazy,BumpedOnly>": {0.7: "#1F78B4", 0.9: "#80B1D3"},
    "InlineOneHotCuckooSet<Lazy,BumpedOnly>": {0.7: "#D81B60", 0.9: "#F48FB1"},
    "InlineOneHotCuckooSet<Lazy,BumpedOnly> v2": {0.7: "#9B00FF", 0.9: "#DC72FF"},
    "InlineOneHotCuckooSet<Lazy,BumpedOnly,Branchless>": {0.7: "#F50057", 0.9: "#FF6090"},
    "SpillCuckooSet<Lazy,BeforePrimary,u8>": {0.7: "#6b21a8", 0.9: "#c4b5fd"},
    "SpillCuckooSet<Lazy,AfterMiss,u8>": {0.7: "#0f766e", 0.9: "#5eead4"},
    "SpillCuckooSet<Lazy,BeforePrimary,u16>": {0.7: "#4c1d95", 0.9: "#ddd6fe"},
    "SpillCuckooSet<Lazy,AfterMiss,u16>": {0.7: "#115e59", 0.9: "#99f6e4"},
    "HashbrownSet": {0.7: "#FF0000", 0.9: "#FF0000"},
    "PhfSet<PtrHash>": {0.7: "#FFD600", 0.9: "#FFEB66"},
    "kPhfSet<kPtrHash>": {0.7: "#0000FF", 0.9: "#0000FF", 0.95: "#0000FF"},
    "MockHashSet": {0.7: "#000000", 0.9: "#000000", 1.0: "#000000"},
}


def series_color(name, alpha):
    shades = variant_load_colors.get(name, {})
    return shades.get(round(alpha, 2), label_color[name])

display_name = {
    "CuckooSet<Eager>": "Cuckoo eager",
    "CuckooSet<Lazy>": "Cuckoo lazy",
    "GrowableRightCuckooSet": "Growable rightward cuckoo",
    "HashbrownSet": "Hashbrown",
    "FilteredCuckooSet<Lazy>": "CuckooSet<Lazy> (inline spill filter)",
    "InlineTwoHotCuckooSet<Lazy,BumpedOnly>": "2-hot, bumped",
    "InlineOneHotCuckooSet<Lazy,BumpedOnly>": "1-hot bumped",
    "InlineOneHotCuckooSet<Lazy,BumpedOnly> v2": "1-hot bumped v2",
    "InlineOneHotCuckooSet<Lazy,BumpedOnly,Branchless>": "1-hot bumped, branchless",
    "SpillCuckooSet<Lazy,BeforePrimary,u8>": "Spill u8: before primary",
    "SpillCuckooSet<Lazy,AfterMiss,u8>": "Spill u8: after primary miss",
    "SpillCuckooSet<Lazy,BeforePrimary,u16>": "Spill u16: before primary",
    "SpillCuckooSet<Lazy,AfterMiss,u16>": "Spill u16: after primary miss",
    "CuckooQuotientSet<Lazy,8,SIMD>": "Quotient<Lazy, 8-bit SIMD>",
    "CuckooQuotientSet<Eager,8,SIMD>": "Quotient<Eager, 8-bit SIMD>",
    "CuckooQuotientSet<Lazy,16,SIMD>": "Quotient<Lazy, 16-bit SIMD>",
    "CuckooQuotientSet<Eager,16,SIMD>": "Quotient<Eager, 16-bit SIMD>",
    "CuckooQuotientSet<Lazy,32,SIMD>": "Quotient<Lazy, 32-bit SIMD>",
    "CuckooQuotientSet<Lazy,32,SIMD,OneHot>": "Quotient 32 lazy + 1-hot",
    "CuckooQuotientSet<Lazy,32,SIMD,OneHot> v2": "Quotient 32 lazy + 1-hot v2",
    "CuckooQuotientSet<Lazy,32,SIMD,OneHot> v3": "Quotient 32 lazy + 1-hot v3",
    "CuckooQuotientSet<Lazy,32,SIMD,OneHot> v4": "Quotient 32 lazy + 1-hot v4 (8 slots)",
    "CuckooQuotientSet<Eager,32,SIMD>": "Quotient<Eager, 32-bit SIMD>",
    "CuckooQuotientSet<Lazy,8,SIMD,EF>": "Quotient 8 lazy + EF",
    "CuckooQuotientSet<Eager,8,SIMD,EF>": "Quotient 8 eager + EF",
    "CuckooQuotientSet<Lazy,16,SIMD,EF>": "Quotient 16 lazy + EF",
    "CuckooQuotientSet<Eager,16,SIMD,EF>": "Quotient 16 eager + EF",
    "CuckooQuotientSet<Lazy,32,SIMD,EF>": "Quotient 32 lazy + EF",
    "CuckooQuotientSet<Lazy,32,SIMD,EF,OneHot>": "Quotient 32 lazy + EF + 1-hot",
    "CuckooQuotientSet<Eager,32,SIMD,EF>": "Quotient 32 eager + EF",
    "FphDynSet": "FPH",
    "PhfSet<PHast+>": "PHF-set<PHast+>",
    "PhfSet<PtrHash>": "PtrHash",
    "kPhfSet<kPtrHash>": "k-PtrHash",
    "MockHashSet": "Mock",
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


def alpha_marker(alpha):
    """Distinguish the lower and higher load-factor series in each color."""
    return "o" if alpha < 0.8 else "s"


def plotly_dash(name):
    return "solid"


def interactive_label(name, alpha, subset):
    def compact_range(column, digits=2):
        values = subset[column].dropna()
        if values.empty:
            return None
        lo, hi = values.min(), values.max()
        if round(lo, digits) == round(hi, digits):
            return f"{lo:.{digits}f}"
        return f"{lo:.{digits}f}–{hi:.{digits}f}"

    label = get_display_name(name)
    measured_alpha = f"{alpha:.2f}" if name == "MockHashSet" else compact_range("load_factor")
    if measured_alpha is not None:
        label += f" · α{measured_alpha.lstrip('0')}"
    space_ratio = compact_range("memory_ratio")
    if space_ratio is not None:
        label += f" · space {space_ratio}×"
    primary = subset["primary_hit_ratio"].dropna() * 100
    if "Cuckoo" in name and not primary.empty:
        lo, hi = primary.min(), primary.max()
        hit_rate = f"{lo:.0f}%" if round(lo) == round(hi) else f"{lo:.0f}–{hi:.0f}%"
        label += f" · hit {hit_rate}"
    pass_rates = subset["negative_filter_pass_ratio"].dropna() * 100
    if not pass_rates.empty:
        lo, hi = pass_rates.min(), pass_rates.max()
        pass_rate = f"{lo:.1f}%" if round(lo, 1) == round(hi, 1) else f"{lo:.1f}–{hi:.1f}%"
        label += f" · neg pass {pass_rate}"
    return label


def write_interactive_plot(data, name, cpu_name, thread_specs, metric_specs, output_path=None):
    bumped = data["h"].str.contains("BumpedOnly", regex=False)
    inline_one_hot = data["h"] == "InlineOneHotCuckooSet<Lazy,BumpedOnly>"
    data = data[~bumped | inline_one_hot]
    data = data[~data["h"].str.contains("<Eager", regex=False)]
    data = data[data["h"] != "CuckooSet<Lazy>"]
    quotienting = data["h"].str.startswith("CuckooQuotientSet<")
    lazy_32_one_hot = data["h"].isin([
        "CuckooQuotientSet<Lazy,32,SIMD,OneHot>",
        "CuckooQuotientSet<Lazy,32,SIMD,OneHot> v2",
        "CuckooQuotientSet<Lazy,32,SIMD,OneHot> v3",
        "CuckooQuotientSet<Lazy,32,SIMD,OneHot> v4",
    ])
    data = data[~quotienting | lazy_32_one_hot]
    panel_titles = [title for _ in metric_specs for _ in thread_specs for title in titles]
    nrows = len(thread_specs) * len(metric_specs)
    fig = make_subplots(
        rows=nrows,
        cols=len(queries),
        shared_xaxes=True,
        shared_yaxes="rows",
        subplot_titles=panel_titles,
        vertical_spacing=min(0.045, 0.18 / max(nrows, 1)),
        horizontal_spacing=0.015,
    )
    all_groups = data.groupby(["h", "alpha"], sort=False)
    group_items = sorted(
        all_groups,
        key=lambda item: (
            list(label_color).index(item[0][0]) if item[0][0] in label_color else len(label_color),
            1 - item[0][1],
        ),
    )
    x_ticks = sorted(data["n"].unique())

    row = 0
    for metric in metric_specs:
        for thread in thread_specs:
            row += 1
            trace_metrics = [metric, *batch_dash] if metric == "prefetch2" else [metric]
            row_data = data[
                (data["threads"] == thread) & data["metric"].isin(trace_metrics)
            ]
            visible_row_data = row_data[~row_data["h"].str.startswith("CuckooQuotientSet<")]
            row_y_max = max(float(visible_row_data[query].max()) for query in queries) * 1.05
            for col in range(1, len(queries) + 1):
                fig.update_yaxes(range=[0, row_y_max], autorange=False, fixedrange=True, row=row, col=col)
            for col, query in enumerate(queries, start=1):
                for (h, alpha), _ in group_items:
                    for trace_metric in trace_metrics:
                        subset = row_data[
                            (row_data["h"] == h)
                            & (row_data["alpha"] == alpha)
                            & (row_data["metric"] == trace_metric)
                        ].sort_values("n")
                        if subset.empty:
                            continue
                        hover_data = subset[
                            ["load_factor", "memory_ratio", "primary_hit_ratio"]
                        ].copy()
                        if h == "MockHashSet":
                            hover_data["load_factor"] = alpha
                        batch_label = (
                            f" · {metric_name[trace_metric]}"
                            if trace_metric in batch_dash
                            else ""
                        )
                        hovertemplate = (
                            f"Method: {escape(get_display_name(h))}{batch_label}"
                            "<br>n=%{x:,}<br>%{y:.2f} ns/query"
                        )
                        if hover_data["load_factor"].notna().any():
                            hovertemplate += "<br>α=%{customdata[0]:.2f}"
                        if hover_data["memory_ratio"].notna().any():
                            hovertemplate += "<br>space=%{customdata[1]:.2f}×"
                        if "Cuckoo" in h and hover_data["primary_hit_ratio"].notna().any():
                            hovertemplate += "<br>hit=%{customdata[2]:.0%}"
                        hovertemplate += "<extra></extra>"
                        batch_legend = trace_metric in batch_dash and thread == thread_specs[0]
                        fig.add_trace(
                            go.Scattergl(
                                x=subset["n"],
                                y=subset[query],
                                customdata=hover_data.to_numpy(),
                                mode="lines",
                                visible=("legendonly" if h.startswith("CuckooQuotientSet<") else True),
                                name=interactive_label(h, alpha, subset) + batch_label,
                                legendgroup=(
                                    f"{h}|{alpha}|{trace_metric}"
                                    if batch_label
                                    else f"{h}|{alpha}"
                                ),
                                showlegend=bool(col == 1 and (row == 1 or batch_legend)),
                                line={
                                    "color": series_color(h, alpha),
                                    "width": width(h, alpha),
                                    "dash": batch_dash.get(trace_metric, plotly_dash(h)),
                                },
                                hovertemplate=hovertemplate,
                            ),
                            row=row,
                            col=col,
                        )
                yaxis = "yaxis" if row == 1 and col == 1 else f"yaxis{(row - 1) * len(queries) + col}"
                if col == 1:
                    thread_label = "1 thread" if thread == 1 else f"{thread} threads"
                    fig.layout[yaxis].update(title_text=f"{thread_label} · {metric_name.get(metric, metric)}<br>ns / query", rangemode="tozero")
                else:
                    fig.layout[yaxis].update(rangemode="tozero")

    # The benchmark sizes double at each step. Explicit log-space increments
    # keep every subplot on the same 1M, 2M, 4M, ... axis.
    fig.update_xaxes(
        type="log",
        range=[math.log10(x_ticks[0]), math.log10(x_ticks[-1])],
        autorange=False,
        fixedrange=True,
        tickmode="linear",
        tick0=math.log10(x_ticks[0]),
        dtick=math.log10(2),
        tickformat=".0s",
        showline=True,
        linewidth=1,
        linecolor="#000000",
        mirror=True,
    )
    fig.update_yaxes(
        fixedrange=True,
        showline=True,
        linewidth=1,
        linecolor="#000000",
        mirror=True,
    )
    for col in range(1, len(queries) + 1):
        fig.update_xaxes(title_text="set size n", row=nrows, col=col)

    height = max(680, 230 * nrows + 180)
    fig.update_layout(
        title=f"u64 lookup cost ({cpu_name})",
        template="plotly_white",
        width=1125,
        height=height,
        font={"size": 11},
        hovermode="closest",
        uirevision="latest",
        legend={"title": "Variant", "font": {"size": 10}, "groupclick": "togglegroup", "y": -0.13, "x": 0.5, "xanchor": "center", "yanchor": "top", "orientation": "h"},
        margin={"l": 82, "r": 25, "t": 75, "b": 175},
    )
    output = Path(output_path) if output_path else Path(f"kphf-set-{Path(name).stem}.html")
    fig.write_html(
        output,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
        config={"displaylogo": False, "scrollZoom": False},
    )
    print(output)


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


titles = [
    "1% positive",
    "50% positive",
    "99% positive",
]
names = sys.argv[1:]
cpu_override = None
output_override = None
if "--cpu" in names:
    idx = names.index("--cpu")
    cpu_override = names[idx + 1]
    del names[idx : idx + 2]
if "--output" in names:
    idx = names.index("--output")
    output_override = names[idx + 1]
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

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns

    plt.close()

    metric = "prefetch2"
    threads = 1

    nrows = 3  # One row for each benchmark: laptop, floyd, hellman
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
    benchmark_names = ["laptop", "floyd", "hellman"]
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
                    color=series_color(h, alpha),
                    lw=width(h, alpha),
                    linestyle=line_style(h),
                    marker=alpha_marker(alpha),
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
    # The 16-thread data remains in the CSV for later analysis, but is omitted
    # from the current interactive plot at the user's request.
    thread_specs = [thread for thread in sorted(data["threads"].unique()) if thread != 16] if mt else [1]
    metric_specs = [metric for metric in ["loop", "prefetch2"] if metric in data["metric"].unique()]
    if slides:
        thread_specs = [12 if mt else 1]
    write_interactive_plot(data, name, cpu_name, thread_specs, metric_specs, output_override)
