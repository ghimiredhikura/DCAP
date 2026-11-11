"""Render manuscript Figures 6–8 and 10 from explicit result tables."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from .paths import ROOT, read

METRICS = [
    "average_precision",
    "dice",
    "small_object_recall",
    "boundary_f1",
    "balanced_brier_score",
    "joint_support",
]
LABELS = [
    "AP",
    "Dice",
    "Small-scar recall",
    "Boundary F1",
    "Balanced Brier",
    "Joint support",
]
NAMES = {
    "dcap": "DCAP",
    "taylor": "Group Taylor",
    "magnitude": "Group Magnitude",
    "compact": "Direct compact",
}
COLORS = {
    "dcap": "#00828b",
    "taylor": "#7554a3",
    "magnitude": "#c97135",
    "compact": "#9ba18a",
}




def deployment_results(source):
    """Rebuild Table 13 and Figure 10 from the recorded timing summaries."""
    records = read(Path(source) / "deployment.json")["measurements"]
    by = {(r["device"], r["model"], r["backend"], r["batch"]): r for r in records}
    table, figure = [], []
    for device in ["rtx4080", "orin"]:
        for backend in ["amp", "trt_graph", "fp32_graph"]:
            models = ["dense", "dcap70"]
            if backend == "fp32_graph":
                models += ["taylor70", "sirfp70", "segformer_b2"]
            for model in models:
                single, batch = [by[device, model, backend, b] for b in [1, 4]]
                memory = "trt_context_MiB" if backend == "trt_graph" else "allocated_MiB"
                table.append(dict(device=device, model=model, backend=backend,
                                  latency_ms=single["median_ms"],
                                  throughput_chips_s=4000/batch["median_ms"],
                                  memory_MiB=single[memory],
                                  reserved_MiB=single["reserved_MiB"] if backend == "fp32_graph" else ""))
        for metric, batch in [("latency", 1), ("throughput", 4)]:
            def block_values(model):
                values = np.asarray(by[device, model, "fp32_graph", batch]["block_medians_ms"])
                return values if batch == 1 else 4000/values
            denominator = float(np.median(block_values("dense")))
            for model in ["dense", "compact70", "magnitude70", "taylor70", "dcap70"]:
                values = block_values(model)/denominator
                figure.append(dict(device=device, model=model, metric=metric,
                                   normalized_median=float(np.median(values)),
                                   normalized_minimum=float(min(values)),
                                   normalized_maximum=float(max(values)),
                                   dense_block_median=denominator))
    return pd.DataFrame(table), pd.DataFrame(figure)


def deployment_figure(data):
    """Plot device-relative latency and throughput with timing-block ranges."""
    models = ["dense", "compact70", "magnitude70", "taylor70", "dcap70"]
    names = ["Dense reference", "Direct compact", "Group Magnitude", "Group Taylor", "DCAP"]
    colors = ["#414141", "#65704f", "#926b50", "#75628d", "#A61C2B"]
    markers = ["P", "s", "D", "^", "o"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "text.color": "#222222",
                         "axes.labelcolor": "#222222", "xtick.color": "#333333",
                         "ytick.color": "#222222"})
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.2), sharey=True)
    fig.subplots_adjust(left=.19, right=.98, top=.91, bottom=.13, wspace=.28, hspace=.48)
    throughput_max = data.loc[data.metric == "throughput", "normalized_maximum"].max()
    limits = {"latency": (.30, 1.12),
              "throughput": (.80, float(np.ceil((throughput_max + .25) * 5) / 5))}
    for i, (device, title) in enumerate([("rtx4080", "RTX 4080"), ("orin", "Jetson AGX Orin")]):
        for j, metric in enumerate(["latency", "throughput"]):
            ax = axes[i, j]
            ax.axhspan(3.58, 4.42, color="#A61C2B", alpha=.055, zorder=0)
            selected = data[(data.device == device) & (data.metric == metric)].set_index("model")
            for y, model in enumerate(models):
                row = selected.loc[model]
                center, lo, hi = row[["normalized_median", "normalized_minimum", "normalized_maximum"]]
                if not limits[metric][0] < lo <= center <= hi < limits[metric][1]:
                    raise ValueError(f"Timing range outside display limits: {device}/{model}/{metric}")
                ax.errorbar(center, y, xerr=[[center-lo], [hi-center]], fmt=markers[y],
                            color=colors[y], ms=5.5, capsize=3, lw=1.4, zorder=3)
                ax.annotate(f"{center:.2f}×", (hi, y), xytext=(7, 0), textcoords="offset points",
                            va="center", fontsize=9, color="#A61C2B" if model == "dcap70" else "#222222")
            ax.axvline(1, color="#666666", ls="--", lw=1, zorder=1)
            ax.set_yticks(range(5), names)
            ax.set_ylim(4.55, -.55)
            ax.set_title(f"({chr(97+2*i+j)}) {title}: {metric}", loc="left", pad=9)
            ax.set_xlim(*limits[metric])
            ax.xaxis.set_major_locator(MultipleLocator(.2))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.grid(axis="x", color="#dddddd", alpha=.65, lw=.6)
            ax.tick_params(axis="both", length=3, width=.6)
            for spine in ax.spines.values():
                spine.set_color("#888888")
                spine.set_linewidth(.7)
            for tick in ax.get_yticklabels():
                if tick.get_text() == "DCAP":
                    tick.set_color("#A61C2B")
            label = ("Latency relative to dense (×)  ← lower is better" if j == 0 else
                     "Throughput relative to dense (×)  higher is better →")
            ax.set_xlabel(label, fontsize=9, labelpad=7)
    return fig


def plot(numbers=(6, 7, 8), source=None, output=None):
    source = Path(source or ROOT / "results")
    output = Path(output or ROOT / "generated/figures")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    created = []

    def save(fig, n):
        for ext in ["pdf", "png"]:
            path = output / f"figure{n}.{ext}"
            fig.savefig(path, bbox_inches="tight", dpi=180)
            created.append(str(path))
        plt.close(fig)

    if 6 in numbers:
        data = pd.read_csv(source / "figure6.csv")
        dense = read(source / "representative.json")["dense_test"]
        colors = {**COLORS, "dcap": "#A61C2B", "taylor": "#9e95aa", "magnitude": "#af9b8c"}
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
        fig.subplots_adjust(hspace=0.38, wspace=0.30, bottom=0.16)
        for i, ax in enumerate(axes.flat):
            for method in NAMES:
                d = data[data.method == method].sort_values("target_percent")
                if len(d) != 5:
                    raise ValueError("Figure 6 requires five budgets for " + method)
                ax.plot(
                    d.mac_reduction_percent,
                    d[METRICS[i]],
                    "--" if method == "compact" else "-",
                    marker={
                        "dcap": "o",
                        "taylor": "^",
                        "magnitude": "D",
                        "compact": "s",
                    }[method],
                    ms=5,
                    lw=1.6,
                    color=colors[method],
                )
            ax.axhline(
                dense[METRICS[i]] if i < 5 else 0.95, color="#555555", ls=":", lw=1.1
            )
            ax.set_title(LABELS[i])
            ax.set_xlabel("Measured MAC reduction (%)")
            ax.set_xticks([25, 40, 55, 70, 85])
            ax.grid(alpha=0.2)
            if i in [2, 5]:
                ax.set_ylim(-0.02, 1.03 if i == 5 else 0.6)
        fig.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=colors[m],
                    ls="--" if m == "compact" else "-",
                    label=NAMES[m],
                )
                for m in NAMES
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.065),
            borderaxespad=0,
            ncol=4,
            frameon=False,
        )
        for ax in axes.flat:
            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ax.get_xticks()))
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ax.get_yticks()))
        for artist in fig.findobj(matplotlib.text.Text):
            artist.set_fontsize(artist.get_fontsize() * 0.9)
        save(fig, 6)
    if 7 in numbers:
        data = pd.read_csv(source / "figure7.csv")
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 7))
        fig.subplots_adjust(left=0.16, bottom=0.15, hspace=0.38, wspace=0.3)
        for i, (ax, metric, margin) in enumerate(
            zip(axes.flat, METRICS[1:5], [-0.02, -0.05, -0.03, 0.01])
        ):
            for y, method in enumerate(["dcap", "taylor", "magnitude"]):
                for split, offset, marker, face in [
                    ("validation", -0.13, "o", COLORS[method]),
                    ("test", 0.13, "D", "white"),
                ]:
                    r = data[
                        (data.method == method)
                        & (data.split == split)
                        & (data.metric == metric)
                    ].iloc[0]
                    ax.hlines(
                        y + offset,
                        r.ci_low,
                        r.ci_high,
                        color=COLORS[method],
                        alpha=0.6,
                        lw=2,
                    )
                    ax.plot(
                        r.delta,
                        y + offset,
                        marker,
                        color=COLORS[method],
                        mfc=face,
                        ms=5,
                    )
            ax.axvline(0, color="gray", lw=0.7)
            ax.axvline(margin, color="#b24b48", ls="--")
            ax.set_yticks(range(3), ["DCAP (Ours)", "Group Taylor", "Group Magnitude"])
            ax.invert_yaxis()
            ax.set_title(LABELS[i + 1])
            ax.set_xlabel("Difference from dense")
            ax.grid(axis="x", alpha=0.2)
        fig.legend(
            handles=[
                Line2D([0], [0], marker="o", color="gray", ls="", label="Validation"),
                Line2D(
                    [0],
                    [0],
                    marker="D",
                    mfc="white",
                    color="gray",
                    ls="",
                    label="Retrospective test",
                ),
                Line2D([0], [0], color="#b24b48", ls="--", label="Preservation margin"),
            ],
            loc="lower center",
            ncol=3,
            frameon=False,
        )
        save(fig, 7)
    if 8 in numbers:
        data = pd.read_csv(source / "figure8.csv")
        data = data[data.split == "retrospective_test"]
        with plt.rc_context({"font.size": 7.2}):
            fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
            fig.subplots_adjust(
                left=0.095, right=0.99, top=0.84, bottom=0.34, wspace=0.16
            )
            norm = matplotlib.colors.Normalize(vmin=0.8, vmax=1)
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "warm_support", ["#f6f1e8", "#dfcdb5", "#bfa080", "#946f52", "#624635"]
            )
            for i, (ax, scale) in enumerate(zip(axes, [0.5, 1, 1.5])):
                matrix = np.array(
                    [
                        [
                            data[
                                (data.margin_scale == scale)
                                & (data.width_km == width)
                                & (data.origin_x_fraction == ox)
                                & (data.origin_y_fraction == oy)
                            ]
                            .iloc[0]
                            .joint_support
                            for ox, oy in [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)]
                        ]
                        for width in [10, 20, 30]
                    ]
                )
                im = ax.pcolormesh(
                    np.arange(-0.5, 4, 1),
                    np.arange(-0.5, 3, 1),
                    matrix,
                    norm=norm,
                    cmap=cmap,
                    shading="flat",
                    rasterized=False,
                    edgecolors="none",
                    antialiased=False,
                )
                ax.set_xlim(-0.5, 3.5)
                ax.set_ylim(2.5, -0.5)
                ax.set_title(f"({chr(97+i)}) {scale:g}× margins", fontsize=8, pad=7)
                ax.set_xticks(range(4), ["0", "E", "N", "EN"])
                ax.set_yticks(range(3), ["10", "20", "30"])
                ax.tick_params(length=0, pad=4)
                ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
                ax.grid(which="minor", color="white", linewidth=0.7)
                ax.tick_params(which="minor", length=0)
                for row in range(3):
                    for col in range(4):
                        v = matrix[row, col]
                        rgb = np.array(cmap(norm(v))[:3])
                        linear = np.where(
                            rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4
                        )
                        lum = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
                        ax.text(
                            col,
                            row,
                            f"{v:.4f}",
                            ha="center",
                            va="center",
                            fontsize=7 / 1.2,
                            color="white" if lum < 0.179 else "black",
                            fontweight="bold" if v >= 0.95 else "normal",
                        )
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if scale == 1:
                    ax.add_patch(
                        Rectangle(
                            (-0.48, 0.52),
                            0.96,
                            0.96,
                            fill=False,
                            edgecolor="#8B0000",
                            linewidth=1.8,
                        )
                    )
            axes[0].set_ylabel("Block width (km)", fontsize=7.2)
            fig.text(0.55, 0.175, "Grid origin", ha="center", fontsize=7.2)
            cax = fig.add_axes([0.28, 0.07, 0.54, 0.045])
            bar = fig.colorbar(
                im, cax=cax, orientation="horizontal", ticks=[0.8, 0.9, 0.95, 1]
            )
            bar.ax.tick_params(labelsize=6.75, length=2, pad=2)
            bar.outline.set_visible(False)
            save(fig, 8)
    if 10 in numbers:
        table, data = deployment_results(source)
        for name, frame in [("table13.csv", table), ("figure10.csv", data)]:
            frame.to_csv(output / name, index=False)
            created.append(str(output / name))
        save(deployment_figure(data), 10)
    return dict(files=created)
