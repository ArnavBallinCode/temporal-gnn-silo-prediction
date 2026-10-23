"""
Figures for the second revision, drawn from outputs/revision2/results.json.

    python3 src/training/revision2_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

RES = Path("outputs/revision2/results.json")
OUT = Path("outputs/revision2/figures")
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})

COL = {"MLP-res": "#f58518", "GCN-res": "#54a24b", "SAGE-res": "#4c78a8", "GCN+GRU-res": "#b279a2",
       "GCN-plain": "#7a7a7a", "recal": "#9d755d", "ar": "#bab0ac", "gbdt": "#e45756", "gbdt_nbr": "#ff9da6"}
LABEL = {"recal": "Recal. persistence", "ar": "Ridge AR", "gbdt": "GBDT", "gbdt_nbr": "GBDT + neighbours"}
CONS = {"clique": "Maximal cliques", "clique_multi": "Cliques, not one email", "clique_w2": "Cliques, edges in $\\geq$2 emails",
        "cpm3": "Clique percolation (k=3)", "louvain": "Louvain communities"}


def fig_curves(R):
    cf = R["chrono_fit"]
    fig, axes = plt.subplots(1, len(cf), figsize=(7.2, 1.9), sharey=False)
    for ax, (m, v) in zip(axes, cf.items()):
        c = np.array(v["curve"])
        ax.plot(c[:, 0], c[:, 1], color="#4c78a8", lw=1, label="train")
        ax.plot(c[:, 0], c[:, 2], color="#e45756", lw=1, label="validation")
        ax.axvline(v["best_epoch"], color="grey", ls=":", lw=0.8)
        ax.set_title(m, fontsize=8)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("loss (BCE + scaled MSE)")
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figS1_training_curves.png", dpi=300)
    plt.close(fig)


def fig_forest(R):
    combos = [("clique", "raw")] + [tuple(k.split("/")) for k in R["sweep"]]
    order = ["MLP-res", "GCN-res", "SAGE-res", "GCN+GRU-res", "recal", "ar", "gbdt_nbr"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.8), sharey=True)
    for ax, tmode in zip(axes, ["raw", "norm"]):
        rows = [c for c in combos if c[1] == tmode]
        y = 0
        yt, yl = [], []
        for c, _ in rows:
            res = R["main_raw"] if (c, tmode) == ("clique", "raw") else R["sweep"][f"{c}/{tmode}"]
            B = res["protocol_B"]
            for j, m in enumerate(order):
                if m not in B:
                    continue
                r, (lo, hi) = B[m]["mse_ratio"], B[m]["ratio_ci95"]
                ax.errorbar(r, y - j * 0.11, xerr=[[r - lo], [hi - r]], fmt="o", ms=2.5, color=COL[m], lw=0.8,
                            label=LABEL.get(m, m) if (c == rows[0][0]) else None)
            yt.append(y - 0.33)
            yl.append(CONS[c])
            y -= 1
        ax.axvline(1, color="black", lw=0.8, ls="--")
        ax.set_xscale("log")
        ticks = [0.75, 1, 1.5, 2, 4, 8]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(0.5, 10)
        ax.set_xlabel("MSE ratio to persistence (log scale)")
        ax.set_title("raw tension" if tmode == "raw" else "normalized tension $T/(K_k-1)$", fontsize=8.5)
        ax.set_yticks(yt)
        ax.set_yticklabels(yl)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "figS2_constructions_forest.png", dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = json.loads(RES.read_text())
    fig_curves(R)
    if "sweep" in R and R["sweep"]:
        fig_forest(R)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
