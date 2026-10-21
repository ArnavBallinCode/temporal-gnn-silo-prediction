"""
Figures for the revised manuscript, drawn from outputs/revision/revision_results.json.

    python3 src/training/revision_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("outputs/revision/revision_results.json")
OUT = Path("outputs/revision/figures")

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})

C_BASE = "#7a7a7a"
C_TAB = "#4c78a8"
C_COND = "#f58518"
C_TWO = "#54a24b"
C_ZERO = "#b279a2"


def colour(name: str) -> str:
    if "two-stage" in name:
        return C_TWO
    if "cond-only" in name:
        return C_COND
    if "zero-fill" in name:
        return C_ZERO
    if name in ("persistence", "hist_mean"):
        return C_BASE
    return C_TAB


LABEL = {"persistence": "Persistence", "recal": "Recalibrated persistence", "ar": "Ridge AR (tension history)",
         "ridge_all": "Ridge (all features)", "gbdt": "GBDT (5 feats + age)", "hist_mean": "Historical mean"}


def fig_protocol_b(res):
    main = res["main"]
    order = ["persistence", "recal", "ar", "gbdt", "MLP zero-fill", "GCN zero-fill", "GCN+GRU zero-fill",
             "MLP cond-only", "GCN cond-only", "GCN+GRU cond-only", "MLP two-stage", "GCN two-stage", "GCN+GRU two-stage"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1), sharey=True)
    for ax, (key, title) in zip(axes, [("main", "Chronological split (test = Nov 2001-Mar 2002)"),
                                       ("rolling", "Rolling origin (pooled over origins)")]):
        if key == "main":
            pb, boot = main["protocol_B"], main["bootstrap_B"]
            m = [pb[k]["mse_mean"] for k in order]
            lo = [boot[k]["mse_lo"] for k in order]
            hi = [boot[k]["mse_hi"] for k in order]
        else:
            pb = res["rolling"]["protocol_B"]
            order_r = [k for k in order if k in pb]
            m = [pb[k]["mse_mean"] if k in pb else np.nan for k in order]
            lo = [pb[k]["mse_lo"] if k in pb else np.nan for k in order]
            hi = [pb[k]["mse_hi"] if k in pb else np.nan for k in order]
        y = np.arange(len(order))
        m, lo, hi = np.array(m), np.array(lo), np.array(hi)
        ax.barh(y, m, color=[colour(k) for k in order], height=0.65)
        ok = ~np.isnan(m)
        ax.errorbar(m[ok], y[ok], xerr=[m[ok] - lo[ok], hi[ok] - m[ok]], fmt="none", ecolor="black", lw=0.8, capsize=2)
        ref = m[0]
        ax.axvline(ref, color=C_BASE, ls="--", lw=0.8)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("MSE on surviving cliques (Protocol B)")
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL.get(k, k) for k in order])
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "figR1_protocol_b_controls.png", dpi=300)
    plt.close(fig)


def fig_rolling(res):
    ro = res["rolling"]["per_origin"]
    o = [r["origin"] for r in ro]
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
    surv = [r["n_surv"] / r["n_cliques"] for r in ro]
    axes[0].bar(o, surv, color=C_BASE, width=0.7)
    axes[0].set_ylabel("Survival rate")
    axes[0].set_ylim(0, 1)
    axes[0].axvspan(35.5, max(o) + 0.5, color="#f2d0d0", alpha=0.5, lw=0)
    for key, lab, c, mk in [("mse_persist", "Persistence", C_BASE, "o"), ("mse_recal", "Recal. persistence", C_TAB, "s"),
                            ("mse_gcn2", "GCN two-stage", C_TWO, "^"), ("mse_gru2", "GCN+GRU two-stage", "#2a6b2a", "v")]:
        axes[1].plot(o, [r[key] for r in ro], marker=mk, ms=3, lw=1, color=c, label=lab)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Protocol B MSE (log)")
    axes[1].set_xlabel("Forecast origin $k$ (predict $k\\to k+1$)")
    axes[1].axvspan(35.5, max(o) + 0.5, color="#f2d0d0", alpha=0.5, lw=0)
    axes[1].legend(fontsize=6.5, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "figR2_rolling_origin.png", dpi=300)
    plt.close(fig)


def fig_sensitivity(res):
    rows = res["sensitivity"]["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2), sharey=True)
    for ax, param, xl in zip(axes, ["tau", "top_k", "window/stride"], ["Jaccard threshold $\\tau$", "Retained cliques $K'$", "Window / stride (days)"]):
        rr = [r for r in rows if r["param"] == param]
        x = np.arange(len(rr))
        ax.plot(x, [r["survival_all"] for r in rr], marker="o", color=C_BASE, label="all transitions")
        ax.plot(x, [r["survival_train"] for r in rr], marker="s", ms=3, lw=0.8, color=C_TAB, label="train")
        ax.plot(x, [r["survival_test"] for r in rr], marker="^", ms=3, lw=0.8, color="#c44e52", label="test")
        ax.set_xticks(x)
        ax.set_xticklabels([("all" if r["value"] == "None" else r["value"]) for r in rr], fontsize=7)
        ax.set_xlabel(xl)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Clique survival rate")
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figR3_sensitivity.png", dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads(RES.read_text())
    fig_protocol_b(res)
    fig_rolling(res)
    fig_sensitivity(res)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
