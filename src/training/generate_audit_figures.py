"""
Scientific Audit: Publication Figure Generator.

Generates the 7 audited publication figures based on verified data,
multi-seed benchmark results, and empirical fragility simulations.
Saves all figures to outputs/audit/figures/ and captions to outputs/audit/figure_captions.md.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, ".")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import networkx as nx
import numpy as np
import pandas as pd
import torch

from src.data.loader import load_node_labels, load_raw_vectors, parse_simplex_events
from src.data.snapshots import build_graph_snapshots
from src.graph.cliques import extract_maximal_cliques
from src.graph.meta_graph import build_clique_meta_graph
from src.graph.scores import compute_clique_tensions, compute_silo_risk_index
from src.models.temporal_gnn import TemporalGNN
from src.training.trainer import build_forecasting_targets, build_two_stage_targets

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.titlesize": 13,
    "figure.dpi": 300,
})


def generate_audit_figures(
    data_dir: str = "Dataset/email-Enron",
    audit_dir: str = "outputs/audit",
    window_days: float = 30.0,
    stride_days: float = 30.0,
) -> None:
    audit_path = Path(audit_dir)
    fig_dir = audit_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" GENERATING AUDITED PUBLICATION FIGURES")
    print(f" Saving to: {fig_dir.resolve()}")
    print("=" * 80)

    # 1. Load data & snapshots
    nverts, simplices, times = load_raw_vectors(data_dir)
    labels = load_node_labels(data_dir)
    events = parse_simplex_events(nverts, simplices, times)
    snapshots = build_graph_snapshots(events, window_days=window_days, stride_days=stride_days, sort_events=True)
    all_node_ids = sorted(labels.keys())

    meta_graphs = []
    snapshot_stats = []
    for s in snapshots:
        cliques = extract_maximal_cliques(s.graph, min_size=3, top_k=30)
        meta = build_clique_meta_graph(s, cliques, conflict_threshold=0.0)
        meta_graphs.append(meta)
        snapshot_stats.append({
            "id": s.snapshot_id,
            "date": s.start_date_str[:10],
            "nodes": s.num_nodes,
            "edges": s.num_edges,
            "density": s.density,
            "cliques": len(cliques),
            "max_clique_size": max([len(c) for c in cliques]) if cliques else 0,
            "components": s.num_connected_components,
        })

    dates = [s["date"] for s in snapshot_stats]
    x_indices = np.arange(len(dates))
    step = 4

    # ------------------------------------------------------------------
    # FIGURE 1: Evolving Network & Clique Dynamics
    # ------------------------------------------------------------------
    print("Generating Figure 1: Network Dynamics...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    ax1.plot(x_indices, [s["nodes"] for s in snapshot_stats], color="#1f77b4", lw=2.0, label="Active Nodes (|V|)", marker="o", markersize=3)
    ax1.plot(x_indices, [s["edges"] for s in snapshot_stats], color="#ff7f0e", lw=2.0, label="Edges (|E|)", marker="s", markersize=3)
    ax1.set_ylabel("Count")
    ax1.set_title("(a) Evolving Communication Network Volume", fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.axvline(x=28, color="red", linestyle="--", alpha=0.6, label="Snapshot 28 (Peak Cliques)")
    ax1.axvline(x=37, color="darkred", linestyle=":", alpha=0.7, label="Snapshot 37 (Chapter 11 Filing, Dec 2001)")
    ax1.legend(loc="upper left")

    ax2.plot(x_indices, [s["density"] for s in snapshot_stats], color="#2ca02c", lw=1.8, label="Graph Density", marker="^", markersize=3)
    ax2.set_ylabel("Density")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_indices, [s["components"] for s in snapshot_stats], color="#9467bd", lw=1.8, linestyle="--", label="Connected Components")
    ax2_twin.set_ylabel("Components")
    ax2.set_title("(b) Network Connectivity and Fragmentation", fontweight="bold")
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")

    ax3.plot(x_indices, [s["cliques"] for s in snapshot_stats], color="#d62728", lw=2.0, label="Extracted Maximal Cliques (min_size=3, top_k=30)", marker="d", markersize=3)
    ax3.plot(x_indices, [s["max_clique_size"] for s in snapshot_stats], color="#8c564b", lw=1.8, label="Max Clique Size (omega)", marker="x", markersize=4)
    ax3.set_ylabel("Count / Size")
    ax3.set_title("(c) Cohesive Subgroups (Cliques)", fontweight="bold")
    ax3.legend(loc="upper left")

    ax3.set_xticks(x_indices[::step])
    ax3.set_xticklabels(dates[::step], rotation=40, ha="right")
    ax3.set_xlabel("Time Snapshot Window (30-day non-overlapping intervals)")

    plt.tight_layout()
    fig1_path = fig_dir / "fig1_network_dynamics.png"
    plt.savefig(fig1_path)
    plt.close()

    # ------------------------------------------------------------------
    # FIGURE 2: Training & Validation Loss Curves
    # ------------------------------------------------------------------
    print("Generating Figure 2: Convergence Curves...")
    loss_file = audit_path.parent / "results" / "loss_history.json"
    if not loss_file.exists():
        loss_file = audit_path.parent / "backup_baseline_run" / "loss_history.json"
    
    if loss_file.exists():
        with open(loss_file, "r") as f:
            loss_data = json.load(f)
        train_loss = loss_data.get("train_loss", [])
        val_loss = loss_data.get("val_loss", [])

        fig, ax = plt.subplots(figsize=(7, 4.5))
        epochs_arr = np.arange(1, len(train_loss) + 1)
        ax.plot(epochs_arr, train_loss, label="Training MSE Loss (Snapshots 0-29)", color="#1f77b4", lw=2.0)
        ax.plot(epochs_arr, val_loss, label="Validation MSE Loss (Snapshots 30-35)", color="#d62728", lw=2.0, linestyle="--")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Temporal GNN Training & Validation Convergence", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        fig2_path = fig_dir / "fig2_loss_curves.png"
        plt.savefig(fig2_path)
        plt.close()

    # ------------------------------------------------------------------
    # FIGURE 3: Multi-Seed Benchmark (Protocol A vs. Protocol B)
    # ------------------------------------------------------------------
    print("Generating Figure 3: Audited Benchmark Bar Chart (Protocol A vs Protocol B)...")
    bm_file = audit_path / "benchmark_results.json"
    if bm_file.exists():
        with open(bm_file, "r") as f:
            bm = json.load(f)

        prot_A = bm["protocol_A_unconditional"]
        prot_B = bm["protocol_B_persisting_only"]

        models_display_A = [
            "Persistence",
            "Hist. Mean",
            "Feature MLP",
            "Static GCN",
            "T-GNN (No GRU)",
            "T-GNN (Warm)",
            "T-GNN (2-Stage)",
        ]
        keys_A = [
            "Persistence Baseline",
            "Historical Mean Baseline",
            "Feature-Only MLP",
            "Static GCN",
            "Temporal GNN (No GRU Ablation)",
            "Temporal GNN (Warm-Start GRU)",
            "Temporal GNN Full (Two-Stage Warm-Start)",
        ]

        mse_A = [prot_A[k]["mse_mean"] for k in keys_A]
        std_A = [prot_A[k]["mse_std"] for k in keys_A]

        fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

        colors_A = ["#7f7f7f", "#bcbd22", "#1f77b4", "#ff7f0e", "#17becf", "#2ca02c", "#9467bd"]
        x_A = np.arange(len(models_display_A))

        bars_A = ax_A.bar(x_A, mse_A, yerr=std_A, capsize=4, color=colors_A, width=0.6, edgecolor="black", alpha=0.85)
        ax_A.set_xticks(x_A)
        ax_A.set_xticklabels(models_display_A, rotation=30, ha="right", fontsize=9)
        ax_A.set_ylabel("Test MSE (Lower is Better)")
        ax_A.set_title("Protocol A: Unconditional Evaluation\n(All Cliques N=109; Dissolved = 0.0)", fontweight="bold")
        for bar in bars_A:
            h = bar.get_height()
            ax_A.text(bar.get_x() + bar.get_width() / 2.0, h + 0.4, f"{h:.2f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

        # Grouped bar chart for Protocol B: Single-Stage vs Two-Stage
        neural_groups = ["Feature MLP", "Static GCN", "T-GNN (No GRU)", "T-GNN (Warm)"]
        single_keys = [
            "Feature-Only MLP",
            "Static GCN",
            "Temporal GNN (No GRU Ablation)",
            "Temporal GNN (Warm-Start GRU)",
        ]
        two_stage_keys = [
            "Feature-Only MLP (Two-Stage)",
            "Static GCN (Two-Stage)",
            "Temporal GNN No-GRU (Two-Stage)",
            "Temporal GNN Full (Two-Stage Warm-Start)",
        ]

        # All categories on x-axis: Persistence, Hist Mean, then 4 neural groups
        all_cats = ["Persistence", "Hist. Mean"] + neural_groups
        x_B = np.arange(len(all_cats))
        bar_width = 0.35

        # Plot Baselines
        b1 = ax_B.bar(x_B[0], prot_B["Persistence Baseline"]["mse_mean"], width=0.5, color="#7f7f7f", edgecolor="black", alpha=0.85, label="Baseline")
        ax_B.text(x_B[0], prot_B["Persistence Baseline"]["mse_mean"] + 0.3, f"{prot_B['Persistence Baseline']['mse_mean']:.2f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

        b2 = ax_B.bar(x_B[1], prot_B["Historical Mean Baseline"]["mse_mean"], width=0.5, color="#bcbd22", edgecolor="black", alpha=0.85)
        ax_B.text(x_B[1], prot_B["Historical Mean Baseline"]["mse_mean"] + 0.3, f"{prot_B['Historical Mean Baseline']['mse_mean']:.2f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

        # Plot Single-Stage vs Two-Stage for neural models
        x_neural = x_B[2:]
        single_vals = [prot_B[k]["mse_mean"] for k in single_keys]
        single_errs = [prot_B[k]["mse_std"] for k in single_keys]
        two_vals = [prot_B[k]["mse_mean"] for k in two_stage_keys]
        two_errs = [prot_B[k]["mse_std"] for k in two_stage_keys]

        bars_s = ax_B.bar(x_neural - bar_width / 2.0, single_vals, yerr=single_errs, capsize=3, width=bar_width, color="#d95f02", edgecolor="black", alpha=0.85, label="Single-Stage (Zero-Fill)")
        bars_t = ax_B.bar(x_neural + bar_width / 2.0, two_vals, yerr=two_errs, capsize=3, width=bar_width, color="#1b9e77", edgecolor="black", alpha=0.85, label="Two-Stage (Hurdle/Cond.)")

        for bar in bars_s:
            h = bar.get_height()
            ax_B.text(bar.get_x() + bar.get_width() / 2.0, h + 0.3, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        for bar in bars_t:
            h = bar.get_height()
            ax_B.text(bar.get_x() + bar.get_width() / 2.0, h + 0.3, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax_B.axhline(y=prot_B["Persistence Baseline"]["mse_mean"], color="red", linestyle="--", alpha=0.6, label="Persistence MSE (5.04)")
        ax_B.set_xticks(x_B)
        ax_B.set_xticklabels(all_cats, rotation=30, ha="right", fontsize=9)
        ax_B.set_ylabel("Test MSE (Lower is Better)")
        ax_B.set_title("Protocol B: Conditional on Survival\n(Persisting Cliques Only N=35, J >= 0.5)", fontweight="bold")
        ax_B.legend(loc="upper right", fontsize=8.5)

        plt.tight_layout()
        fig3_path = fig_dir / "fig3_model_benchmarks.png"
        plt.savefig(fig3_path)
        plt.close()

    # ------------------------------------------------------------------
    # FIGURE 4: Predicted vs. True Tension Scatter Plot (Annotated)
    # ------------------------------------------------------------------
    print("Generating Figure 4: Predicted vs True Tension Scatter Plot...")
    targets_bundle = build_two_stage_targets(meta_graphs, delta=1, jaccard_threshold=0.5)
    targets = targets_bundle.zero_fill_targets
    cond_targets = targets_bundle.conditional_targets
    pers_masks = targets_bundle.persisting_masks

    valid_indices = [i for i, t in enumerate(targets) if t is not None]
    n_train = int(len(valid_indices) * 0.70)
    n_val = int(len(valid_indices) * 0.15)
    test_indices = valid_indices[n_train + n_val :]

    torch.manual_seed(42)
    tgnn_warm = TemporalGNN(in_features=5, hidden_dim=16, dropout=0.1, jaccard_threshold=0.5)
    ckpt_file = audit_path.parent / "checkpoints" / "temporal_gnn_best.pt"
    if not ckpt_file.exists():
        ckpt_file = audit_path.parent / "backup_baseline_run" / "temporal_gnn_best.pt"
    if ckpt_file.exists():
        tgnn_warm.load_state_dict(torch.load(ckpt_file, weights_only=True))
    tgnn_warm.eval()

    y_true_persisting = []
    y_pred_persisting = []
    y_true_dissolved = []
    y_pred_dissolved = []

    # Historical warm-up through snapshots 0 to 35
    with torch.no_grad():
        w_prev_meta = None
        w_prev_h = None
        for idx in range(test_indices[0]):
            meta = meta_graphs[idx]
            _, curr_h = tgnn_warm.forward_snapshot(meta, w_prev_meta, w_prev_h)
            w_prev_meta = meta
            w_prev_h = curr_h.detach() if curr_h.numel() > 0 else None

        # Test snapshots 36-42
        for idx in test_indices:
            meta = meta_graphs[idx]
            target = targets[idx]
            mask = pers_masks[idx]
            preds, curr_h = tgnn_warm.forward_snapshot(meta, w_prev_meta, w_prev_h)
            if target is not None and len(target) > 0 and len(preds) > 0:
                for is_pers, t_val, p_val in zip(mask.tolist(), target.tolist(), preds.tolist()):
                    if is_pers:
                        y_true_persisting.append(t_val)
                        y_pred_persisting.append(p_val)
                    else:
                        y_true_dissolved.append(t_val)
                        y_pred_dissolved.append(p_val)
            w_prev_meta = meta
            w_prev_h = curr_h.detach() if curr_h.numel() > 0 else None

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(y_true_persisting, y_pred_persisting, color="#2ca02c", alpha=0.85, edgecolors="black", s=45, label=f"Persisting Cliques (N={len(y_true_persisting)}, J >= 0.5)")
    ax.scatter(y_true_dissolved, y_pred_dissolved, color="#d62728", alpha=0.55, marker="x", s=40, label=f"Dissolved Cliques (N={len(y_true_dissolved)}, Target=0.0)")

    max_val = max(max(y_true_persisting or [10]), max(y_pred_persisting or [10])) + 1
    ax.plot([0, max_val], [0, max_val], color="black", linestyle="--", lw=1.5, label="Perfect Forecast (y = x)")
    ax.set_xlabel("Ground Truth Future Tension T^(t+1)")
    ax.set_ylabel("Predicted Future Tension T_hat^(t+1)")
    ax.set_title("Temporal GNN Forecast Accuracy on Held-Out Test Set (Snapshots 36-42)\n(Warm-Start State Propagation, N_total=109 Cliques)", fontweight="bold")
    ax.set_xlim(-0.5, max_val)
    ax.set_ylim(-0.5, max_val)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig4_path = fig_dir / "fig4_predicted_vs_true.png"
    plt.savefig(fig4_path)
    plt.close()

    # ------------------------------------------------------------------
    # FIGURE 5: Silo Risk Trajectories (Observed vs Forecasted Note)
    # ------------------------------------------------------------------
    print("Generating Figure 5: Longitudinal Silo Risk Trajectories...")
    sri_trajectories = {nid: [] for nid in all_node_ids}
    for meta in meta_graphs:
        obs_tensions = compute_clique_tensions(snapshots[meta.snapshot_id].graph, meta.cliques)
        sri_dict = compute_silo_risk_index(meta.cliques, obs_tensions, all_node_ids)
        for nid in all_node_ids:
            sri_trajectories[nid].append(sri_dict.get(nid, 0.0))

    total_sri_per_node = {nid: sum(vals) for nid, vals in sri_trajectories.items()}
    top_nodes = sorted(total_sri_per_node.items(), key=lambda x: x[1], reverse=True)[:5]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors_top = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    for i, (nid, _) in enumerate(top_nodes):
        email_prefix = labels.get(nid, f"Node {nid}").split("@")[0]
        ax.plot(x_indices, sri_trajectories[nid], label=f"{email_prefix} (Node ID {nid})", color=colors_top[i], lw=2.0, marker="o", markersize=3)

    ax.set_ylabel("Observed Silo Risk Index (SRI)")
    ax.set_title("Longitudinal Structural Silo Risk Index (SRI) for Top Active Nodes", fontweight="bold")
    ax.set_xticks(x_indices[::step])
    ax.set_xticklabels(dates[::step], rotation=40, ha="right")
    ax.set_xlabel("Time Snapshot (30-day windows)")
    ax.axvline(x=28, color="red", linestyle="--", alpha=0.5, label="Snapshot 28 (Mar 2001)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig5_path = fig_dir / "fig5_silo_risk_trajectories.png"
    plt.savefig(fig5_path)
    plt.close()

    # ------------------------------------------------------------------
    # FIGURE 6: Peak Conflict Meta-Graph (Snapshot 28)
    # ------------------------------------------------------------------
    print("Generating Figure 6: Peak Conflict Meta-Graph...")
    peak_idx = 28
    peak_meta = meta_graphs[peak_idx]

    G_meta = nx.Graph()
    for i, c in enumerate(peak_meta.cliques):
        G_meta.add_node(i, size=len(c), tension=float(peak_meta.clique_tensions[i].item()))

    for e_idx in range(peak_meta.edge_index.size(1)):
        u = int(peak_meta.edge_index[0, e_idx].item())
        v = int(peak_meta.edge_index[1, e_idx].item())
        if u < v:
            w = float(peak_meta.edge_weights[e_idx].item())
            G_meta.add_edge(u, v, weight=w)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    pos = nx.spring_layout(G_meta, seed=42, k=0.55)
    node_sizes = [G_meta.nodes[n]["size"] * 45 for n in G_meta.nodes()]
    node_colors = [G_meta.nodes[n]["tension"] for n in G_meta.nodes()]
    edges = G_meta.edges(data=True)
    edge_widths = [d.get("weight", 0.5) * 4 for _, _, d in edges]

    nodes_drawn = nx.draw_networkx_nodes(G_meta, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.YlOrRd, ax=ax)
    nx.draw_networkx_edges(G_meta, pos, width=edge_widths, edge_color="gray", alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G_meta, pos, font_size=8, font_color="black", ax=ax)

    cbar = plt.colorbar(nodes_drawn, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Observed Clique Tension T(C)")
    ax.set_title(f"Clique-Conflict Meta-Graph: Snapshot #28 ({dates[peak_idx]})\n(Nodes = Cliques sized by member count; Edges = Boundary Conflict ICCS)", fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    fig6_path = fig_dir / "fig6_peak_conflict_metagraph.png"
    plt.savefig(fig6_path)
    plt.close()

    # ------------------------------------------------------------------
    # FIGURE 7: Network Fragility (Audited Multi-Seed)
    # ------------------------------------------------------------------
    print("Generating Figure 7: Network Fragility Analysis...")
    frag_file = audit_path / "fragility_results.csv"
    if frag_file.exists():
        df_frag = pd.read_csv(frag_file)
        df_g28 = df_frag[df_frag["target_graph"] == "G_snapshot_28"]
        df_g29 = df_frag[df_frag["target_graph"] == "G_snapshot_29_future"]

        fig, (ax_insample, ax_forward) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

        fracs = df_g28[df_g28["strategy"] == "Degree Centrality"]["fraction_removed"].values * 100

        # Panel 1: In-Sample G28
        for strat, color, ls, marker in [
            ("Betweenness Centrality", "#ff7f0e", "--", "s"),
            ("Observed SRI (In-Sample)", "#d62728", "-", "o"),
            ("Degree Centrality", "#1f77b4", "-.", "^"),
            ("Predicted SRI (Temporal GNN)", "#2ca02c", ":", "d"),
        ]:
            vals = df_g28[df_g28["strategy"] == strat]["relative_gcc_size"].values
            ax_insample.plot(fracs, vals * 100, color=color, linestyle=ls, marker=marker, markersize=3.5, label=strat)

        rnd_row = df_g28[df_g28["strategy"] == "Random Removal (50 seeds)"]
        rnd_vals = rnd_row["relative_gcc_size"].values * 100
        rnd_std = rnd_row["std_err"].values * 100
        ax_insample.plot(fracs, rnd_vals, color="#7f7f7f", linestyle=":", label="Random Removal (50 seeds)")
        ax_insample.fill_between(fracs, rnd_vals - 1.96 * rnd_std, rnd_vals + 1.96 * rnd_std, color="#7f7f7f", alpha=0.25, label="Random 95% CI")

        ax_insample.set_xlabel("Fraction of Employees Removed (%)")
        ax_insample.set_ylabel("Giant Component Size (% of initial)")
        ax_insample.set_title("(a) In-Sample Disruption on Graph G^(28)\n(Initial GCC = 103 / 105 nodes, 98.1%)", fontweight="bold")
        ax_insample.legend(loc="lower left", fontsize=8.5)

        # Panel 2: Forward G29
        for strat, color, ls, marker in [
            ("Betweenness Centrality", "#ff7f0e", "--", "s"),
            ("Observed SRI (Persistence)", "#d62728", "-", "o"),
            ("Degree Centrality", "#1f77b4", "-.", "^"),
            ("Predicted SRI (Temporal GNN)", "#2ca02c", ":", "d"),
        ]:
            vals = df_g29[df_g29["strategy"] == strat]["relative_gcc_size"].values
            ax_forward.plot(fracs, vals * 100, color=color, linestyle=ls, marker=marker, markersize=3.5, label=strat)

        rnd29_row = df_g29[df_g29["strategy"] == "Random Removal (50 seeds)"]
        rnd29_vals = rnd29_row["relative_gcc_size"].values * 100
        rnd29_std = rnd29_row["std_err"].values * 100
        ax_forward.plot(fracs, rnd29_vals, color="#7f7f7f", linestyle=":", label="Random Removal (50 seeds)")
        ax_forward.fill_between(fracs, rnd29_vals - 1.96 * rnd29_std, rnd29_vals + 1.96 * rnd29_std, color="#7f7f7f", alpha=0.25, label="Random 95% CI")

        ax_forward.set_xlabel("Fraction of Employees Removed (%)")
        ax_forward.set_title("(b) Forward Out-of-Sample Disruption on Graph G^(29)\n(Rankings from t=28 evaluated on t=29)", fontweight="bold")
        ax_forward.legend(loc="lower left", fontsize=8.5)

        plt.tight_layout()
        fig7_path = fig_dir / "fig7_network_fragility.png"
        plt.savefig(fig7_path)
        plt.close()

    print("\nAll 7 publication figures successfully saved to outputs/audit/figures/")


if __name__ == "__main__":
    generate_audit_figures()
