#!/usr/bin/env python3
"""
Publication-Quality Visualization & Empirical Statistics Generator.

Generates 7 analytical figures and empirical statistics for the paper:
1. Fig 1: Evolving Network Dynamics (Nodes, Edges, Cliques across 44 snapshots)
2. Fig 2: Training & Validation Loss Curves
3. Fig 3: Model Benchmark Comparison (MSE, MAE, Spearman rho, Top-k overlap)
4. Fig 4: Predicted vs. True Future Clique Tension Scatter Plot
5. Fig 5: Longitudinal Silo Risk Trajectories for Top Individuals
6. Fig 6: Peak Conflict Meta-Graph Visualization (March 2001 California Crisis)
7. Fig 7: Network Fragility Analysis (Targeted Node Removal Simulation)

Usage:
    python3 src/training/visualize_results.py [--output_dir outputs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import networkx as nx
import numpy as np
import torch

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import load_node_labels, load_raw_vectors, parse_simplex_events
from src.data.snapshots import build_graph_snapshots
from src.graph.cliques import extract_maximal_cliques
from src.graph.meta_graph import build_clique_meta_graph
from src.graph.scores import compute_clique_tensions, compute_silo_risk_index
from src.models.temporal_gnn import TemporalGNN
from src.training.trainer import build_forecasting_targets

# Use clean modern styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def generate_all_visualizations(
    data_dir: str = "Dataset/email-Enron",
    output_dir: str = "outputs",
    window_days: float = 30.0,
    stride_days: float = 30.0,
) -> None:
    out_path = Path(output_dir)
    fig_dir = out_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" GENERATING PUBLICATION-QUALITY FIGURES & EMPIRICAL STATISTICS")
    print(f" Output Directory: {fig_dir.resolve()}")
    print("=" * 80)

    # 1. Load data & snapshots
    print("\n[1/7] Processing 44 graph snapshots and maximal cliques...")
    nverts, simplices, times = load_raw_vectors(data_dir)
    labels = load_node_labels(data_dir)
    events = parse_simplex_events(nverts, simplices, times)
    snapshots = build_graph_snapshots(events, window_days=window_days, stride_days=stride_days)
    all_node_ids = sorted(labels.keys())

    meta_graphs = []
    snapshot_stats = []
    for s in snapshots:
        cliques = extract_maximal_cliques(s.graph, min_size=3, top_k=30)
        meta = build_clique_meta_graph(s, cliques, conflict_threshold=0.0)
        meta_graphs.append(meta)

        max_c_sz = max([len(c) for c in cliques]) if cliques else 0
        snapshot_stats.append({
            "id": s.snapshot_id,
            "date": s.start_date_str[:10],
            "nodes": s.num_nodes,
            "edges": s.num_edges,
            "density": s.density,
            "cliques": len(cliques),
            "max_clique_size": max_c_sz,
            "components": s.num_connected_components,
        })

    # -------------------------------------------------------------
    # FIGURE 1: Network & Clique Evolution Dynamics
    # -------------------------------------------------------------
    print("[2/7] Generating Figure 1: Network & Clique Dynamics...")
    dates = [s["date"] for s in snapshot_stats]
    x_indices = np.arange(len(dates))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Nodes & Edges
    ax1.plot(x_indices, [s["nodes"] for s in snapshot_stats], color="#1f77b4", lw=2.2, label="Active Employees (|V|)", marker="o", markersize=3)
    ax1.plot(x_indices, [s["edges"] for s in snapshot_stats], color="#ff7f0e", lw=2.2, label="Communication Edges (|E|)", marker="s", markersize=3)
    ax1.set_ylabel("Count")
    ax1.set_title("(a) Evolving Organizational Communication Volume", fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.axvline(x=28, color="red", linestyle="--", alpha=0.7, label="Peak California Crisis (Mar 2001)")
    ax1.axvline(x=37, color="darkred", linestyle=":", alpha=0.8, label="Bankruptcy Filing (Dec 2001)")

    # Panel 2: Density & Components
    ax2.plot(x_indices, [s["density"] for s in snapshot_stats], color="#2ca02c", lw=2, label="Graph Density", marker="^", markersize=3)
    ax2.set_ylabel("Density")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_indices, [s["components"] for s in snapshot_stats], color="#9467bd", lw=1.8, linestyle="--", label="Connected Components")
    ax2_twin.set_ylabel("Components")
    ax2.set_title("(b) Network Connectivity & Fragmentation", fontweight="bold")
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")

    # Panel 3: Maximal Cliques & Max Clique Size
    ax3.plot(x_indices, [s["cliques"] for s in snapshot_stats], color="#d62728", lw=2.2, label="Maximal Cliques (|C|)", marker="d", markersize=3)
    ax3.plot(x_indices, [s["max_clique_size"] for s in snapshot_stats], color="#8c564b", lw=2.0, label="Max Clique Size (omega)", marker="x", markersize=4)
    ax3.set_ylabel("Count / Size")
    ax3.set_title("(c) Cohesive Subgroups (Cliques)", fontweight="bold")
    ax3.legend(loc="upper left")

    # X-axis ticks every 4 snapshots
    step = 4
    ax3.set_xticks(x_indices[::step])
    ax3.set_xticklabels(dates[::step], rotation=40, ha="right")
    ax3.set_xlabel("Time Snapshot Window (30-day intervals)")

    plt.tight_layout()
    fig1_path = fig_dir / "fig1_network_dynamics.png"
    plt.savefig(fig1_path)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 2: Training & Validation Loss Curves
    # -------------------------------------------------------------
    print("[3/7] Generating Figure 2: Training & Validation Loss Curves...")
    loss_file = out_path / "results" / "loss_history.json"
    if loss_file.exists():
        with open(loss_file, "r") as f:
            loss_data = json.load(f)
        train_loss = loss_data.get("train_loss", [])
        val_loss = loss_data.get("val_loss", [])

        fig, ax = plt.subplots(figsize=(8, 5))
        epochs_arr = np.arange(1, len(train_loss) + 1)
        ax.plot(epochs_arr, train_loss, label="Training MSE Loss", color="#1f77b4", lw=2.2)
        ax.plot(epochs_arr, val_loss, label="Validation MSE Loss", color="#d62728", lw=2.2, linestyle="--")
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("Mean Squared Error (MSE)")
        ax.set_title("Temporal GNN Training & Validation Convergence", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        fig2_path = fig_dir / "fig2_loss_curves.png"
        plt.savefig(fig2_path)
        plt.close()

    # -------------------------------------------------------------
    # FIGURE 3: Benchmark Model Comparisons
    # -------------------------------------------------------------
    print("[4/7] Generating Figure 3: Benchmark Model Comparison Bar Charts...")
    metrics_file = out_path / "results" / "benchmark_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            bm = json.load(f)

        models = ["Persistence", "Feature MLP", "Static GCN", "Temporal GNN"]
        mse_vals = [
            bm["persistence_baseline"]["mse"],
            bm["feature_only_mlp"]["mse"],
            bm["static_gcn"]["mse"],
            bm["temporal_gnn"]["mse"],
        ]
        mae_vals = [
            bm["persistence_baseline"]["mae"],
            bm["feature_only_mlp"]["mae"],
            bm["static_gcn"]["mae"],
            bm["temporal_gnn"]["mae"],
        ]

        fig, (ax_mse, ax_mae) = plt.subplots(1, 2, figsize=(12, 5))

        colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]
        bars1 = ax_mse.bar(models, mse_vals, color=colors, width=0.55, edgecolor="black", alpha=0.85)
        ax_mse.set_ylabel("Test MSE (Lower is Better)")
        ax_mse.set_title("Future Clique Tension Forecasting Error (MSE)", fontweight="bold")
        for bar in bars1:
            h = bar.get_height()
            ax_mse.text(bar.get_x() + bar.get_width() / 2.0, h + 0.4, f"{h:.2f}", ha="center", va="bottom", fontweight="bold")

        bars2 = ax_mae.bar(models, mae_vals, color=colors, width=0.55, edgecolor="black", alpha=0.85)
        ax_mae.set_ylabel("Test MAE (Lower is Better)")
        ax_mae.set_title("Mean Absolute Error (MAE)", fontweight="bold")
        for bar in bars2:
            h = bar.get_height()
            ax_mae.text(bar.get_x() + bar.get_width() / 2.0, h + 0.08, f"{h:.2f}", ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        fig3_path = fig_dir / "fig3_model_benchmarks.png"
        plt.savefig(fig3_path)
        plt.close()

    # -------------------------------------------------------------
    # FIGURE 4: Predicted vs. True Future Tension Scatter Plot
    # -------------------------------------------------------------
    print("[5/7] Generating Figure 4: Predicted vs True Tension Scatter Plot...")
    targets = build_forecasting_targets(meta_graphs, delta=1)
    valid_indices = [i for i, t in enumerate(targets) if t is not None]
    n_train = int(len(valid_indices) * 0.70)
    n_val = int(len(valid_indices) * 0.15)
    test_indices = valid_indices[n_train + n_val :]

    # Load model checkpoint
    model = TemporalGNN(in_features=5, hidden_dim=16)
    ckpt_path = out_path / "checkpoints" / "temporal_gnn_best.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        v_prev_meta = None
        v_prev_h = None
        for idx in test_indices:
            meta = meta_graphs[idx]
            target = targets[idx]
            preds, curr_h = model.forward_snapshot(meta, v_prev_meta, v_prev_h)
            if target is not None and len(target) > 0 and len(preds) > 0:
                y_true_list.extend(target.tolist())
                y_pred_list.extend(preds.tolist())
            v_prev_meta = meta
            v_prev_h = curr_h

    if y_true_list:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_true_list, y_pred_list, color="#1f77b4", alpha=0.6, edgecolors="none", s=35, label="Test Cliques")
        max_val = max(max(y_true_list), max(y_pred_list)) + 1
        ax.plot([0, max_val], [0, max_val], color="red", linestyle="--", lw=1.8, label="Ideal Forecast (y = x)")
        ax.set_xlabel("True Future Tension T^(k+1)")
        ax.set_ylabel("Predicted Future Tension T_hat^(k+1)")
        ax.set_title("Forecast Accuracy on Held-Out Test Set", fontweight="bold")
        ax.set_xlim(-0.5, max_val)
        ax.set_ylim(-0.5, max_val)
        ax.legend()
        plt.tight_layout()
        fig4_path = fig_dir / "fig4_predicted_vs_true.png"
        plt.savefig(fig4_path)
        plt.close()

    # -------------------------------------------------------------
    # FIGURE 5: High-Silo Risk Trajectories for Key Individuals
    # -------------------------------------------------------------
    print("[6/7] Generating Figure 5: Longitudinal Silo Risk Trajectories...")
    # Track SRI across all snapshots for top individuals
    sri_trajectories = {nid: [] for nid in all_node_ids}
    for meta in meta_graphs:
        sri_dict = compute_silo_risk_index(meta.cliques, compute_clique_tensions(snapshots[meta.snapshot_id].graph, meta.cliques), all_node_ids)
        for nid in all_node_ids:
            sri_trajectories[nid].append(sri_dict.get(nid, 0.0))

    # Identify top 5 individuals with highest lifetime tension
    total_tension_per_node = {nid: sum(vals) for nid, vals in sri_trajectories.items()}
    top_nodes = sorted(total_tension_per_node.items(), key=lambda x: x[1], reverse=True)[:5]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors_top = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    for i, (nid, total_sri) in enumerate(top_nodes):
        email_prefix = labels.get(nid, f"Node {nid}").split("@")[0]
        ax.plot(x_indices, sri_trajectories[nid], label=f"{email_prefix} (ID {nid})", color=colors_top[i], lw=2.2, marker="o", markersize=3)

    ax.set_ylabel("Silo Risk Index (SRI)")
    ax.set_title("Evolving Silo Risk Trajectories for Top Broker/Boundary Employees", fontweight="bold")
    ax.set_xticks(x_indices[::step])
    ax.set_xticklabels(dates[::step], rotation=40, ha="right")
    ax.set_xlabel("Time Snapshot (30-day windows)")
    ax.axvline(x=28, color="red", linestyle="--", alpha=0.6, label="California Crisis Peak")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig5_path = fig_dir / "fig5_silo_risk_trajectories.png"
    plt.savefig(fig5_path)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 6: Peak Conflict Meta-Graph (Snapshot #28 - March 2001)
    # -------------------------------------------------------------
    print("[7/7] Generating Figure 6: Peak Conflict Meta-Graph Visualization...")
    peak_snap_idx = 28  # March 2001
    peak_meta = meta_graphs[peak_snap_idx]

    G_meta = nx.Graph()
    for i, c in enumerate(peak_meta.cliques):
        G_meta.add_node(i, size=len(c), tension=float(peak_meta.clique_tensions[i].item()))

    for e_idx in range(peak_meta.edge_index.size(1)):
        u = int(peak_meta.edge_index[0, e_idx].item())
        v = int(peak_meta.edge_index[1, e_idx].item())
        if u < v:
            w = float(peak_meta.edge_weights[e_idx].item())
            G_meta.add_edge(u, v, weight=w)

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G_meta, seed=42, k=0.55)
    node_sizes = [G_meta.nodes[n]["size"] * 45 for n in G_meta.nodes()]
    node_colors = [G_meta.nodes[n]["tension"] for n in G_meta.nodes()]

    edges = G_meta.edges(data=True)
    edge_widths = [d.get("weight", 0.5) * 4 for _, _, d in edges]

    nodes_drawn = nx.draw_networkx_nodes(G_meta, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.YlOrRd, ax=ax)
    nx.draw_networkx_edges(G_meta, pos, width=edge_widths, edge_color="gray", alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G_meta, pos, font_size=8, font_color="black", ax=ax)

    cbar = plt.colorbar(nodes_drawn, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Clique Tension T(C)")
    ax.set_title(f"Clique Conflict Meta-Graph: Snapshot #28 ({dates[peak_snap_idx]})\n(Nodes = Cliques sized by team size, Edges = Boundary Conflict ICCS)", fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    fig6_path = fig_dir / "fig6_peak_conflict_metagraph.png"
    plt.savefig(fig6_path)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 7: Network Fragility Analysis (Targeted Removal)
    # -------------------------------------------------------------
    print("Generating Figure 7: Network Fragility Analysis (Section V-D)...")
    # Use aggregated global graph or peak graph
    G_eval = snapshots[28].graph.copy()
    initial_nodes = list(G_eval.nodes())
    N_total = len(initial_nodes)

    # Compute different node ranking criteria
    sri_scores = compute_silo_risk_index(
        meta_graphs[28].cliques,
        compute_clique_tensions(G_eval, meta_graphs[28].cliques),
        initial_nodes,
    )
    degrees = dict(G_eval.degree())
    betweenness = nx.betweenness_centrality(G_eval)

    ranking_sri = sorted(initial_nodes, key=lambda n: sri_scores.get(n, 0.0), reverse=True)
    ranking_degree = sorted(initial_nodes, key=lambda n: degrees.get(n, 0), reverse=True)
    ranking_betweenness = sorted(initial_nodes, key=lambda n: betweenness.get(n, 0.0), reverse=True)
    np.random.seed(42)
    ranking_random = list(np.random.permutation(initial_nodes))

    fractions = np.linspace(0.0, 0.50, 20)

    def simulate_removal(node_order):
        gcc_sizes = []
        for frac in fractions:
            k_rem = int(frac * N_total)
            nodes_to_remove = set(node_order[:k_rem])
            subgraph = G_eval.subgraph([n for n in initial_nodes if n not in nodes_to_remove])
            if subgraph.number_of_nodes() > 0:
                gcc = len(max(nx.connected_components(subgraph), key=len))
            else:
                gcc = 0
            gcc_sizes.append(gcc / float(N_total))
        return gcc_sizes

    gcc_sri = simulate_removal(ranking_sri)
    gcc_deg = simulate_removal(ranking_degree)
    gcc_bet = simulate_removal(ranking_betweenness)
    gcc_rnd = simulate_removal(ranking_random)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(fractions * 100, gcc_sri, color="#d62728", lw=2.5, marker="o", markersize=4, label="Targeted Removal by Silo Risk Index (SRI)")
    ax.plot(fractions * 100, gcc_bet, color="#ff7f0e", lw=2.0, marker="s", markersize=4, linestyle="--", label="Targeted Removal by Betweenness Centrality")
    ax.plot(fractions * 100, gcc_deg, color="#1f77b4", lw=2.0, marker="^", markersize=4, linestyle="-.", label="Targeted Removal by Degree Centrality")
    ax.plot(fractions * 100, gcc_rnd, color="#7f7f7f", lw=1.8, marker="x", markersize=4, linestyle=":", label="Random Removal Baseline")

    ax.set_xlabel("Fraction of Employees Removed (%)")
    ax.set_ylabel("Relative Giant Component Size (|GCC| / N)")
    ax.set_title("Network Fragility Analysis: Network Collapse under Targeted Node Removal", fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig7_path = fig_dir / "fig7_network_fragility.png"
    plt.savefig(fig7_path)
    plt.close()

    print("\n" + "=" * 80)
    print(" ALL 7 PUBLICATION FIGURES SUCCESSFULLY GENERATED & SAVED:")
    print("=" * 80)
    print(f"  - Fig 1: {fig1_path.name}  (Evolving network volume, density, maximal cliques)")
    print(f"  - Fig 2: {fig2_path.name}  (Training vs Validation convergence loss)")
    print(f"  - Fig 3: {fig3_path.name}  (Model benchmark error comparison: MSE & MAE)")
    print(f"  - Fig 4: {fig4_path.name}  (Predicted vs ground-truth tension scatter plot)")
    print(f"  - Fig 5: {fig5_path.name}  (Silo Risk Index trajectories for top individuals)")
    print(f"  - Fig 6: {fig6_path.name}  (Peak conflict meta-graph in March 2001)")
    print(f"  - Fig 7: {fig7_path.name}  (Network fragility & rapid fragmentation analysis)")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures and empirical statistics.")
    parser.add_argument("--data_dir", type=str, default="Dataset/email-Enron")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--window_days", type=float, default=30.0)
    parser.add_argument("--stride_days", type=float, default=30.0)
    args = parser.parse_args()

    generate_all_visualizations(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_days=args.window_days,
        stride_days=args.stride_days,
    )


if __name__ == "__main__":
    main()
