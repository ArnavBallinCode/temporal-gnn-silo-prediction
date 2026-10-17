"""
Auditing Network Fragility Analysis and Targeted Node Removal.

Methodological Checks:
1. Multi-seed random removal: Run 50 random permutations to compute mean and 95% confidence interval.
2. In-sample disruption on G^(28): Compare:
   - Observed SRI at snapshot 28
   - Degree Centrality at snapshot 28
   - Betweenness Centrality at snapshot 28
   - Random removal (50 seeds)
3. Forward out-of-sample disruption on G^(29):
   - Predicted Future SRI (forecasted by Temporal GNN from snapshot 28)
   - Observed SRI at snapshot 28 (Persistence)
   - Degree Centrality at snapshot 28
   - Betweenness Centrality at snapshot 28
   - Random removal (50 seeds)
4. Records initial graph size, initial GCC size, and normalized GCC fraction.
5. Saves machine-readable results to outputs/audit/fragility_results.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, ".")

import numpy as np
import networkx as nx
import torch

from src.data.loader import load_raw_vectors, load_node_labels, parse_simplex_events
from src.data.snapshots import build_graph_snapshots
from src.graph.cliques import extract_maximal_cliques
from src.graph.scores import compute_clique_tensions, compute_silo_risk_index
from src.graph.meta_graph import build_clique_meta_graph
from src.models.temporal_gnn import TemporalGNN
from src.training.trainer import build_forecasting_targets


def simulate_node_removal(G: nx.Graph, ranking: List[int], fractions: np.ndarray) -> List[float]:
    initial_nodes = list(G.nodes())
    N_total = len(initial_nodes)
    if N_total == 0:
        return [0.0] * len(fractions)

    initial_sub = G.subgraph(initial_nodes)
    initial_gcc = len(max(nx.connected_components(initial_sub), key=len)) if initial_sub.number_of_nodes() > 0 else 1

    gcc_fractions = []
    for frac in fractions:
        k_rem = int(round(frac * N_total))
        nodes_to_remove = set(ranking[:k_rem])
        remaining = [n for n in initial_nodes if n not in nodes_to_remove]
        if remaining:
            sub = G.subgraph(remaining)
            gcc = len(max(nx.connected_components(sub), key=len)) if sub.number_of_nodes() > 0 else 0
        else:
            gcc = 0
        gcc_fractions.append(gcc / float(initial_gcc))
    return gcc_fractions


def run_fragility_audit(output_dir: str = "outputs/audit", num_random_seeds: int = 50):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" AUDITING NETWORK FRAGILITY SIMULATION")
    print(f" Random removal seeds: {num_random_seeds}")
    print("=" * 80)

    # 1. Load data & snapshots
    nverts, simplices, times = load_raw_vectors("Dataset/email-Enron")
    labels = load_node_labels("Dataset/email-Enron")
    events = parse_simplex_events(nverts, simplices, times)
    snapshots = build_graph_snapshots(events, window_days=30.0, stride_days=30.0, sort_events=True)
    all_node_ids = sorted(labels.keys())

    meta_graphs = [build_clique_meta_graph(s, extract_maximal_cliques(s.graph, min_size=3, top_k=30)) for s in snapshots]
    targets = build_forecasting_targets(meta_graphs, delta=1, jaccard_threshold=0.5)

    # Train Temporal GNN on seed 42 to obtain predicted tensions
    torch.manual_seed(42)
    np.random.seed(42)
    valid_indices = [i for i, t in enumerate(targets) if t is not None]
    n_train = int(len(valid_indices) * 0.70)
    train_indices = valid_indices[:n_train]

    tgnn = TemporalGNN(in_features=5, hidden_dim=16, dropout=0.1, jaccard_threshold=0.5)
    opt = torch.optim.Adam(tgnn.parameters(), lr=0.01, weight_decay=1e-4)
    crit = torch.nn.MSELoss()

    for _ in range(35):
        tgnn.train()
        opt.zero_grad()
        p_m, p_h = None, None
        for idx in train_indices:
            m = meta_graphs[idx]
            t = targets[idx]
            preds, curr_h = tgnn.forward_snapshot(m, p_m, p_h)
            if t is not None and len(t) > 0 and len(preds) > 0:
                loss = crit(preds, t)
                loss.backward()
            p_m = m
            p_h = curr_h.detach() if curr_h.numel() > 0 else None
        opt.step()

    tgnn.eval()

    # Snapshot 28 analysis
    snap28 = snapshots[28]
    G28 = snap28.graph.copy()
    nodes28 = list(G28.nodes())
    N28 = len(nodes28)
    gcc28_init = len(max(nx.connected_components(G28), key=len))

    print(f"\nSnapshot 28 ({snap28.start_date_str[:10]} to {snap28.end_date_str[:10]}):")
    print(f"  - Total nodes: {N28}, Total edges: {G28.number_of_edges()}")
    print(f"  - Initial GCC size: {gcc28_init} ({gcc28_init / N28 * 100:.1f}%)")

    # Rankings on Snapshot 28
    obs_tensions_28 = compute_clique_tensions(G28, meta_graphs[28].cliques)
    sri_obs_28 = compute_silo_risk_index(meta_graphs[28].cliques, obs_tensions_28, nodes28)
    deg_28 = dict(G28.degree())
    bet_28 = nx.betweenness_centrality(G28)

    # Predicted SRI from Temporal GNN at snapshot 28
    with torch.no_grad():
        preds_28, _ = tgnn.forward_snapshot(meta_graphs[28])
        sri_pred_28 = tgnn.predict_node_sri(preds_28, meta_graphs[28], nodes28)

    ranking_obs_sri = sorted(nodes28, key=lambda n: sri_obs_28.get(n, 0.0), reverse=True)
    ranking_pred_sri = sorted(nodes28, key=lambda n: sri_pred_28.get(n, 0.0), reverse=True)
    ranking_deg = sorted(nodes28, key=lambda n: deg_28.get(n, 0), reverse=True)
    ranking_bet = sorted(nodes28, key=lambda n: bet_28.get(n, 0.0), reverse=True)

    fractions = np.linspace(0.0, 0.50, 21)

    # 1. In-Sample Disruption of G28
    gcc_obs_sri = simulate_node_removal(G28, ranking_obs_sri, fractions)
    gcc_pred_sri = simulate_node_removal(G28, ranking_pred_sri, fractions)
    gcc_deg = simulate_node_removal(G28, ranking_deg, fractions)
    gcc_bet = simulate_node_removal(G28, ranking_bet, fractions)

    # Random removal over 50 seeds
    rnd_gccs = []
    for s in range(num_random_seeds):
        rng = np.random.default_rng(s)
        rnd_order = list(rng.permutation(nodes28))
        rnd_gccs.append(simulate_node_removal(G28, rnd_order, fractions))
    rnd_gccs = np.array(rnd_gccs)
    rnd_mean = np.mean(rnd_gccs, axis=0)
    rnd_std = np.std(rnd_gccs, axis=0)

    # 2. Out-of-Sample Disruption of Future Graph G29
    snap29 = snapshots[29]
    G29 = snap29.graph.copy()
    nodes29 = list(G29.nodes())
    N29 = len(nodes29)
    gcc29_init = len(max(nx.connected_components(G29), key=len))

    # Evaluate how rankings from snapshot 28 disrupt the future graph G29
    ranking_obs_sri_on_29 = [n for n in ranking_obs_sri if n in G29] + [n for n in nodes29 if n not in ranking_obs_sri]
    ranking_pred_sri_on_29 = [n for n in ranking_pred_sri if n in G29] + [n for n in nodes29 if n not in ranking_pred_sri]
    ranking_deg_on_29 = [n for n in ranking_deg if n in G29] + [n for n in nodes29 if n not in ranking_deg]
    ranking_bet_on_29 = [n for n in ranking_bet if n in G29] + [n for n in nodes29 if n not in ranking_bet]

    gcc29_obs_sri = simulate_node_removal(G29, ranking_obs_sri_on_29, fractions)
    gcc29_pred_sri = simulate_node_removal(G29, ranking_pred_sri_on_29, fractions)
    gcc29_deg = simulate_node_removal(G29, ranking_deg_on_29, fractions)
    gcc29_bet = simulate_node_removal(G29, ranking_bet_on_29, fractions)

    rnd29_gccs = []
    for s in range(num_random_seeds):
        rng = np.random.default_rng(s + 100)
        rnd_order = list(rng.permutation(nodes29))
        rnd29_gccs.append(simulate_node_removal(G29, rnd_order, fractions))
    rnd29_gccs = np.array(rnd29_gccs)
    rnd29_mean = np.mean(rnd29_gccs, axis=0)
    rnd29_std = np.std(rnd29_gccs, axis=0)

    # Save to CSV
    csv_file = out_path / "fragility_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "target_graph", "fraction_removed", "strategy", "relative_gcc_size", "std_err"
        ])
        for i, frac in enumerate(fractions):
            # In-sample G28
            writer.writerow(["G_snapshot_28", round(frac, 4), "Observed SRI (In-Sample)", round(gcc_obs_sri[i], 4), 0.0])
            writer.writerow(["G_snapshot_28", round(frac, 4), "Predicted SRI (Temporal GNN)", round(gcc_pred_sri[i], 4), 0.0])
            writer.writerow(["G_snapshot_28", round(frac, 4), "Degree Centrality", round(gcc_deg[i], 4), 0.0])
            writer.writerow(["G_snapshot_28", round(frac, 4), "Betweenness Centrality", round(gcc_bet[i], 4), 0.0])
            writer.writerow(["G_snapshot_28", round(frac, 4), "Random Removal (50 seeds)", round(rnd_mean[i], 4), round(rnd_std[i], 4)])

            # Forward G29
            writer.writerow(["G_snapshot_29_future", round(frac, 4), "Observed SRI (Persistence)", round(gcc29_obs_sri[i], 4), 0.0])
            writer.writerow(["G_snapshot_29_future", round(frac, 4), "Predicted SRI (Temporal GNN)", round(gcc29_pred_sri[i], 4), 0.0])
            writer.writerow(["G_snapshot_29_future", round(frac, 4), "Degree Centrality", round(gcc29_deg[i], 4), 0.0])
            writer.writerow(["G_snapshot_29_future", round(frac, 4), "Betweenness Centrality", round(gcc29_bet[i], 4), 0.0])
            writer.writerow(["G_snapshot_29_future", round(frac, 4), "Random Removal (50 seeds)", round(rnd29_mean[i], 4), round(rnd29_std[i], 4)])

    print(f"Fragility simulation results saved to {csv_file}")

    print("\n" + "=" * 80)
    print(" IN-SAMPLE DISRUPTION OF G(28) AT 10% AND 20% REMOVAL:")
    print("=" * 80)
    idx_10 = int(np.argmin(np.abs(fractions - 0.10)))
    idx_20 = int(np.argmin(np.abs(fractions - 0.20)))
    print(f"{'Strategy':<30} | {'GCC at 10% Removal':<22} | {'GCC at 20% Removal':<22}")
    print("-" * 78)
    print(f"{'Observed SRI':<30} | {gcc_obs_sri[idx_10] * 100:<20.1f}% | {gcc_obs_sri[idx_20] * 100:<20.1f}%")
    print(f"{'Predicted SRI (T-GNN)':<30} | {gcc_pred_sri[idx_10] * 100:<20.1f}% | {gcc_pred_sri[idx_20] * 100:<20.1f}%")
    print(f"{'Degree Centrality':<30} | {gcc_deg[idx_10] * 100:<20.1f}% | {gcc_deg[idx_20] * 100:<20.1f}%")
    print(f"{'Betweenness Centrality':<30} | {gcc_bet[idx_10] * 100:<20.1f}% | {gcc_bet[idx_20] * 100:<20.1f}%")
    print(f"{'Random (50 seeds)':<30} | {rnd_mean[idx_10] * 100:<5.1f}% ± {rnd_std[idx_10] * 100:<12.1f}% | {rnd_mean[idx_20] * 100:<5.1f}% ± {rnd_std[idx_20] * 100:<12.1f}%")
    print("=" * 78)

    print("\n" + "=" * 80)
    print(" OUT-OF-SAMPLE FUTURE DISRUPTION OF G(29) AT 10% AND 20% REMOVAL:")
    print("=" * 80)
    print(f"{'Strategy':<30} | {'GCC at 10% Removal':<22} | {'GCC at 20% Removal':<22}")
    print("-" * 78)
    print(f"{'Observed SRI (Persistence)':<30} | {gcc29_obs_sri[idx_10] * 100:<20.1f}% | {gcc29_obs_sri[idx_20] * 100:<20.1f}%")
    print(f"{'Predicted SRI (T-GNN)':<30} | {gcc29_pred_sri[idx_10] * 100:<20.1f}% | {gcc29_pred_sri[idx_20] * 100:<20.1f}%")
    print(f"{'Degree Centrality':<30} | {gcc29_deg[idx_10] * 100:<20.1f}% | {gcc29_deg[idx_20] * 100:<20.1f}%")
    print(f"{'Betweenness Centrality':<30} | {gcc29_bet[idx_10] * 100:<20.1f}% | {gcc29_bet[idx_20] * 100:<20.1f}%")
    print(f"{'Random (50 seeds)':<30} | {rnd29_mean[idx_10] * 100:<5.1f}% ± {rnd29_std[idx_10] * 100:<12.1f}% | {rnd29_mean[idx_20] * 100:<5.1f}% ± {rnd29_std[idx_20] * 100:<12.1f}%")
    print("=" * 78)


if __name__ == "__main__":
    run_fragility_audit()
