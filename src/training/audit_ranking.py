#!/usr/bin/env python3
"""
Comprehensive Audit of Node-Level Silo Risk Index (SRI) Ranking.

Investigates and validates:
1. Target alignment:
   - Projected Target SRI (future tension projected through current snapshot cliques)
   - Actual Ground-Truth SRI (observed directly on next snapshot graph G^(k+1))
2. Evaluation populations:
   - All 148 employee IDs (including inactive and degree-0 isolates)
   - Active employees facing non-zero risk (true SRI > 0 or predicted SRI > 0)
3. Full baseline comparison:
   - Persistence Observed SRI
   - Historical Mean SRI
   - Degree Centrality
   - Betweenness Centrality
   - Feature-Only MLP Predicted SRI
   - Static GCN Predicted SRI
   - Temporal GNN (No GRU Ablation)
   - Temporal GNN (Cold-Start GRU)
   - Temporal GNN (Warm-Start GRU)
   - Two-Stage Temporal GNN (Hurdle: P(persist) * T_cond)
4. Statistical metrics:
   - Spearman rho (with two-sided p-value and exact tie handling)
   - Pearson correlation r
   - MAE and MSE
   - Top-5 and Top-10 overlap

Outputs:
- outputs/audit/ranking_evaluation.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parent.parent.parent
if not (project_root / "src").exists():
    project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import scipy.stats as stats
import networkx as nx
import torch

from src.data.loader import load_raw_vectors, load_node_labels, parse_simplex_events
from src.data.snapshots import build_graph_snapshots
from src.graph.cliques import extract_maximal_cliques
from src.graph.meta_graph import build_clique_meta_graph
from src.graph.scores import compute_clique_tensions, compute_silo_risk_index
from src.models.temporal_gnn import TemporalGNN
from src.models.baselines import (
    FeatureOnlyMLP,
    HistoricalMeanBaseline,
    PersistenceBaseline,
    StaticGCNModel,
    TemporalGNNAblationNoGRU,
)
from src.training.metrics import compute_ranking_metrics, compute_comprehensive_ranking_metrics
from src.training.trainer import build_two_stage_targets


def run_ranking_audit(output_dir: str = "outputs/audit", seed: int = 42) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 105)
    print(" COMPREHENSIVE AUDIT OF NODE-LEVEL SRI RANKING PROTOCOLS AND BASELINES")
    print(f" Random Seed: {seed} | Evaluation Split: Test Snapshots 36 to 42")
    print("=" * 105)

    # 1. Load data & construct snapshots and meta-graphs
    nverts, simplices, times = load_raw_vectors("Dataset/email-Enron")
    labels = load_node_labels("Dataset/email-Enron")
    events = parse_simplex_events(nverts, simplices, times)
    snapshots = build_graph_snapshots(events, window_days=30.0, stride_days=30.0, sort_events=True)
    all_node_ids = sorted(labels.keys())

    meta_graphs = [
        build_clique_meta_graph(s, extract_maximal_cliques(s.graph, min_size=3, top_k=30))
        for s in snapshots
    ]

    targets_bundle = build_two_stage_targets(
        meta_graphs=meta_graphs,
        delta=1,
        jaccard_threshold=0.5,
        matching_method="greedy",
    )
    z_targets = targets_bundle.zero_fill_targets
    p_targets = targets_bundle.persistence_targets
    c_targets = targets_bundle.conditional_targets
    p_masks = targets_bundle.persisting_masks

    valid_indices = [i for i, t in enumerate(z_targets) if t is not None]
    n_train = int(len(valid_indices) * 0.70)
    n_val = int(len(valid_indices) * 0.15)

    train_indices = valid_indices[:n_train]
    val_indices = valid_indices[n_train : n_train + n_val]
    test_indices = valid_indices[n_train + n_val :]

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2. Train candidate neural models on training split
    crit_mse = torch.nn.MSELoss()
    crit_bce = torch.nn.BCEWithLogitsLoss()

    # Model A: Feature-Only MLP (Single-stage)
    mlp_m = FeatureOnlyMLP(in_features=5, hidden_dim=16)
    mlp_opt = torch.optim.Adam(mlp_m.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(35):
        mlp_m.train()
        mlp_opt.zero_grad()
        for idx in train_indices:
            m = meta_graphs[idx]
            t = z_targets[idx]
            if t is not None and len(t) > 0 and m.num_cliques > 0:
                loss = crit_mse(mlp_m(m.features), t)
                loss.backward()
        mlp_opt.step()
    mlp_m.eval()

    # Model B: Static GCN
    sgcn_m = StaticGCNModel(in_features=5, hidden_dim=16, dropout=0.1)
    sgcn_opt = torch.optim.Adam(sgcn_m.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(35):
        sgcn_m.train()
        sgcn_opt.zero_grad()
        for idx in train_indices:
            m = meta_graphs[idx]
            t = z_targets[idx]
            if t is not None and len(t) > 0 and m.num_cliques > 0:
                loss = crit_mse(sgcn_m(m), t)
                loss.backward()
        sgcn_opt.step()
    sgcn_m.eval()

    # Model C: Temporal GNN (No GRU Ablation)
    ablation_m = TemporalGNNAblationNoGRU(in_features=5, hidden_dim=16, dropout=0.1)
    ablation_opt = torch.optim.Adam(ablation_m.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(35):
        ablation_m.train()
        ablation_opt.zero_grad()
        for idx in train_indices:
            m = meta_graphs[idx]
            t = z_targets[idx]
            if t is not None and len(t) > 0 and m.num_cliques > 0:
                loss = crit_mse(ablation_m(m), t)
                loss.backward()
        ablation_opt.step()
    ablation_m.eval()

    # Model D: Temporal GNN Full (Single-stage)
    tgnn_m = TemporalGNN(in_features=5, hidden_dim=16, dropout=0.1, jaccard_threshold=0.5, one_to_one_matching=True)
    tgnn_opt = torch.optim.Adam(tgnn_m.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(35):
        tgnn_m.train()
        tgnn_opt.zero_grad()
        p_m, p_h = None, None
        for idx in train_indices:
            m = meta_graphs[idx]
            t = z_targets[idx]
            preds, curr_h = tgnn_m.forward_snapshot(m, p_m, p_h)
            if t is not None and len(t) > 0 and len(preds) > 0:
                loss = crit_mse(preds, t)
                loss.backward()
            p_m = m
            p_h = curr_h.detach() if (curr_h is not None and curr_h.numel() > 0) else None
        tgnn_opt.step()
    tgnn_m.eval()

    # Model E: Two-Stage Temporal GNN
    ts_tgnn_m = TemporalGNN(in_features=5, hidden_dim=16, dropout=0.1, jaccard_threshold=0.5, one_to_one_matching=True)
    ts_opt = torch.optim.Adam(ts_tgnn_m.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(35):
        ts_tgnn_m.train()
        ts_opt.zero_grad()
        p_m, p_h = None, None
        for idx in train_indices:
            m = meta_graphs[idx]
            p_t = p_targets[idx]
            c_t = c_targets[idx]
            mask = p_masks[idx]
            if p_t is not None and len(p_t) > 0 and m.num_cliques > 0:
                exp_t, p_prob, cond_t, curr_h = ts_tgnn_m.forward_two_stage(m, p_m, p_h)
                p_logits = ts_tgnn_m.persistence_head(curr_h).squeeze(-1)
                loss_p = crit_bce(p_logits, p_t)
                loss_c = crit_mse(cond_t[mask], c_t[mask]) if (mask is not None and mask.sum() > 0) else torch.tensor(0.0)
                loss = loss_p + loss_c
                loss.backward()
            else:
                curr_h = None
            p_m = m
            p_h = curr_h.detach() if (curr_h is not None and curr_h.numel() > 0) else None
        ts_opt.step()
    ts_tgnn_m.eval()

    # Baselines: Historical Mean
    hm_baseline = HistoricalMeanBaseline()
    for idx in train_indices:
        hm_baseline.update(meta_graphs[idx])

    # 3. Snapshot-by-Snapshot Ranking Evaluation
    ranking_records = []

    # Warm-up states for warm-start models
    w_meta, w_h = None, None
    for w_idx in range(0, test_indices[0]):
        m_w = meta_graphs[w_idx]
        _, w_h = tgnn_m.forward_snapshot(m_w, w_meta, w_h)
        w_meta = m_w

    ts_w_meta, ts_w_h = None, None
    for w_idx in range(0, test_indices[0]):
        m_w = meta_graphs[w_idx]
        _, _, _, ts_w_h = ts_tgnn_m.forward_two_stage(m_w, ts_w_meta, ts_w_h)
        ts_w_meta = m_w

    with torch.no_grad():
        cold_p_m, cold_p_h = None, None
        warm_p_m, warm_p_h = w_meta, w_h
        ts_p_m, ts_p_h = ts_w_meta, ts_w_h

        for s_idx in test_indices:
            snap = snapshots[s_idx]
            meta = meta_graphs[s_idx]
            target_z = z_targets[s_idx]

            if target_z is None or len(target_z) == 0:
                continue

            # Target 1: Projected future SRI (target tensions mapped through snapshot s_idx cliques)
            true_sri_proj = tgnn_m.predict_node_sri(target_z, meta, all_node_ids)

            # Target 2: Actual observed SRI on next snapshot graph G^(s_idx + 1)
            if s_idx + 1 < len(snapshots):
                next_snap = snapshots[s_idx + 1]
                next_meta = meta_graphs[s_idx + 1]
                true_sri_actual = compute_silo_risk_index(
                    next_meta.cliques,
                    compute_clique_tensions(next_snap.graph, next_meta.cliques),
                    all_node_ids,
                )
            else:
                true_sri_actual = {nid: 0.0 for nid in all_node_ids}

            # Predictions from each model/baseline
            # 1. Persistence Observed SRI
            sri_persistence = tgnn_m.predict_node_sri(meta.clique_tensions, meta, all_node_ids)

            # 2. Historical Mean Baseline SRI
            hm_preds = hm_baseline.predict(meta)
            sri_hm = tgnn_m.predict_node_sri(hm_preds, meta, all_node_ids)

            # 3. Degree Centrality
            deg_cent = nx.degree_centrality(snap.graph)
            sri_deg = {nid: deg_cent.get(nid, 0.0) for nid in all_node_ids}

            # 4. Betweenness Centrality
            bet_cent = nx.betweenness_centrality(snap.graph)
            sri_bet = {nid: bet_cent.get(nid, 0.0) for nid in all_node_ids}

            # 5. Feature-Only MLP
            mlp_preds = mlp_m(meta.features)
            sri_mlp = tgnn_m.predict_node_sri(mlp_preds, meta, all_node_ids)

            # 6. Static GCN
            sgcn_preds = sgcn_m(meta)
            sri_sgcn = tgnn_m.predict_node_sri(sgcn_preds, meta, all_node_ids)

            # 7. Temporal GNN (No GRU Ablation)
            ablation_preds = ablation_m(meta)
            sri_ablation = tgnn_m.predict_node_sri(ablation_preds, meta, all_node_ids)

            # 8. Temporal GNN (Cold-Start GRU)
            cold_preds, curr_cold_h = tgnn_m.forward_snapshot(meta, cold_p_m, cold_p_h)
            cold_p_m, cold_p_h = meta, curr_cold_h
            sri_cold = tgnn_m.predict_node_sri(cold_preds, meta, all_node_ids)

            # 9. Temporal GNN (Warm-Start GRU)
            warm_preds, curr_warm_h = tgnn_m.forward_snapshot(meta, warm_p_m, warm_p_h)
            warm_p_m, warm_p_h = meta, curr_warm_h
            sri_warm = tgnn_m.predict_node_sri(warm_preds, meta, all_node_ids)

            # 10. Two-Stage Hurdle Temporal GNN (Warm-Start)
            ts_exp, ts_prob, ts_cond, curr_ts_h = ts_tgnn_m.forward_two_stage(meta, ts_p_m, ts_p_h)
            ts_p_m, ts_p_h = meta, curr_ts_h
            sri_two_stage = tgnn_m.predict_node_sri(ts_exp, meta, all_node_ids)

            strategies = {
                "Persistence Observed SRI": sri_persistence,
                "Historical Mean SRI": sri_hm,
                "Degree Centrality": sri_deg,
                "Betweenness Centrality": sri_bet,
                "Feature-Only MLP Predicted SRI": sri_mlp,
                "Static GCN Predicted SRI": sri_sgcn,
                "Temporal GNN (No GRU) Predicted SRI": sri_ablation,
                "Temporal GNN (Cold-Start GRU) Predicted SRI": sri_cold,
                "Temporal GNN (Warm-Start GRU) Predicted SRI": sri_warm,
                "Two-Stage Temporal GNN Predicted SRI": sri_two_stage,
            }

            date_str = f"{snap.start_date_str[:10]} to {snap.end_date_str[:10]}"

            for strat_name, p_dict in strategies.items():
                m_proj_all = compute_ranking_metrics(true_sri_proj, p_dict, top_k=10, active_only=False)
                m_proj_act = compute_ranking_metrics(true_sri_proj, p_dict, top_k=10, active_only=True)
                m_proj_t5 = compute_ranking_metrics(true_sri_proj, p_dict, top_k=5, active_only=False)

                m_act_all = compute_ranking_metrics(true_sri_actual, p_dict, top_k=10, active_only=False)
                m_act_act = compute_ranking_metrics(true_sri_actual, p_dict, top_k=10, active_only=True)

                ranking_records.append({
                    "snapshot_id": s_idx,
                    "date_range": date_str,
                    "strategy": strat_name,
                    "num_cliques": meta.num_cliques,
                    "spearman_rho_all": m_proj_all["spearman_rho"],
                    "p_value_all": m_proj_all["p_value"],
                    "pearson_r_all": m_proj_all["pearson_r"],
                    "mae_all": m_proj_all["mae"],
                    "top10_overlap_all": m_proj_all["top_k_overlap"],
                    "top5_overlap_all": m_proj_t5["top_k_overlap"],
                    "spearman_rho_active": m_proj_act["spearman_rho"],
                    "p_value_active": m_proj_act["p_value"],
                    "pearson_r_active": m_proj_act["pearson_r"],
                    "mae_active": m_proj_act["mae"],
                    "top10_overlap_active": m_proj_act["top_k_overlap"],
                    "num_active_nodes": m_proj_act["num_nodes"],
                    "actual_next_spearman_all": m_act_all["spearman_rho"],
                    "actual_next_spearman_active": m_act_act["spearman_rho"],
                })

    # Save to CSV
    csv_file = out_path / "ranking_evaluation.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "snapshot_id", "date_range", "strategy", "num_cliques",
            "spearman_rho_all", "p_value_all", "pearson_r_all", "mae_all", "top10_overlap_all", "top5_overlap_all",
            "spearman_rho_active", "p_value_active", "pearson_r_active", "mae_active", "top10_overlap_active", "num_active_nodes",
            "actual_next_spearman_all", "actual_next_spearman_active",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ranking_records:
            writer.writerow(r)

    print(f"\nDetailed snapshot ranking records saved to: {csv_file}")

    # Summary table per strategy
    print("\n" + "=" * 125)
    print(" SUMMARY OF NODE SRI RANKING BENCHMARK (Across Test Snapshots 36 to 42):")
    print("=" * 125)
    print(f"{'Strategy':<42} | {'rho (All 148)':<14} | {'rho (Active)':<14} | {'Top-5 Overlap':<14} | {'Top-10 Overlap':<14} | {'Actual Next rho'}")
    print("-" * 125)

    unique_strats = [
        "Persistence Observed SRI",
        "Historical Mean SRI",
        "Degree Centrality",
        "Betweenness Centrality",
        "Feature-Only MLP Predicted SRI",
        "Static GCN Predicted SRI",
        "Temporal GNN (No GRU) Predicted SRI",
        "Temporal GNN (Cold-Start GRU) Predicted SRI",
        "Temporal GNN (Warm-Start GRU) Predicted SRI",
        "Two-Stage Temporal GNN Predicted SRI",
    ]

    for strat in unique_strats:
        s_recs = [r for r in ranking_records if r["strategy"] == strat]
        if not s_recs:
            continue
        rho_all = np.mean([r["spearman_rho_all"] for r in s_recs])
        rho_act = np.mean([r["spearman_rho_active"] for r in s_recs])
        top5 = np.mean([r["top5_overlap_all"] for r in s_recs])
        top10 = np.mean([r["top10_overlap_all"] for r in s_recs])
        act_rho = np.mean([r["actual_next_spearman_active"] for r in s_recs])
        print(f"{strat:<42} | {rho_all:>7.4f}        | {rho_act:>7.4f}        | {top5 * 100:>6.1f}%        | {top10 * 100:>6.1f}%        | {act_rho:>7.4f}")
    print("=" * 125)


if __name__ == "__main__":
    run_ranking_audit()
