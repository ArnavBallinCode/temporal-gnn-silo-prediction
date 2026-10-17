#!/usr/bin/env python3
"""
Multi-Seed Scientific Benchmark & Ablation Audit.

Evaluates 11 model configurations across 5 independent random seeds:
Baselines:
1. Persistence Baseline
2. Historical Mean Baseline

Single-Stage Zero-Fill Formulations (Protocol A Baseline):
3. Feature-Only MLP
4. Static GCN
5. Temporal GNN (No GRU Ablation)
6. Temporal GNN (Full GCN + GRU, Cold-Start)
7. Temporal GNN (Full GCN + GRU, Warm-Start)

Revised Two-Stage Hurdle Formulations:
8. Feature-Only MLP (Two-Stage)
9. Static GCN (Two-Stage)
10. Temporal GNN (No GRU Two-Stage)
11. Temporal GNN (Full GCN + GRU Two-Stage Warm-Start)

Evaluates under:
- Protocol A: All test cliques (N = 109, unmatched = 0.0)
- Protocol B: Persisting test cliques (N = 35, 1-to-1 J >= 0.5)

Outputs:
- outputs/audit/benchmark_results.json
- outputs/audit/benchmark_results.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys

project_root = Path(__file__).resolve().parent.parent.parent
if not (project_root / "src").exists():
    project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import load_node_labels, load_raw_vectors, parse_simplex_events
from src.data.snapshots import build_graph_snapshots
from src.graph.cliques import extract_maximal_cliques, match_cliques_bipartite
from src.graph.meta_graph import build_clique_meta_graph
from src.models.baselines import (
    FeatureOnlyMLP,
    HistoricalMeanBaseline,
    PersistenceBaseline,
    StaticGCNModel,
    TemporalGNNAblationNoGRU,
)
from src.models.temporal_gnn import TemporalGNN
from src.training.metrics import (
    compute_ranking_metrics,
    compute_regression_metrics,
    compute_comprehensive_ranking_metrics,
)
from src.training.trainer import (
    build_two_stage_targets,
    build_forecasting_targets,
)


def run_multi_seed_benchmark(
    seeds: List[int] = [42, 123, 456, 789, 999],
    epochs: int = 35,
    lr: float = 0.01,
    hidden_dim: int = 16,
    output_dir: str = "outputs/audit",
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 95)
    print(" MULTI-SEED SCIENTIFIC BENCHMARK & ABLATION AUDIT (REVISED METHODOLOGY)")
    print(f" Seeds: {seeds} | Epochs: {epochs} | LR: {lr} | Hidden Dim: {hidden_dim}")
    print("=" * 95)

    # 1. Load data & construct snapshots and meta-graphs
    nverts, simplices, times = load_raw_vectors("Dataset/email-Enron")
    labels = load_node_labels("Dataset/email-Enron")
    events = parse_simplex_events(nverts, simplices, times)
    snapshots = build_graph_snapshots(events, window_days=30.0, stride_days=30.0, sort_events=True)
    all_node_ids = sorted(labels.keys())

    # Extract all maximal cliques (untruncated) for disappearance audit
    all_snapshots_cliques = []
    for s in snapshots:
        raw_c = extract_maximal_cliques(s.graph, min_size=3, top_k=None)
        all_snapshots_cliques.append(raw_c)

    meta_graphs = [
        build_clique_meta_graph(s, extract_maximal_cliques(s.graph, min_size=3, top_k=30))
        for s in snapshots
    ]

    # Build targets with 1-to-1 bipartite matching
    targets_bundle = build_two_stage_targets(
        meta_graphs=meta_graphs,
        all_snapshots_cliques=all_snapshots_cliques,
        delta=1,
        jaccard_threshold=0.5,
        matching_method="greedy",
    )

    p_targets = targets_bundle.persistence_targets
    c_targets = targets_bundle.conditional_targets
    z_targets = targets_bundle.zero_fill_targets
    p_masks = targets_bundle.persisting_masks

    valid_indices = [i for i, t in enumerate(z_targets) if t is not None]
    n_train = int(len(valid_indices) * 0.70)
    n_val = int(len(valid_indices) * 0.15)

    train_indices = valid_indices[:n_train]          # 0 to 29
    val_indices = valid_indices[n_train : n_train + n_val]  # 30 to 35
    test_indices = valid_indices[n_train + n_val :]         # 36 to 42

    # Map test persisting mask
    test_clique_counts = sum(meta_graphs[idx].num_cliques for idx in test_indices)
    test_persisting_counts = sum(
        p_masks[idx].sum().item() for idx in test_indices if p_masks[idx] is not None
    )

    print(f"Total test cliques: {test_clique_counts} (Protocol A: {test_clique_counts}, Protocol B: {test_persisting_counts})")

    # Define all model keys
    model_names = [
        "Persistence Baseline",
        "Historical Mean Baseline",
        "Feature-Only MLP",
        "Static GCN",
        "Temporal GNN (No GRU Ablation)",
        "Temporal GNN (Full GCN + GRU)",
        "Temporal GNN (Warm-Start GRU)",
        "Feature-Only MLP (Two-Stage)",
        "Static GCN (Two-Stage)",
        "Temporal GNN No-GRU (Two-Stage)",
        "Temporal GNN Full (Two-Stage Warm-Start)",
    ]

    results_A: Dict[str, Dict[str, List[float]]] = {
        m: {
            "mse": [], "rmse": [], "mae": [], "pearson_r": [],
            "spearman_rho": [], "spearman_rho_active": [],
            "top5": [], "top10": [],
        }
        for m in model_names
    }

    results_B: Dict[str, Dict[str, List[float]]] = {
        m: {"mse": [], "rmse": [], "mae": [], "pearson_r": []}
        for m in model_names
    }

    mse_crit = nn.MSELoss()
    bce_crit = nn.BCEWithLogitsLoss()

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # -------------------------------------------------------------
        # Model 1: Persistence Baseline
        # -------------------------------------------------------------
        pers_model = PersistenceBaseline()
        p_trues_A, p_preds_A = [], []
        p_trues_B, p_preds_B = [], []
        p_rhos_all, p_rhos_act, p_t5, p_t10 = [], [], [] ,[]

        for idx in test_indices:
            meta = meta_graphs[idx]
            z_t = z_targets[idx]
            c_t = c_targets[idx]
            mask = p_masks[idx]

            if z_t is not None and len(z_t) > 0 and meta.num_cliques > 0:
                pred = pers_model.predict(meta)
                p_preds_A.extend(pred.tolist())
                p_trues_A.extend(z_t.tolist())

                if mask is not None:
                    for i in range(meta.num_cliques):
                        if mask[i].item():
                            p_preds_B.append(float(pred[i].item()))
                            p_trues_B.append(float(c_t[i].item()))

                pred_sri = {nid: sum(float(pred[c].item()) for c in meta.node_to_clique_map.get(nid, []) if c < len(pred)) for nid in all_node_ids}
                true_sri = {nid: sum(float(z_t[c].item()) for c in meta.node_to_clique_map.get(nid, []) if c < len(z_t)) for nid in all_node_ids}
                m_rank = compute_comprehensive_ranking_metrics(true_sri, pred_sri, top_k_list=[5, 10], all_node_ids=all_node_ids)
                p_rhos_all.append(m_rank["all_nodes"]["spearman_rho"])
                p_rhos_act.append(m_rank["active_nodes"]["spearman_rho"])
                p_t5.append(m_rank["all_nodes"]["top_k_overlaps"]["top_5"])
                p_t10.append(m_rank["all_nodes"]["top_k_overlaps"]["top_10"])

        mA_p = compute_regression_metrics(np.array(p_trues_A), np.array(p_preds_A))
        mB_p = compute_regression_metrics(np.array(p_trues_B), np.array(p_preds_B))
        results_A["Persistence Baseline"]["mse"].append(mA_p["mse"])
        results_A["Persistence Baseline"]["rmse"].append(mA_p["rmse"])
        results_A["Persistence Baseline"]["mae"].append(mA_p["mae"])
        results_A["Persistence Baseline"]["pearson_r"].append(mA_p["pearson_r"])
        results_A["Persistence Baseline"]["spearman_rho"].append(float(np.mean(p_rhos_all)))
        results_A["Persistence Baseline"]["spearman_rho_active"].append(float(np.mean(p_rhos_act)))
        results_A["Persistence Baseline"]["top5"].append(float(np.mean(p_t5)))
        results_A["Persistence Baseline"]["top10"].append(float(np.mean(p_t10)))

        results_B["Persistence Baseline"]["mse"].append(mB_p["mse"])
        results_B["Persistence Baseline"]["rmse"].append(mB_p["rmse"])
        results_B["Persistence Baseline"]["mae"].append(mB_p["mae"])
        results_B["Persistence Baseline"]["pearson_r"].append(mB_p["pearson_r"])

        # -------------------------------------------------------------
        # Model 2: Historical Mean Baseline
        # -------------------------------------------------------------
        hm_model = HistoricalMeanBaseline()
        for idx in train_indices:
            hm_model.update(meta_graphs[idx])

        hm_trues_A, hm_preds_A = [], []
        hm_trues_B, hm_preds_B = [], []
        hm_rhos_all, hm_rhos_act, hm_t5, hm_t10 = [], [], [] ,[]

        for idx in test_indices:
            meta = meta_graphs[idx]
            z_t = z_targets[idx]
            c_t = c_targets[idx]
            mask = p_masks[idx]

            if z_t is not None and len(z_t) > 0 and meta.num_cliques > 0:
                pred = hm_model.predict(meta)
                hm_preds_A.extend(pred.tolist())
                hm_trues_A.extend(z_t.tolist())

                if mask is not None:
                    for i in range(meta.num_cliques):
                        if mask[i].item():
                            hm_preds_B.append(float(pred[i].item()))
                            hm_trues_B.append(float(c_t[i].item()))

                pred_sri = {nid: sum(float(pred[c].item()) for c in meta.node_to_clique_map.get(nid, []) if c < len(pred)) for nid in all_node_ids}
                true_sri = {nid: sum(float(z_t[c].item()) for c in meta.node_to_clique_map.get(nid, []) if c < len(z_t)) for nid in all_node_ids}
                m_rank = compute_comprehensive_ranking_metrics(true_sri, pred_sri, top_k_list=[5, 10], all_node_ids=all_node_ids)
                hm_rhos_all.append(m_rank["all_nodes"]["spearman_rho"])
                hm_rhos_act.append(m_rank["active_nodes"]["spearman_rho"])
                hm_t5.append(m_rank["all_nodes"]["top_k_overlaps"]["top_5"])
                hm_t10.append(m_rank["all_nodes"]["top_k_overlaps"]["top_10"])

        mA_hm = compute_regression_metrics(np.array(hm_trues_A), np.array(hm_preds_A))
        mB_hm = compute_regression_metrics(np.array(hm_trues_B), np.array(hm_preds_B))
        results_A["Historical Mean Baseline"]["mse"].append(mA_hm["mse"])
        results_A["Historical Mean Baseline"]["rmse"].append(mA_hm["rmse"])
        results_A["Historical Mean Baseline"]["mae"].append(mA_hm["mae"])
        results_A["Historical Mean Baseline"]["pearson_r"].append(mA_hm["pearson_r"])
        results_A["Historical Mean Baseline"]["spearman_rho"].append(float(np.mean(hm_rhos_all)))
        results_A["Historical Mean Baseline"]["spearman_rho_active"].append(float(np.mean(hm_rhos_act)))
        results_A["Historical Mean Baseline"]["top5"].append(float(np.mean(hm_t5)))
        results_A["Historical Mean Baseline"]["top10"].append(float(np.mean(hm_t10)))

        results_B["Historical Mean Baseline"]["mse"].append(mB_hm["mse"])
        results_B["Historical Mean Baseline"]["rmse"].append(mB_hm["rmse"])
        results_B["Historical Mean Baseline"]["mae"].append(mB_hm["mae"])
        results_B["Historical Mean Baseline"]["pearson_r"].append(mB_hm["pearson_r"])

        # -------------------------------------------------------------
        # Helper for Neural Model Training (Single-Stage Zero-Fill)
        # -------------------------------------------------------------
        def train_and_eval_single_stage(model_obj, model_key, is_recurrent=False, warm_start=False):
            optimizer = optim.Adam(model_obj.parameters(), lr=lr, weight_decay=1e-4)
            best_val = float("inf")
            best_state = None

            for _ in range(epochs):
                model_obj.train()
                optimizer.zero_grad()
                p_meta, p_h = None, None
                for idx in train_indices:
                    m = meta_graphs[idx]
                    t = z_targets[idx]
                    preds, curr_h = model_obj.forward_snapshot(m, p_meta, p_h) if is_recurrent else (model_obj(m.features) if isinstance(model_obj, FeatureOnlyMLP) else model_obj(m), None)
                    if t is not None and len(t) > 0 and len(preds) > 0:
                        loss = mse_crit(preds, t)
                        loss.backward()
                    if is_recurrent:
                        p_meta = m
                        p_h = curr_h.detach() if (curr_h is not None and curr_h.numel() > 0) else None
                optimizer.step()

                # Validation step
                model_obj.eval()
                val_loss = 0.0
                v_count = 0
                with torch.no_grad():
                    vp_meta, vp_h = None, None
                    if is_recurrent and warm_start:
                        for w_idx in train_indices:
                            w_m = meta_graphs[w_idx]
                            _, w_h = model_obj.forward_snapshot(w_m, vp_meta, vp_h)
                            vp_meta, vp_h = w_m, w_h
                    for idx in val_indices:
                        m = meta_graphs[idx]
                        t = z_targets[idx]
                        preds, curr_h = model_obj.forward_snapshot(m, vp_meta, vp_h) if is_recurrent else (model_obj(m.features) if isinstance(model_obj, FeatureOnlyMLP) else model_obj(m), None)
                        if t is not None and len(t) > 0 and len(preds) > 0:
                            val_loss += mse_crit(preds, t).item()
                            v_count += 1
                        if is_recurrent:
                            vp_meta = m
                            vp_h = curr_h

                avg_v = val_loss / v_count if v_count else 0.0
                if avg_v < best_val:
                    best_val = avg_v
                    best_state = {k: v.cpu().clone() for k, v in model_obj.state_dict().items()}

            model_obj.load_state_dict(best_state)
            model_obj.eval()

            # Testing
            nn_trues_A, nn_preds_A = [], []
            nn_trues_B, nn_preds_B = [], []
            nn_rhos_all, nn_rhos_act, nn_t5, nn_t10 = [], [], [] ,[]

            with torch.no_grad():
                tp_meta, tp_h = None, None
                if is_recurrent and warm_start:
                    for w_idx in range(0, test_indices[0]):
                        w_m = meta_graphs[w_idx]
                        _, w_h = model_obj.forward_snapshot(w_m, tp_meta, tp_h)
                        tp_meta, tp_h = w_m, w_h

                for idx in test_indices:
                    m = meta_graphs[idx]
                    z_t = z_targets[idx]
                    c_t = c_targets[idx]
                    mask = p_masks[idx]

                    preds, curr_h = model_obj.forward_snapshot(m, tp_meta, tp_h) if is_recurrent else (model_obj(m.features) if isinstance(model_obj, FeatureOnlyMLP) else model_obj(m), None)

                    if z_t is not None and len(z_t) > 0 and len(preds) > 0:
                        nn_preds_A.extend(preds.tolist())
                        nn_trues_A.extend(z_t.tolist())

                        if mask is not None:
                            for i in range(m.num_cliques):
                                if mask[i].item():
                                    nn_preds_B.append(float(preds[i].item()))
                                    nn_trues_B.append(float(c_t[i].item()))

                        pred_sri = {nid: sum(float(preds[c].item()) for c in m.node_to_clique_map.get(nid, []) if c < len(preds)) for nid in all_node_ids}
                        true_sri = {nid: sum(float(z_t[c].item()) for c in m.node_to_clique_map.get(nid, []) if c < len(z_t)) for nid in all_node_ids}
                        m_rank = compute_comprehensive_ranking_metrics(true_sri, pred_sri, top_k_list=[5, 10], all_node_ids=all_node_ids)
                        nn_rhos_all.append(m_rank["all_nodes"]["spearman_rho"])
                        nn_rhos_act.append(m_rank["active_nodes"]["spearman_rho"])
                        nn_t5.append(m_rank["all_nodes"]["top_k_overlaps"]["top_5"])
                        nn_t10.append(m_rank["all_nodes"]["top_k_overlaps"]["top_10"])

                    if is_recurrent:
                        tp_meta = m
                        tp_h = curr_h

            mA = compute_regression_metrics(np.array(nn_trues_A), np.array(nn_preds_A))
            mB = compute_regression_metrics(np.array(nn_trues_B), np.array(nn_preds_B))
            results_A[model_key]["mse"].append(mA["mse"])
            results_A[model_key]["rmse"].append(mA["rmse"])
            results_A[model_key]["mae"].append(mA["mae"])
            results_A[model_key]["pearson_r"].append(mA["pearson_r"])
            results_A[model_key]["spearman_rho"].append(float(np.mean(nn_rhos_all)))
            results_A[model_key]["spearman_rho_active"].append(float(np.mean(nn_rhos_act)))
            results_A[model_key]["top5"].append(float(np.mean(nn_t5)))
            results_A[model_key]["top10"].append(float(np.mean(nn_t10)))

            results_B[model_key]["mse"].append(mB["mse"])
            results_B[model_key]["rmse"].append(mB["rmse"])
            results_B[model_key]["mae"].append(mB["mae"])
            results_B[model_key]["pearson_r"].append(mB["pearson_r"])

        # Train Single-Stage Models
        mlp_m = FeatureOnlyMLP(in_features=5, hidden_dim=hidden_dim)
        train_and_eval_single_stage(mlp_m, "Feature-Only MLP", is_recurrent=False)

        sgcn_m = StaticGCNModel(in_features=5, hidden_dim=hidden_dim, dropout=0.1)
        train_and_eval_single_stage(sgcn_m, "Static GCN", is_recurrent=False)

        ablation_m = TemporalGNNAblationNoGRU(in_features=5, hidden_dim=hidden_dim, dropout=0.1)
        train_and_eval_single_stage(ablation_m, "Temporal GNN (No GRU Ablation)", is_recurrent=False)

        tgnn_cold_m = TemporalGNN(in_features=5, hidden_dim=hidden_dim, dropout=0.1, jaccard_threshold=0.5, one_to_one_matching=True)
        train_and_eval_single_stage(tgnn_cold_m, "Temporal GNN (Full GCN + GRU)", is_recurrent=True, warm_start=False)

        tgnn_warm_m = TemporalGNN(in_features=5, hidden_dim=hidden_dim, dropout=0.1, jaccard_threshold=0.5, one_to_one_matching=True)
        train_and_eval_single_stage(tgnn_warm_m, "Temporal GNN (Warm-Start GRU)", is_recurrent=True, warm_start=True)

        # -------------------------------------------------------------
        # Helper for Two-Stage Hurdle Model Training
        # -------------------------------------------------------------
        def train_and_eval_two_stage(model_obj, model_key, is_recurrent=False, warm_start=True):
            optimizer = optim.Adam(model_obj.parameters(), lr=lr, weight_decay=1e-4)
            best_val = float("inf")
            best_state = None

            for _ in range(epochs):
                model_obj.train()
                optimizer.zero_grad()
                p_meta, p_h = None, None
                for idx in train_indices:
                    m = meta_graphs[idx]
                    p_t = p_targets[idx]
                    c_t = c_targets[idx]
                    mask = p_masks[idx]

                    if p_t is not None and len(p_t) > 0 and m.num_cliques > 0:
                        if is_recurrent:
                            exp_t, p_prob, cond_t, curr_h = model_obj.forward_two_stage(m, p_meta, p_h)
                            p_logits = model_obj.persistence_head(curr_h).squeeze(-1)
                        elif isinstance(model_obj, FeatureOnlyMLP):
                            exp_t, p_prob, cond_t = model_obj.forward_two_stage(m.features)
                            h = model_obj.encoder(m.features)
                            p_logits = model_obj.persistence_head(h).squeeze(-1)
                            curr_h = None
                        else:
                            exp_t, p_prob, cond_t = model_obj.forward_two_stage(m)
                            emb = model_obj.spatial_gnn(m.features, m.edge_index, m.edge_weights)
                            p_logits = model_obj.persistence_head(emb).squeeze(-1)
                            curr_h = None

                        loss_p = bce_crit(p_logits, p_t)
                        if mask is not None and mask.sum() > 0:
                            loss_c = mse_crit(cond_t[mask], c_t[mask])
                        else:
                            loss_c = torch.tensor(0.0, device=m.features.device)

                        loss = loss_p + loss_c
                        loss.backward()
                    else:
                        curr_h = None

                    if is_recurrent:
                        p_meta = m
                        p_h = curr_h.detach() if (curr_h is not None and curr_h.numel() > 0) else None
                optimizer.step()

                # Validation
                model_obj.eval()
                val_loss = 0.0
                v_count = 0
                with torch.no_grad():
                    vp_meta, vp_h = None, None
                    if is_recurrent and warm_start:
                        for w_idx in train_indices:
                            w_m = meta_graphs[w_idx]
                            _, _, _, w_h = model_obj.forward_two_stage(w_m, vp_meta, vp_h)
                            vp_meta, vp_h = w_m, w_h

                    for idx in val_indices:
                        m = meta_graphs[idx]
                        p_t = p_targets[idx]
                        c_t = c_targets[idx]
                        mask = p_masks[idx]

                        if p_t is not None and len(p_t) > 0 and m.num_cliques > 0:
                            if is_recurrent:
                                exp_t, p_prob, cond_t, curr_h = model_obj.forward_two_stage(m, vp_meta, vp_h)
                                p_logits = model_obj.persistence_head(curr_h).squeeze(-1)
                                vp_meta, vp_h = m, curr_h
                            elif isinstance(model_obj, FeatureOnlyMLP):
                                exp_t, p_prob, cond_t = model_obj.forward_two_stage(m.features)
                                h = model_obj.encoder(m.features)
                                p_logits = model_obj.persistence_head(h).squeeze(-1)
                            else:
                                exp_t, p_prob, cond_t = model_obj.forward_two_stage(m)
                                emb = model_obj.spatial_gnn(m.features, m.edge_index, m.edge_weights)
                                p_logits = model_obj.persistence_head(emb).squeeze(-1)

                            loss_p = bce_crit(p_logits, p_t)
                            if mask is not None and mask.sum() > 0:
                                loss_c = mse_crit(cond_t[mask], c_t[mask])
                            else:
                                loss_c = torch.tensor(0.0, device=m.features.device)

                            val_loss += (loss_p + loss_c).item()
                            v_count += 1

                avg_v = val_loss / v_count if v_count else 0.0
                if avg_v < best_val:
                    best_val = avg_v
                    best_state = {k: v.cpu().clone() for k, v in model_obj.state_dict().items()}

            model_obj.load_state_dict(best_state)
            model_obj.eval()

            # Testing
            ts_trues_A, ts_preds_A = [], []
            ts_trues_B, ts_preds_B = [], []
            ts_rhos_all, ts_rhos_act, ts_t5, ts_t10 = [], [], [] ,[]

            with torch.no_grad():
                tp_meta, tp_h = None, None
                if is_recurrent and warm_start:
                    for w_idx in range(0, test_indices[0]):
                        w_m = meta_graphs[w_idx]
                        _, _, _, w_h = model_obj.forward_two_stage(w_m, tp_meta, tp_h)
                        tp_meta, tp_h = w_m, w_h

                for idx in test_indices:
                    m = meta_graphs[idx]
                    z_t = z_targets[idx]
                    c_t = c_targets[idx]
                    mask = p_masks[idx]

                    if is_recurrent:
                        exp_t, p_prob, cond_t, curr_h = model_obj.forward_two_stage(m, tp_meta, tp_h)
                        tp_meta, tp_h = m, curr_h
                    elif isinstance(model_obj, FeatureOnlyMLP):
                        exp_t, p_prob, cond_t = model_obj.forward_two_stage(m.features)
                    else:
                        exp_t, p_prob, cond_t = model_obj.forward_two_stage(m)

                    if z_t is not None and len(z_t) > 0 and len(exp_t) > 0:
                        ts_preds_A.extend(exp_t.tolist())
                        ts_trues_A.extend(z_t.tolist())

                        if mask is not None:
                            for i in range(m.num_cliques):
                                if mask[i].item():
                                    ts_preds_B.append(float(cond_t[i].item()))
                                    ts_trues_B.append(float(c_t[i].item()))

                        pred_sri = {nid: sum(float(exp_t[c].item()) for c in m.node_to_clique_map.get(nid, []) if c < len(exp_t)) for nid in all_node_ids}
                        true_sri = {nid: sum(float(z_t[c].item()) for c in m.node_to_clique_map.get(nid, []) if c < len(z_t)) for nid in all_node_ids}
                        m_rank = compute_comprehensive_ranking_metrics(true_sri, pred_sri, top_k_list=[5, 10], all_node_ids=all_node_ids)
                        ts_rhos_all.append(m_rank["all_nodes"]["spearman_rho"])
                        ts_rhos_act.append(m_rank["active_nodes"]["spearman_rho"])
                        ts_t5.append(m_rank["all_nodes"]["top_k_overlaps"]["top_5"])
                        ts_t10.append(m_rank["all_nodes"]["top_k_overlaps"]["top_10"])

            mA = compute_regression_metrics(np.array(ts_trues_A), np.array(ts_preds_A))
            mB = compute_regression_metrics(np.array(ts_trues_B), np.array(ts_preds_B))
            results_A[model_key]["mse"].append(mA["mse"])
            results_A[model_key]["rmse"].append(mA["rmse"])
            results_A[model_key]["mae"].append(mA["mae"])
            results_A[model_key]["pearson_r"].append(mA["pearson_r"])
            results_A[model_key]["spearman_rho"].append(float(np.mean(ts_rhos_all)))
            results_A[model_key]["spearman_rho_active"].append(float(np.mean(ts_rhos_act)))
            results_A[model_key]["top5"].append(float(np.mean(ts_t5)))
            results_A[model_key]["top10"].append(float(np.mean(ts_t10)))

            results_B[model_key]["mse"].append(mB["mse"])
            results_B[model_key]["rmse"].append(mB["rmse"])
            results_B[model_key]["mae"].append(mB["mae"])
            results_B[model_key]["pearson_r"].append(mB["pearson_r"])

        # Train Two-Stage Models
        ts_mlp = FeatureOnlyMLP(in_features=5, hidden_dim=hidden_dim)
        train_and_eval_two_stage(ts_mlp, "Feature-Only MLP (Two-Stage)")

        ts_sgcn = StaticGCNModel(in_features=5, hidden_dim=hidden_dim, dropout=0.1)
        train_and_eval_two_stage(ts_sgcn, "Static GCN (Two-Stage)")

        ts_ablation = TemporalGNNAblationNoGRU(in_features=5, hidden_dim=hidden_dim, dropout=0.1)
        train_and_eval_two_stage(ts_ablation, "Temporal GNN No-GRU (Two-Stage)")

        ts_tgnn = TemporalGNN(in_features=5, hidden_dim=hidden_dim, dropout=0.1, jaccard_threshold=0.5, one_to_one_matching=True)
        train_and_eval_two_stage(ts_tgnn, "Temporal GNN Full (Two-Stage Warm-Start)", is_recurrent=True, warm_start=True)

    # -------------------------------------------------------------
    # Summarize across seeds
    # -------------------------------------------------------------
    def summarize(res_dict):
        summary = {}
        for m, mets in res_dict.items():
            summary[m] = {}
            for metric, vals in mets.items():
                mean_val = float(np.mean(vals)) if vals else 0.0
                std_val = float(np.std(vals)) if vals else 0.0
                summary[m][f"{metric}_mean"] = round(mean_val, 4)
                summary[m][f"{metric}_std"] = round(std_val, 4)
        return summary

    summary_A = summarize(results_A)
    summary_B = summarize(results_B)

    # Save JSON
    final_json = {
        "metadata": {
            "dataset": "email-Enron",
            "seeds": seeds,
            "epochs": epochs,
            "lr": lr,
            "hidden_dim": hidden_dim,
            "matching_strategy": "1-to-1 bipartite maximum Jaccard (greedy)",
            "test_cliques_protocol_A": test_clique_counts,
            "test_cliques_protocol_B": test_persisting_counts,
        },
        "protocol_A_unconditional": summary_A,
        "protocol_B_persisting_only": summary_B,
    }

    with open(out_path / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)

    # Save CSV
    csv_path = out_path / "benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "protocol", "model",
            "mse_mean", "mse_std",
            "rmse_mean", "rmse_std",
            "mae_mean", "mae_std",
            "pearson_r_mean", "pearson_r_std",
            "spearman_rho_all_mean", "spearman_rho_all_std",
            "spearman_rho_active_mean", "spearman_rho_active_std",
            "top5_overlap_mean", "top5_overlap_std",
            "top10_overlap_mean", "top10_overlap_std",
        ])

        for m, mets in summary_A.items():
            writer.writerow([
                f"Protocol A (Unconditional, N={test_clique_counts})", m,
                mets["mse_mean"], mets["mse_std"],
                mets["rmse_mean"], mets["rmse_std"],
                mets["mae_mean"], mets["mae_std"],
                mets["pearson_r_mean"], mets["pearson_r_std"],
                mets["spearman_rho_mean"], mets["spearman_rho_std"],
                mets["spearman_rho_active_mean"], mets["spearman_rho_active_std"],
                mets["top5_mean"], mets["top5_std"],
                mets["top10_mean"], mets["top10_std"],
            ])

        for m, mets in summary_B.items():
            writer.writerow([
                f"Protocol B (Persisting Only, N={test_persisting_counts})", m,
                mets["mse_mean"], mets["mse_std"],
                mets["rmse_mean"], mets["rmse_std"],
                mets["mae_mean"], mets["mae_std"],
                mets["pearson_r_mean"], mets["pearson_r_std"],
                "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
            ])

    print("\n" + "=" * 115)
    print(" PROTOCOL A (UNCONDITIONAL FORECASTING, ALL TEST CLIQUES N = 109, UNMATCHED = 0.0):")
    print("=" * 115)
    print(f"{'Model':<40} | {'MSE (mean±std)':<16} | {'MAE (mean±std)':<16} | {'Pearson r':<12} | {'rho (All)':<10} | {'rho (Active)'}")
    print("-" * 115)
    for m, mets in summary_A.items():
        print(f"{m:<40} | {mets['mse_mean']:>5.2f} ± {mets['mse_std']:<5.2f}  | {mets['mae_mean']:>5.2f} ± {mets['mae_std']:<5.2f}  | {mets['pearson_r_mean']:>6.4f}     | {mets['spearman_rho_mean']:>6.4f}   | {mets['spearman_rho_active_mean']:>6.4f}")
    print("=" * 115)

    print("\n" + "=" * 115)
    print(" PROTOCOL B (CONDITIONAL ON SURVIVAL, PERSISTING CLIQUES ONLY N = 35, J >= 0.5):")
    print("=" * 115)
    print(f"{'Model':<40} | {'MSE (mean±std)':<16} | {'MAE (mean±std)':<16} | {'Pearson r (mean±std)'}")
    print("-" * 115)
    for m, mets in summary_B.items():
        print(f"{m:<40} | {mets['mse_mean']:>5.2f} ± {mets['mse_std']:<5.2f}  | {mets['mae_mean']:>5.2f} ± {mets['mae_std']:<5.2f}  | {mets['pearson_r_mean']:>6.4f} ± {mets['pearson_r_std']:<6.4f}")
    print("=" * 115)
    print(f"\nArtifacts saved to:\n  - {out_path / 'benchmark_results.json'}\n  - {out_path / 'benchmark_results.csv'}")


if __name__ == "__main__":
    run_multi_seed_benchmark()
