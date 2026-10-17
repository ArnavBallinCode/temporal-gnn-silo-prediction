"""
Temporal GNN Training and Chronological Evaluation Pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.graph.meta_graph import CliqueMetaGraph
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.graph.cliques import (
    CliqueMatchResult,
    categorize_clique_disappearance,
    match_cliques_bipartite,
)
from src.graph.meta_graph import CliqueMetaGraph
from src.models.temporal_gnn import TemporalGNN, match_cliques_jaccard
from src.training.metrics import (
    compute_ranking_metrics,
    compute_regression_metrics,
    compute_comprehensive_ranking_metrics,
)


@dataclass
class TwoStageTargets:
    """
    Two-stage forecasting target bundle for a sequence of snapshots.

    Attributes:
        persistence_targets: List of binary tensors [K_k] (1.0 = persisted 1-to-1, 0.0 = unmatched).
        conditional_targets: List of float tensors [K_k] (future tension if persisted, 0.0 if not).
        zero_fill_targets: List of float tensors [K_k] (Protocol A baseline: future tension if matched, 0.0 if not).
        persisting_masks: List of boolean tensors [K_k] indicating surviving cliques.
        disappearance_records: Per-snapshot list of categorization dicts for each clique.
    """
    persistence_targets: List[Optional[torch.Tensor]]
    conditional_targets: List[Optional[torch.Tensor]]
    zero_fill_targets: List[Optional[torch.Tensor]]
    persisting_masks: List[Optional[torch.Tensor]]
    disappearance_records: List[Optional[List[Dict[str, Any]]]]


def build_two_stage_targets(
    meta_graphs: List[CliqueMetaGraph],
    all_snapshots_cliques: Optional[List[List[Tuple[int, ...]]]] = None,
    delta: int = 1,
    jaccard_threshold: float = 0.5,
    matching_method: str = "greedy",
) -> TwoStageTargets:
    """
    Constructs rigorous two-stage forecasting targets across snapshots.

    Disentangles two distinct physical outcomes:
    1. Stage A: Clique persistence (binary classification: 1 = persists, 0 = dissolves).
    2. Stage B: Conditional tension forecasting (regression: future tension given persistence).

    Guarantees:
    - 1-to-1 bipartite matching prevents multiple successors mapping to the same predecessor.
    - Explicitly categorizes unobserved cliques: distinguishes top-k truncation from structural dissolution.
    - Generates separate masks and targets for conditional and unconditional protocols.

    Args:
        meta_graphs: Sequence of CliqueMetaGraph objects.
        all_snapshots_cliques: Optional list of ALL maximal cliques before top-k truncation.
        delta: Prediction horizon in snapshots (default: 1).
        jaccard_threshold: Minimum node overlap to consider a group persisting (default: 0.5).
        matching_method: "greedy" or "hungarian".

    Returns:
        TwoStageTargets dataclass.
    """
    p_targets: List[Optional[torch.Tensor]] = []
    c_targets: List[Optional[torch.Tensor]] = []
    z_targets: List[Optional[torch.Tensor]] = []
    masks: List[Optional[torch.Tensor]] = []
    records: List[Optional[List[Dict[str, Any]]]] = []

    T = len(meta_graphs)

    for k in range(T):
        future_idx = k + delta
        if future_idx >= T:
            p_targets.append(None)
            c_targets.append(None)
            z_targets.append(None)
            masks.append(None)
            records.append(None)
            continue

        curr_meta = meta_graphs[k]
        future_meta = meta_graphs[future_idx]

        if curr_meta.num_cliques == 0:
            empty_f = torch.zeros((0,), dtype=torch.float32)
            empty_b = torch.zeros((0,), dtype=torch.bool)
            p_targets.append(empty_f)
            c_targets.append(empty_f)
            z_targets.append(empty_f)
            masks.append(empty_b)
            records.append([])
            continue

        # 1-to-1 bipartite matching
        match_res = match_cliques_bipartite(
            prev_cliques=curr_meta.cliques,
            curr_cliques=future_meta.cliques,
            threshold=jaccard_threshold,
            method=matching_method,
        )

        next_all = all_snapshots_cliques[future_idx] if all_snapshots_cliques and future_idx < len(all_snapshots_cliques) else None

        p_vals: List[float] = []
        c_vals: List[float] = []
        z_vals: List[float] = []
        m_vals: List[bool] = []
        rec_vals: List[Dict[str, Any]] = []

        for curr_i in range(curr_meta.num_cliques):
            if curr_i in match_res.prev_to_curr:
                f_idx = match_res.prev_to_curr[curr_i]
                tension_val = float(future_meta.clique_tensions[f_idx].item())
                p_vals.append(1.0)
                c_vals.append(tension_val)
                z_vals.append(tension_val)
                m_vals.append(True)
                rec_vals.append({
                    "clique_idx": curr_i,
                    "matched_future_idx": f_idx,
                    "jaccard": match_res.matched_similarities.get((curr_i, f_idx), 1.0),
                    "category": "matched_retained",
                })
            else:
                p_vals.append(0.0)
                c_vals.append(0.0)
                z_vals.append(0.0)
                m_vals.append(False)
                # Audit why it was unmatched
                audit_info = categorize_clique_disappearance(
                    prev_clique=curr_meta.cliques[curr_i],
                    next_retained_cliques=future_meta.cliques,
                    next_all_cliques=next_all,
                    threshold=jaccard_threshold,
                )
                audit_info["clique_idx"] = curr_i
                rec_vals.append(audit_info)

        p_targets.append(torch.tensor(p_vals, dtype=torch.float32))
        c_targets.append(torch.tensor(c_vals, dtype=torch.float32))
        z_targets.append(torch.tensor(z_vals, dtype=torch.float32))
        masks.append(torch.tensor(m_vals, dtype=torch.bool))
        records.append(rec_vals)

    return TwoStageTargets(
        persistence_targets=p_targets,
        conditional_targets=c_targets,
        zero_fill_targets=z_targets,
        persisting_masks=masks,
        disappearance_records=records,
    )


def build_forecasting_targets(
    meta_graphs: List[CliqueMetaGraph],
    delta: int = 1,
    jaccard_threshold: float = 0.5,
    one_to_one: bool = True,
) -> List[Optional[torch.Tensor]]:
    """
    Constructs future tension targets for each snapshot k (Protocol A baseline).

    Uses 1-to-1 bipartite matching to ensure uniqueness.
    """
    targets_bundle = build_two_stage_targets(
        meta_graphs=meta_graphs,
        delta=delta,
        jaccard_threshold=jaccard_threshold,
        matching_method="greedy" if one_to_one else "greedy",
    )
    return targets_bundle.zero_fill_targets


def chronological_split(
    items: List[object],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[List[object], List[object], List[object]]:
    """
    Partitions a temporal sequence chronologically into Train, Val, and Test splits.
    Guarantees no future information leaks into training.
    """
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_set = items[:n_train]
    val_set = items[n_train : n_train + n_val]
    test_set = items[n_train + n_val :]

    return train_set, val_set, test_set


def train_temporal_gnn(
    model: TemporalGNN,
    meta_graphs: List[CliqueMetaGraph],
    targets: Union[List[Optional[torch.Tensor]], TwoStageTargets],
    train_indices: List[int],
    val_indices: List[int],
    epochs: int = 50,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    seed: int = 42,
    target_mode: str = "zero_fill",
    state_protocol: str = "warm_start",
) -> Tuple[TemporalGNN, List[float], List[float]]:
    """
    Trains the Temporal GNN using chronological sequence progression and Adam optimizer.

    Supports:
    - target_mode="zero_fill": Original Protocol A single-stage regression.
    - target_mode="two_stage": Hurdle model (Stage A persistence BCE + Stage B conditional tension MSE).
    - state_protocol="warm_start": Warmed up hidden states carried chronologically across train and val.
    - state_protocol="cold_start": Hidden states reset to None at boundary.

    Returns:
        (best_model, train_loss_history, val_loss_history)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    train_loss_hist: List[float] = []
    val_loss_hist: List[float] = []

    # Unpack targets if TwoStageTargets passed
    if isinstance(targets, TwoStageTargets):
        p_targets = targets.persistence_targets
        c_targets = targets.conditional_targets
        p_masks = targets.persisting_masks
        z_targets = targets.zero_fill_targets
    else:
        p_targets = None
        c_targets = None
        p_masks = None
        z_targets = targets

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        total_train_loss = 0.0
        train_steps = 0

        prev_meta: Optional[CliqueMetaGraph] = None
        prev_h: Optional[torch.Tensor] = None

        # Sequence through training snapshots
        for idx in train_indices:
            meta = meta_graphs[idx]
            if meta.num_cliques == 0:
                prev_meta = meta
                prev_h = None
                continue

            if target_mode == "two_stage" and p_targets is not None and c_targets is not None:
                p_t = p_targets[idx]
                c_t = c_targets[idx]
                m_t = p_masks[idx]

                if p_t is not None and len(p_t) > 0:
                    exp_t, p_prob, cond_t, curr_h = model.forward_two_stage(meta, prev_meta, prev_h)
                    # Stage A loss (Persistence classification logit)
                    p_logits = model.persistence_head(curr_h).squeeze(-1)
                    loss_a = bce_criterion(p_logits, p_t)

                    # Stage B loss (Conditional tension on surviving cliques only)
                    if m_t is not None and m_t.sum() > 0:
                        loss_b = mse_criterion(cond_t[m_t], c_t[m_t])
                    else:
                        loss_b = torch.tensor(0.0, device=curr_h.device)

                    loss = loss_a + loss_b
                    total_train_loss += loss.item()
                    train_steps += 1
                    loss.backward()
                else:
                    curr_h = model.spatial_gnn(meta.features, meta.edge_index, meta.edge_weights)
            else:
                # Single-stage zero-fill
                t = z_targets[idx]
                preds, curr_h = model.forward_snapshot(meta, prev_meta, prev_h)
                if t is not None and len(t) > 0 and len(preds) > 0:
                    loss = mse_criterion(preds, t)
                    total_train_loss += loss.item()
                    train_steps += 1
                    loss.backward()

            prev_meta = meta
            # Truncated BPTT: detach hidden states
            prev_h = curr_h.detach() if (curr_h is not None and curr_h.numel() > 0) else None

        if train_steps > 0:
            optimizer.step()
            avg_train_loss = total_train_loss / train_steps
        else:
            avg_train_loss = 0.0

        train_loss_hist.append(avg_train_loss)

        # Validation step
        model.eval()
        total_val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            v_prev_meta: Optional[CliqueMetaGraph] = None
            v_prev_h: Optional[torch.Tensor] = None

            if state_protocol == "warm_start":
                # Warm up hidden state by executing chronological history up to first validation snapshot
                for w_idx in train_indices:
                    w_meta = meta_graphs[w_idx]
                    _, w_curr_h = model.forward_snapshot(w_meta, v_prev_meta, v_prev_h)
                    v_prev_meta = w_meta
                    v_prev_h = w_curr_h

            for idx in val_indices:
                meta = meta_graphs[idx]
                if meta.num_cliques == 0:
                    v_prev_meta = meta
                    v_prev_h = None
                    continue

                if target_mode == "two_stage" and p_targets is not None and c_targets is not None:
                    p_t = p_targets[idx]
                    c_t = c_targets[idx]
                    m_t = p_masks[idx]

                    if p_t is not None and len(p_t) > 0:
                        exp_t, p_prob, cond_t, curr_h = model.forward_two_stage(meta, v_prev_meta, v_prev_h)
                        p_logits = model.persistence_head(curr_h).squeeze(-1)
                        loss_a = bce_criterion(p_logits, p_t)
                        if m_t is not None and m_t.sum() > 0:
                            loss_b = mse_criterion(cond_t[m_t], c_t[m_t])
                        else:
                            loss_b = torch.tensor(0.0, device=curr_h.device)
                        v_loss = loss_a + loss_b
                        total_val_loss += v_loss.item()
                        val_steps += 1
                    else:
                        curr_h = None
                else:
                    t = z_targets[idx]
                    preds, curr_h = model.forward_snapshot(meta, v_prev_meta, v_prev_h)
                    if t is not None and len(t) > 0 and len(preds) > 0:
                        v_loss = mse_criterion(preds, t)
                        total_val_loss += v_loss.item()
                        val_steps += 1

                v_prev_meta = meta
                v_prev_h = curr_h

        avg_val_loss = total_val_loss / val_steps if val_steps > 0 else avg_train_loss
        val_loss_hist.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model, train_loss_hist, val_loss_hist


def evaluate_model(
    model: TemporalGNN,
    meta_graphs: List[CliqueMetaGraph],
    targets: Union[List[Optional[torch.Tensor]], TwoStageTargets],
    eval_indices: List[int],
    all_node_ids: Optional[List[int]] = None,
    target_mode: str = "zero_fill",
    state_protocol: str = "warm_start",
    snapshots: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates model predictions across evaluation snapshots.

    Reports:
    - Protocol A (Unconditional): MSE, RMSE, MAE, Pearson r
    - Protocol B (Persisting cliques only): MSE, RMSE, MAE, Pearson r
    - Node SRI forecasting (All 148 nodes): Spearman rho, Top-5, Top-10 overlap
    - Node SRI forecasting (Active nodes only): Spearman rho, Top-5, Top-10 overlap
    """
    model.eval()

    if isinstance(targets, TwoStageTargets):
        z_targets = targets.zero_fill_targets
        c_targets = targets.conditional_targets
        p_masks = targets.persisting_masks
    else:
        z_targets = targets
        c_targets = None
        p_masks = None

    all_y_true_A: List[float] = []
    all_y_pred_A: List[float] = []

    all_y_true_B: List[float] = []
    all_y_pred_B: List[float] = []

    spearman_rhos_all: List[float] = []
    spearman_rhos_act: List[float] = []
    top5_overlaps: List[float] = []
    top10_overlaps: List[float] = []

    with torch.no_grad():
        prev_meta: Optional[CliqueMetaGraph] = None
        prev_h: Optional[torch.Tensor] = None

        if state_protocol == "warm_start" and len(eval_indices) > 0:
            first_eval_idx = eval_indices[0]
            for w_idx in range(0, first_eval_idx):
                w_meta = meta_graphs[w_idx]
                _, curr_h = model.forward_snapshot(w_meta, prev_meta, prev_h)
                prev_meta = w_meta
                prev_h = curr_h

        for idx in eval_indices:
            meta = meta_graphs[idx]
            if meta.num_cliques == 0:
                prev_meta = meta
                prev_h = None
                continue

            target_A = z_targets[idx]
            mask_B = p_masks[idx] if p_masks is not None else None
            target_B = c_targets[idx] if c_targets is not None else None

            if target_mode == "two_stage":
                exp_t, p_prob, cond_t, curr_h = model.forward_two_stage(meta, prev_meta, prev_h)
                preds_A = exp_t
                preds_B = cond_t
            else:
                preds_A, curr_h = model.forward_snapshot(meta, prev_meta, prev_h)
                preds_B = preds_A

            if target_A is not None and len(target_A) > 0 and len(preds_A) > 0:
                # Protocol A
                all_y_true_A.extend(target_A.tolist())
                all_y_pred_A.extend(preds_A.tolist())

                # Protocol B
                if mask_B is not None and target_B is not None:
                    for i in range(meta.num_cliques):
                        if mask_B[i].item():
                            all_y_true_B.append(float(target_B[i].item()))
                            all_y_pred_B.append(float(preds_B[i].item()))

                # Node-level SRI ranking
                pred_sri = model.predict_node_sri(preds_A, meta, all_node_ids)
                true_sri = model.predict_node_sri(target_A, meta, all_node_ids)

                m_rank = compute_comprehensive_ranking_metrics(true_sri, pred_sri, top_k_list=[5, 10], all_node_ids=all_node_ids)
                spearman_rhos_all.append(m_rank["all_nodes"]["spearman_rho"])
                spearman_rhos_act.append(m_rank["active_nodes"]["spearman_rho"])
                top5_overlaps.append(m_rank["all_nodes"]["top_k_overlaps"]["top_5"])
                top10_overlaps.append(m_rank["all_nodes"]["top_k_overlaps"]["top_10"])

            prev_meta = meta
            prev_h = curr_h

    reg_metrics_A = compute_regression_metrics(np.array(all_y_true_A), np.array(all_y_pred_A))
    reg_metrics_B = compute_regression_metrics(np.array(all_y_true_B), np.array(all_y_pred_B))

    return {
        "mse": reg_metrics_A["mse"],
        "rmse": reg_metrics_A["rmse"],
        "mae": reg_metrics_A["mae"],
        "pearson_r": reg_metrics_A["pearson_r"],
        "protocol_b_mse": reg_metrics_B["mse"],
        "protocol_b_rmse": reg_metrics_B["rmse"],
        "protocol_b_mae": reg_metrics_B["mae"],
        "protocol_b_pearson_r": reg_metrics_B["pearson_r"],
        "avg_spearman_rho": round(float(np.mean(spearman_rhos_all)) if spearman_rhos_all else 0.0, 4),
        "avg_spearman_rho_active": round(float(np.mean(spearman_rhos_act)) if spearman_rhos_act else 0.0, 4),
        "avg_top5_overlap": round(float(np.mean(top5_overlaps)) if top5_overlaps else 0.0, 4),
        "avg_top10_overlap": round(float(np.mean(top10_overlaps)) if top10_overlaps else 0.0, 4),
        "num_protocol_a_cliques": len(all_y_true_A),
        "num_protocol_b_cliques": len(all_y_true_B),
    }

