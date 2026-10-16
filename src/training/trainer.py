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
from src.models.temporal_gnn import TemporalGNN, match_cliques_jaccard
from src.training.metrics import compute_regression_metrics, compute_ranking_metrics


def build_forecasting_targets(
    meta_graphs: List[CliqueMetaGraph],
    delta: int = 1,
    jaccard_threshold: float = 0.5,
) -> List[Optional[torch.Tensor]]:
    """
    Constructs future tension targets for each snapshot k.

    Target definition:
    For snapshot k and horizon delta:
    For each clique C_i in snapshot k:
    - We check snapshot k + delta for a matching persisting clique (Jaccard >= threshold).
    - If found, target y_i = ground_truth_tension(C_matched).
    - If dissolved / unobserved, target y_i = 0.0.

    Args:
        meta_graphs: Sequence of CliqueMetaGraph objects.
        delta: Prediction horizon in snapshots (e.g. delta=1 means 1 snapshot ahead).
        jaccard_threshold: Minimum node overlap to consider a group persisting.

    Returns:
        List of target tensors [K_k] for each snapshot (None for the final delta snapshots).
    """
    targets: List[Optional[torch.Tensor]] = []
    T = len(meta_graphs)

    for k in range(T):
        future_idx = k + delta
        if future_idx >= T:
            targets.append(None)
            continue

        curr_meta = meta_graphs[k]
        future_meta = meta_graphs[future_idx]

        if curr_meta.num_cliques == 0:
            targets.append(torch.zeros((0,), dtype=torch.float32))
            continue

        # Match current cliques to future cliques
        matches = match_cliques_jaccard(
            prev_cliques=curr_meta.cliques,
            curr_cliques=future_meta.cliques,
            threshold=jaccard_threshold,
        )
        # matches is {future_clique_idx: curr_clique_idx}
        # Invert to map curr_clique_idx -> future_clique_idx
        curr_to_future: Dict[int, int] = {c_idx: f_idx for f_idx, c_idx in matches.items()}

        target_vals = []
        for curr_i in range(curr_meta.num_cliques):
            if curr_i in curr_to_future:
                f_idx = curr_to_future[curr_i]
                target_vals.append(float(future_meta.clique_tensions[f_idx].item()))
            else:
                # Group dissolved or lost coherence
                target_vals.append(0.0)

        targets.append(torch.tensor(target_vals, dtype=torch.float32))

    return targets


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
    targets: List[Optional[torch.Tensor]],
    train_indices: List[int],
    val_indices: List[int],
    epochs: int = 50,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> Tuple[TemporalGNN, List[float], List[float]]:
    """
    Trains the Temporal GNN using chronological sequence progression and Adam optimizer.

    Returns:
        (best_model, train_loss_history, val_loss_history)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    train_loss_hist: List[float] = []
    val_loss_hist: List[float] = []

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
            target = targets[idx]

            preds, curr_h = model.forward_snapshot(meta, prev_meta, prev_h)

            if target is not None and len(target) > 0 and len(preds) > 0:
                loss = criterion(preds, target)
                total_train_loss += loss.item()
                train_steps += 1
                # Accumulate gradients
                loss.backward()

            prev_meta = meta
            # Detach hidden state to prevent backpropagation through entire multi-year history (truncated BPTT)
            prev_h = curr_h.detach() if curr_h.numel() > 0 else None

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

            for idx in val_indices:
                meta = meta_graphs[idx]
                target = targets[idx]

                preds, curr_h = model.forward_snapshot(meta, v_prev_meta, v_prev_h)

                if target is not None and len(target) > 0 and len(preds) > 0:
                    v_loss = criterion(preds, target)
                    total_val_loss += v_loss.item()
                    val_steps += 1

                v_prev_meta = meta
                v_prev_h = curr_h

        avg_val_loss = total_val_loss / val_steps if val_steps > 0 else avg_train_loss
        val_loss_hist.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())

    # Load best weights
    model.load_state_dict(best_weights)
    return model, train_loss_hist, val_loss_hist


def evaluate_model(
    model: TemporalGNN,
    meta_graphs: List[CliqueMetaGraph],
    targets: List[Optional[torch.Tensor]],
    eval_indices: List[int],
    all_node_ids: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Evaluates model predictions across evaluation snapshots.

    Computes:
    - Clique tension forecasting: MSE, MAE, Pearson r
    - Node SRI forecasting: Spearman rho, Top-5 overlap, Top-10 overlap
    """
    model.eval()
    all_y_true: List[float] = []
    all_y_pred: List[float] = []

    spearman_rhos: List[float] = []
    top5_overlaps: List[float] = []
    top10_overlaps: List[float] = []

    with torch.no_grad():
        prev_meta: Optional[CliqueMetaGraph] = None
        prev_h: Optional[torch.Tensor] = None

        for idx in eval_indices:
            meta = meta_graphs[idx]
            target = targets[idx]

            preds, curr_h = model.forward_snapshot(meta, prev_meta, prev_h)

            if target is not None and len(target) > 0 and len(preds) > 0:
                y_t = target.tolist()
                y_p = preds.tolist()
                all_y_true.extend(y_t)
                all_y_pred.extend(y_p)

                # Compute node-level predicted SRI
                pred_sri = model.predict_node_sri(preds, meta, all_node_ids)
                true_sri = model.predict_node_sri(target, meta, all_node_ids)

                rank_5 = compute_ranking_metrics(true_sri, pred_sri, top_k=5)
                rank_10 = compute_ranking_metrics(true_sri, pred_sri, top_k=10)

                spearman_rhos.append(rank_10["spearman_rho"])
                top5_overlaps.append(rank_5["top_k_overlap"])
                top10_overlaps.append(rank_10["top_k_overlap"])

            prev_meta = meta
            prev_h = curr_h

    reg_metrics = compute_regression_metrics(np.array(all_y_true), np.array(all_y_pred))

    return {
        "mse": reg_metrics["mse"],
        "rmse": reg_metrics["rmse"],
        "mae": reg_metrics["mae"],
        "pearson_r": reg_metrics["pearson_r"],
        "avg_spearman_rho": round(float(np.mean(spearman_rhos)) if spearman_rhos else 0.0, 4),
        "avg_top5_overlap": round(float(np.mean(top5_overlaps)) if top5_overlaps else 0.0, 4),
        "avg_top10_overlap": round(float(np.mean(top10_overlaps)) if top10_overlaps else 0.0, 4),
    }
