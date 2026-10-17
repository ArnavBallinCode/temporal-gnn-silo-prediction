"""
Evaluation Metrics for Clique Tension and Node-level Silo Risk Index.

"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
import numpy as np
import scipy.stats as stats


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Computes regression evaluation metrics: MSE, RMSE, MAE, and Pearson correlation.

    Args:
        y_true: Ground truth target array.
        y_pred: Predicted values array.

    Returns:
        Dict containing mse, rmse, mae, pearson_r.
    """
    if len(y_true) == 0:
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "pearson_r": 0.0}

    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    # Pearson correlation
    if len(y_true) > 1 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        r, _ = stats.pearsonr(y_true, y_pred)
        pearson_r = float(r) if not np.isnan(r) else 0.0
    else:
        pearson_r = 0.0

    return {
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "pearson_r": round(pearson_r, 4),
    }


def compute_ranking_metrics(
    true_sri: Dict[int, float],
    pred_sri: Dict[int, float],
    top_k: int = 10,
    active_only: bool = False,
) -> Dict[str, float]:
    """
    Computes node-level ranking metrics: Spearman rank correlation, p-value, and top-k overlap.

    Args:
        true_sri: Ground truth SRI per node.
        pred_sri: Predicted SRI per node.
        top_k: Number of highest-risk employees to consider for top-k overlap.
        active_only: If True, evaluates only active nodes with non-zero true or predicted SRI.
                     If False (default), evaluates across all common nodes.

    Returns:
        Dict containing spearman_rho, p_value, top_k_overlap, mae, mse, pearson_r, num_nodes.
    """
    if active_only:
        nodes = sorted([n for n in true_sri if true_sri[n] > 0.0 or pred_sri.get(n, 0.0) > 0.0])
    else:
        nodes = sorted(set(true_sri.keys()).intersection(pred_sri.keys()))

    if len(nodes) < 2:
        return {
            "spearman_rho": 0.0,
            "p_value": 1.0,
            "top_k_overlap": 0.0,
            "pearson_r": 0.0,
            "mae": 0.0,
            "mse": 0.0,
            "num_nodes": len(nodes),
        }

    y_t = np.array([true_sri[n] for n in nodes], dtype=np.float64)
    y_p = np.array([pred_sri.get(n, 0.0) for n in nodes], dtype=np.float64)

    # Spearman rank correlation with exact tie handling via scipy.stats
    if np.std(y_t) > 1e-8 and np.std(y_p) > 1e-8:
        res = stats.spearmanr(y_t, y_p)
        spearman_rho = float(res.statistic) if not np.isnan(res.statistic) else 0.0
        p_val = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0
        r_res, _ = stats.pearsonr(y_t, y_p)
        pearson_r = float(r_res) if not np.isnan(r_res) else 0.0
    else:
        spearman_rho = 0.0
        p_val = 1.0
        pearson_r = 0.0

    diff = y_p - y_t
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))

    # Top-k overlap
    k = min(top_k, len(nodes))
    sorted_true = sorted(nodes, key=lambda n: true_sri[n], reverse=True)[:k]
    sorted_pred = sorted(nodes, key=lambda n: pred_sri.get(n, 0.0), reverse=True)[:k]

    set_true = set(sorted_true)
    set_pred = set(sorted_pred)
    overlap = len(set_true.intersection(set_pred)) / float(k) if k > 0 else 0.0

    return {
        "spearman_rho": round(spearman_rho, 4),
        "p_value": round(p_val, 6),
        "top_k_overlap": round(overlap, 4),
        "pearson_r": round(pearson_r, 4),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "num_nodes": len(nodes),
    }


def compute_comprehensive_ranking_metrics(
    true_sri: Dict[int, float],
    pred_sri: Dict[int, float],
    top_k_list: List[int] = [5, 10],
    all_node_ids: Optional[List[int]] = None,
) -> Dict[str, any]:
    """
    Computes ranking metrics across both:
    1. All labelled employees (e.g. N = 148).
    2. Active employees with non-zero risk (e.g. N ~ 30-70).

    Returns:
        Structured dictionary with metrics for all nodes and active nodes.
    """
    # Ensure all_node_ids are present if provided
    t_dict = dict(true_sri)
    p_dict = dict(pred_sri)
    if all_node_ids:
        for nid in all_node_ids:
            if nid not in t_dict:
                t_dict[nid] = 0.0
            if nid not in p_dict:
                p_dict[nid] = 0.0

    m_all = compute_ranking_metrics(t_dict, p_dict, top_k=10, active_only=False)
    m_act = compute_ranking_metrics(t_dict, p_dict, top_k=10, active_only=True)

    overlaps_all = {}
    overlaps_act = {}
    for k in top_k_list:
        overlaps_all[f"top_{k}"] = compute_ranking_metrics(t_dict, p_dict, top_k=k, active_only=False)["top_k_overlap"]
        overlaps_act[f"top_{k}"] = compute_ranking_metrics(t_dict, p_dict, top_k=k, active_only=True)["top_k_overlap"]

    return {
        "all_nodes": {
            "num_nodes": m_all["num_nodes"],
            "spearman_rho": m_all["spearman_rho"],
            "p_value": m_all["p_value"],
            "pearson_r": m_all["pearson_r"],
            "mae": m_all["mae"],
            "mse": m_all["mse"],
            "top_k_overlaps": overlaps_all,
        },
        "active_nodes": {
            "num_nodes": m_act["num_nodes"],
            "spearman_rho": m_act["spearman_rho"],
            "p_value": m_act["p_value"],
            "pearson_r": m_act["pearson_r"],
            "mae": m_act["mae"],
            "mse": m_act["mse"],
            "top_k_overlaps": overlaps_act,
        },
    }
