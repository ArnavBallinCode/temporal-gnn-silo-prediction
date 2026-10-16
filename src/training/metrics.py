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
) -> Dict[str, float]:
    """
    Computes node-level ranking metrics: Spearman rank correlation and top-k overlap.

    Args:
        true_sri: Ground truth SRI per node.
        pred_sri: Predicted SRI per node.
        top_k: Number of highest-risk employees to consider for top-k overlap.

    Returns:
        Dict containing spearman_rho and top_k_overlap.
    """
    common_nodes = sorted(set(true_sri.keys()).intersection(pred_sri.keys()))
    if len(common_nodes) < 2:
        return {"spearman_rho": 0.0, "top_k_overlap": 0.0}

    y_t = np.array([true_sri[n] for n in common_nodes])
    y_p = np.array([pred_sri[n] for n in common_nodes])

    # Spearman rank correlation
    if np.std(y_t) > 1e-8 and np.std(y_p) > 1e-8:
        rho, _ = stats.spearmanr(y_t, y_p)
        spearman_rho = float(rho) if not np.isnan(rho) else 0.0
    else:
        spearman_rho = 0.0

    # Top-k overlap
    sorted_true = sorted(common_nodes, key=lambda n: true_sri[n], reverse=True)[:top_k]
    sorted_pred = sorted(common_nodes, key=lambda n: pred_sri[n], reverse=True)[:top_k]

    set_true = set(sorted_true)
    set_pred = set(sorted_pred)
    overlap = len(set_true.intersection(set_pred)) / float(top_k)

    return {
        "spearman_rho": round(spearman_rho, 4),
        "top_k_overlap": round(overlap, 4),
    }
