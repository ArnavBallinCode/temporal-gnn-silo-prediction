#!/usr/bin/env python3
"""
End-to-End Training & Evaluation CLI Pipeline.

Usage:
    python3 src/training/run_experiment.py [--window_days 30] [--epochs 40] [--lr 0.01]

Runs chronological train/val/test evaluation comparing:
1. Historical Persistence Baseline
2. Feature-Only MLP
3. Static GCN
4. Temporal GNN (Spatial GCN + GRU)

Saves all trained checkpoints, metrics, and prediction artifacts to outputs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import load_node_labels, load_raw_vectors, parse_simplex_events
from src.data.snapshots import build_graph_snapshots
from src.graph.cliques import extract_maximal_cliques
from src.graph.meta_graph import build_clique_meta_graph
from src.models.baselines import FeatureOnlyMLP, PersistenceBaseline, StaticGCNModel
from src.models.temporal_gnn import TemporalGNN
from src.training.metrics import compute_ranking_metrics, compute_regression_metrics
from src.training.trainer import build_forecasting_targets, evaluate_model, train_temporal_gnn


def save_artifacts(
    output_dir: Path,
    model: TemporalGNN,
    metrics_summary: dict,
    snapshots: list,
    meta_graphs: list,
    targets: list,
    test_indices: list,
    labels: dict,
    all_node_ids: list,
    train_losses: list,
    val_losses: list,
) -> None:
    """
    Saves trained checkpoints, metric reports, and prediction tables to outputs/.
    """
    checkpoints_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save trained model weights
    checkpoint_path = checkpoints_dir / "temporal_gnn_best.pt"
    torch.save(model.state_dict(), checkpoint_path)

    # 2. Save benchmark metrics JSON
    metrics_path = results_dir / "benchmark_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # 3. Save loss history
    loss_path = results_dir / "loss_history.json"
    with open(loss_path, "w", encoding="utf-8") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses}, f, indent=2)

    # 4. Save snapshot statistics CSV
    stats_path = results_dir / "snapshot_statistics.csv"
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "snapshot_id",
            "start_date",
            "end_date",
            "events_count",
            "active_nodes",
            "edges_count",
            "density",
            "connected_components",
            "num_cliques",
        ])
        for s, m in zip(snapshots, meta_graphs):
            writer.writerow([
                s.snapshot_id,
                s.start_date_str[:10],
                s.end_date_str[:10],
                s.num_events,
                s.num_nodes,
                s.num_edges,
                round(s.density, 4),
                s.num_connected_components,
                m.num_cliques,
            ])

    # 5. Save forecasted SRI predictions on test split
    preds_path = results_dir / "forecasted_sri_predictions.csv"
    model.eval()
    with open(preds_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["snapshot_id", "date_range", "node_id", "email", "predicted_sri", "true_sri"])

        with torch.no_grad():
            v_prev_meta = None
            v_prev_h = None
            for idx in test_indices:
                meta = meta_graphs[idx]
                target = targets[idx]
                preds, curr_h = model.forward_snapshot(meta, v_prev_meta, v_prev_h)

                if target is not None and len(target) > 0 and len(preds) > 0:
                    pred_sri = model.predict_node_sri(preds, meta, all_node_ids)
                    true_sri = model.predict_node_sri(target, meta, all_node_ids)
                    date_range = f"{snapshots[idx].start_date_str[:10]} to {snapshots[idx].end_date_str[:10]}"

                    # Top high-risk employees
                    sorted_nodes = sorted(pred_sri.items(), key=lambda x: x[1], reverse=True)
                    for nid, p_score in sorted_nodes:
                        if p_score > 0.0 or true_sri.get(nid, 0.0) > 0.0:
                            email = labels.get(nid, "unknown")
                            writer.writerow([
                                idx,
                                date_range,
                                nid,
                                email,
                                round(p_score, 4),
                                round(true_sri.get(nid, 0.0), 4),
                            ])

                v_prev_meta = meta
                v_prev_h = curr_h


def run_experiment(
    data_dir: str = "Dataset/email-Enron",
    window_days: float = 30.0,
    stride_days: float = 30.0,
    min_clique_size: int = 3,
    top_k_cliques: int = 30,
    forecast_delta: int = 1,
    epochs: int = 40,
    lr: float = 0.01,
    hidden_dim: int = 16,
    seed: int = 42,
    output_dir: str = "outputs",
) -> int:
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 85)
    print(" TEMPORAL GNN FOR ORGANIZATIONAL SILO DETECTION - BENCHMARK PIPELINE")
    print(f" Dataset:            {data_dir}")
    print(f" Parameters:         window={window_days}d, delta={forecast_delta} step, epochs={epochs}, lr={lr}")
    print("=" * 85)

    # 1. Load data
    print("\n[Step 1/5] Loading raw communication data...")
    nverts, simplices, times = load_raw_vectors(data_dir)
    labels = load_node_labels(data_dir)
    events = parse_simplex_events(nverts, simplices, times)
    print(f"  - Loaded {len(events):,} email events across {len(labels)} employees.")

    # 2. Build snapshots and meta-graphs
    print("\n[Step 2/5] Building snapshots and extracting clique-conflict meta-graphs...")
    snapshots = build_graph_snapshots(events, window_days=window_days, stride_days=stride_days)
    all_node_ids = sorted(labels.keys())

    meta_graphs = []
    for s in snapshots:
        cliques = extract_maximal_cliques(s.graph, min_size=min_clique_size, top_k=top_k_cliques)
        meta = build_clique_meta_graph(s, cliques, conflict_threshold=0.0)
        meta_graphs.append(meta)

    print(f"  - Generated {len(meta_graphs)} snapshot meta-graphs.")

    # 3. Build forecasting targets & chronological splits
    print("\n[Step 3/5] Constructing future tension forecasting targets and chronological splits...")
    targets = build_forecasting_targets(meta_graphs, delta=forecast_delta)

    # Chronological partition (70% Train, 15% Val, 15% Test)
    valid_indices = [i for i, t in enumerate(targets) if t is not None]
    n_valid = len(valid_indices)
    n_train = int(n_valid * 0.70)
    n_val = int(n_valid * 0.15)

    train_indices = valid_indices[:n_train]
    val_indices = valid_indices[n_train : n_train + n_val]
    test_indices = valid_indices[n_train + n_val :]

    print(f"  - Valid forecast steps: {n_valid}")
    print(f"  - Train steps: {len(train_indices)} (snapshots {train_indices[0]} to {train_indices[-1]})")
    print(f"  - Val steps:   {len(val_indices)} (snapshots {val_indices[0]} to {val_indices[-1]})")
    print(f"  - Test steps:  {len(test_indices)} (snapshots {test_indices[0]} to {test_indices[-1]})")

    # 4. Evaluate Baselines on Test Set
    print("\n[Step 4/5] Evaluating Baselines and Training Models...")

    # --- Baseline 1: Persistence ---
    persistence = PersistenceBaseline()
    p_true, p_pred = [], []
    for idx in test_indices:
        meta = meta_graphs[idx]
        target = targets[idx]
        if target is not None and len(target) > 0 and meta.num_cliques > 0:
            p_pred.extend(persistence.predict(meta).tolist())
            p_true.extend(target.tolist())
    p_metrics = compute_regression_metrics(np.array(p_true), np.array(p_pred))

    # --- Baseline 2: Feature-Only MLP ---
    mlp = FeatureOnlyMLP(in_features=5, hidden_dim=hidden_dim)
    mlp_opt = optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        mlp.train()
        for idx in train_indices:
            meta = meta_graphs[idx]
            target = targets[idx]
            if target is not None and len(target) > 0 and meta.num_cliques > 0:
                mlp_opt.zero_grad()
                pred = mlp(meta.features)
                loss = criterion(pred, target)
                loss.backward()
                mlp_opt.step()

    mlp.eval()
    mlp_true, mlp_pred = [], []
    with torch.no_grad():
        for idx in test_indices:
            meta = meta_graphs[idx]
            target = targets[idx]
            if target is not None and len(target) > 0 and meta.num_cliques > 0:
                mlp_pred.extend(mlp(meta.features).tolist())
                mlp_true.extend(target.tolist())
    mlp_metrics = compute_regression_metrics(np.array(mlp_true), np.array(mlp_pred))

    # --- Baseline 3: Static GCN ---
    static_gcn = StaticGCNModel(in_features=5, hidden_dim=hidden_dim, dropout=0.1)
    sgcn_opt = optim.Adam(static_gcn.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        static_gcn.train()
        for idx in train_indices:
            meta = meta_graphs[idx]
            target = targets[idx]
            if target is not None and len(target) > 0 and meta.num_cliques > 0:
                sgcn_opt.zero_grad()
                pred = static_gcn(meta)
                loss = criterion(pred, target)
                loss.backward()
                sgcn_opt.step()

    static_gcn.eval()
    sgcn_true, sgcn_pred = [], []
    with torch.no_grad():
        for idx in test_indices:
            meta = meta_graphs[idx]
            target = targets[idx]
            if target is not None and len(target) > 0 and meta.num_cliques > 0:
                sgcn_pred.extend(static_gcn(meta).tolist())
                sgcn_true.extend(target.tolist())
    sgcn_metrics = compute_regression_metrics(np.array(sgcn_true), np.array(sgcn_pred))

    # --- Proposed Model: Temporal GNN (GCN + GRU) ---
    temporal_gnn = TemporalGNN(
        in_features=5,
        hidden_dim=hidden_dim,
        dropout=0.1,
        jaccard_threshold=0.5,
    )
    temporal_gnn, train_losses, val_losses = train_temporal_gnn(
        model=temporal_gnn,
        meta_graphs=meta_graphs,
        targets=targets,
        train_indices=train_indices,
        val_indices=val_indices,
        epochs=epochs,
        lr=lr,
        seed=seed,
    )
    tgnn_metrics = evaluate_model(
        temporal_gnn,
        meta_graphs,
        targets,
        test_indices,
        all_node_ids=all_node_ids,
    )

    # 5. Display Benchmark Results Table
    print("\n" + "=" * 85)
    print(" [Step 5/5] FINAL HELD-OUT TEST BENCHMARK RESULTS (Future Tension Forecasting):")
    print("=" * 85)
    print(f"{'Model Architecture':<30} | {'MSE':<8} | {'RMSE':<8} | {'MAE':<8} | {'Pearson r':<10}")
    print("-" * 85)
    print(f"{'Persistence Baseline':<30} | {p_metrics['mse']:<8.4f} | {p_metrics['rmse']:<8.4f} | {p_metrics['mae']:<8.4f} | {p_metrics['pearson_r']:<10.4f}")
    print(f"{'Feature-Only MLP':<30} | {mlp_metrics['mse']:<8.4f} | {mlp_metrics['rmse']:<8.4f} | {mlp_metrics['mae']:<8.4f} | {mlp_metrics['pearson_r']:<10.4f}")
    print(f"{'Static GCN':<30} | {sgcn_metrics['mse']:<8.4f} | {sgcn_metrics['rmse']:<8.4f} | {sgcn_metrics['mae']:<8.4f} | {sgcn_metrics['pearson_r']:<10.4f}")
    print(f"{'Temporal GNN (GCN + GRU)':<30} | {tgnn_metrics['mse']:<8.4f} | {tgnn_metrics['rmse']:<8.4f} | {tgnn_metrics['mae']:<8.4f} | {tgnn_metrics['pearson_r']:<10.4f}")
    print("-" * 85)

    print("\nNode-Level Silo Risk Index (SRI) Ranking Performance (Temporal GNN on Test Set):")
    print(f"  - Spearman Rank Correlation (rho): {tgnn_metrics['avg_spearman_rho']:.4f}")
    print(f"  - Top-5 Silo Overlap:             {tgnn_metrics['avg_top5_overlap'] * 100:.1f}%")
    print(f"  - Top-10 Silo Overlap:            {tgnn_metrics['avg_top10_overlap'] * 100:.1f}%")

    # Inspect peak risk snapshot in test split
    if test_indices:
        sample_test_idx = test_indices[0]
        meta = meta_graphs[sample_test_idx]
        with torch.no_grad():
            preds, _ = temporal_gnn.forward_snapshot(meta)
            node_sri = temporal_gnn.predict_node_sri(preds, meta, all_node_ids)

        top_silo_nodes = sorted(node_sri.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop Forecasted High-Silo Individuals for Test Snapshot #{sample_test_idx}:")
        for rank, (nid, score) in enumerate(top_silo_nodes, start=1):
            email = labels.get(nid, "unknown")
            print(f"  [{rank}] Node {nid:<3} ({email:<30}) -> Forecasted SRI: {score:.2f}")

    # 6. Save Artifacts to outputs/
    out_path = Path(output_dir)
    metrics_summary = {
        "persistence_baseline": p_metrics,
        "feature_only_mlp": mlp_metrics,
        "static_gcn": sgcn_metrics,
        "temporal_gnn": tgnn_metrics,
        "parameters": {
            "window_days": window_days,
            "forecast_delta": forecast_delta,
            "epochs": epochs,
            "lr": lr,
            "hidden_dim": hidden_dim,
        },
    }
    save_artifacts(
        output_dir=out_path,
        model=temporal_gnn,
        metrics_summary=metrics_summary,
        snapshots=snapshots,
        meta_graphs=meta_graphs,
        targets=targets,
        test_indices=test_indices,
        labels=labels,
        all_node_ids=all_node_ids,
        train_losses=train_losses,
        val_losses=val_losses,
    )
    print(f"\n[Saved Artifacts] Models and evaluation reports saved to '{out_path.resolve()}':")
    print(f"  - Checkpoint:  {out_path}/checkpoints/temporal_gnn_best.pt")
    print(f"  - Metrics:     {out_path}/results/benchmark_metrics.json")
    print(f"  - Predictions: {out_path}/results/forecasted_sri_predictions.csv")
    print(f"  - Statistics:  {out_path}/results/snapshot_statistics.csv")
    print(f"  - Loss Curve:  {out_path}/results/loss_history.json")
    print("=" * 85)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Temporal GNN training & benchmarking.")
    parser.add_argument("--data_dir", type=str, default="Dataset/email-Enron")
    parser.add_argument("--window_days", type=float, default=30.0)
    parser.add_argument("--stride_days", type=float, default=30.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    sys.exit(
        run_experiment(
            data_dir=args.data_dir,
            window_days=args.window_days,
            stride_days=args.stride_days,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
