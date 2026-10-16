#!/usr/bin/env python3
"""
Graph Snapshot Construction CLI Script.

 
Usage:
    python3 src/data/build_snapshots.py [--window_days 30] [--stride_days 30]

Constructs time-windowed communication graphs from email events and prints
structural statistics for each snapshot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import (
    load_node_labels,
    load_raw_vectors,
    parse_simplex_events,
)
from src.data.snapshots import build_graph_snapshots


def run_snapshot_pipeline(
    data_dir: str | Path = "Dataset/email-Enron",
    window_days: float = 30.0,
    stride_days: float = 30.0,
    min_active_nodes: int = 2,
) -> int:
    """
    Loads dataset, builds snapshots, and displays an educational summary table.
    """
    data_path = Path(data_dir)
    print("=" * 85)
    print(" TEMPORAL GNN - GRAPH SNAPSHOT CONSTRUCTION (STAGE 2)")
    print(f"  ")
    print(f" Window Size:        {window_days} days")
    print(f" Stride Size:        {stride_days} days")
    print("=" * 85)

    # 1. Load data
    try:
        nverts, simplices, times = load_raw_vectors(data_path)
        labels = load_node_labels(data_path)
        events = parse_simplex_events(nverts, simplices, times)
    except Exception as err:
        print(f"[ERROR] Failed to load data: {err}")
        return 1

    print(f"Loaded {len(events):,} email events across {len(labels)} labeled employees.\n")

    # 2. Build snapshots
    all_node_ids = sorted(labels.keys())
    snapshots = build_graph_snapshots(
        events=events,
        window_days=window_days,
        stride_days=stride_days,
        all_node_ids=all_node_ids,
        include_isolated_nodes=False,
        sort_events=True,
    )

    print(f"Generated {len(snapshots)} snapshots over the full timeline.\n")
    print(
        f"{'ID':<4} | {'Start Date':<10} -> {'End Date':<10} | "
        f"{'Events':<6} | {'Nodes':<5} | {'Edges':<6} | {'Density':<7} | {'Components':<10} | {'Max Weight'}"
    )
    print("-" * 85)

    non_empty_count = 0
    total_edges = 0

    for snap in snapshots:
        # Determine maximum edge weight in this snapshot
        max_w = 0
        if snap.num_edges > 0:
            max_w = max(d.get("weight", 1) for _, _, d in snap.graph.edges(data=True))

        # Only count snapshots with at least min_active_nodes
        if snap.num_nodes >= min_active_nodes:
            non_empty_count += 1
            total_edges += snap.num_edges

        print(
            f"{snap.snapshot_id:<4} | {snap.start_date_str[:10]:<10} -> {snap.end_date_str[:10]:<10} | "
            f"{snap.num_events:<6} | {snap.num_nodes:<5} | {snap.num_edges:<6} | "
            f"{snap.density:<7.4f} | {snap.num_connected_components:<10} | {max_w:<10}"
        )

    print("-" * 85)
    print(f"\nSummary of Snapshot Sequence:")
    print(f"  - Total snapshots generated: {len(snapshots)}")
    print(f"  - Active snapshots (>= {min_active_nodes} nodes): {non_empty_count}")
    print(f"  - Average active nodes per snapshot: {sum(s.num_nodes for s in snapshots) / len(snapshots):.1f}")
    print(f"  - Average edges per snapshot: {sum(s.num_edges for s in snapshots) / len(snapshots):.1f}")
    print("=" * 85)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and inspect temporal graph snapshots.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Dataset/email-Enron",
        help="Path to dataset directory (default: Dataset/email-Enron)",
    )
    parser.add_argument(
        "--window_days",
        type=float,
        default=30.0,
        help="Snapshot time window in days (default: 30.0)",
    )
    parser.add_argument(
        "--stride_days",
        type=float,
        default=30.0,
        help="Window step size in days (default: 30.0)",
    )
    args = parser.parse_args()

    sys.exit(run_snapshot_pipeline(args.data_dir, args.window_days, args.stride_days))


if __name__ == "__main__":
    main()
