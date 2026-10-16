#!/usr/bin/env python3
"""
Maximal Clique Extraction CLI & Inspection Tool.

 
Usage:
    python3 src/graph/inspect_cliques.py [--window_days 30] [--top_k 30] [--min_size 3]

Extracts maximal cliques for each snapshot, reports counts and size distributions,
and displays sample cohesive teams.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import (
    load_node_labels,
    load_raw_vectors,
    parse_simplex_events,
)
from src.data.snapshots import build_graph_snapshots
from src.graph.cliques import extract_maximal_cliques, format_clique_str


def run_clique_inspection(
    data_dir: str | Path = "Dataset/email-Enron",
    window_days: float = 30.0,
    stride_days: float = 30.0,
    min_size: int = 3,
    top_k: int = 30,
) -> int:
    """
    Builds snapshots, extracts maximal cliques per snapshot, and prints summary.
    """
    data_path = Path(data_dir)
    print("=" * 85)
    print(" TEMPORAL GNN - MAXIMAL CLIQUE EXTRACTION (STAGE 3)")
    print(f"  ")
    print(f" Settings:           window={window_days}d, min_size={min_size}, top_k={top_k}")
    print("=" * 85)

    # 1. Load data & snapshots
    try:
        nverts, simplices, times = load_raw_vectors(data_path)
        labels = load_node_labels(data_path)
        events = parse_simplex_events(nverts, simplices, times)
        snapshots = build_graph_snapshots(events, window_days=window_days, stride_days=stride_days)
    except Exception as err:
        print(f"[ERROR] Failed to prepare snapshots: {err}")
        return 1

    print(f"Loaded {len(events):,} events across {len(snapshots)} snapshots.\n")
    print(
        f"{'Snap':<5} | {'Date Interval':<23} | {'Nodes':<5} | {'Edges':<6} | "
        f"{'Cliques':<8} | {'Max Size':<8} | {'Avg Size':<8}"
    )
    print("-" * 85)

    snapshots_with_cliques = 0
    total_cliques_extracted = 0

    peak_snapshot_id = -1
    max_clique_ever_size = 0
    peak_cliques = []

    for s in snapshots:
        cliques = extract_maximal_cliques(s.graph, min_size=min_size, top_k=top_k)
        num_cliques = len(cliques)

        if num_cliques > 0:
            snapshots_with_cliques += 1
            total_cliques_extracted += num_cliques
            sizes = [len(c) for c in cliques]
            max_sz = max(sizes)
            avg_sz = sum(sizes) / len(sizes)
            if max_sz > max_clique_ever_size:
                max_clique_ever_size = max_sz
                peak_snapshot_id = s.snapshot_id
                peak_cliques = cliques
        else:
            max_sz = 0
            avg_sz = 0.0

        print(
            f"{s.snapshot_id:<5} | {s.start_date_str[:10]} -> {s.end_date_str[:10]} | "
            f"{s.num_nodes:<5} | {s.num_edges:<6} | {num_cliques:<8} | {max_sz:<8} | {avg_sz:<8.2f}"
        )

    print("-" * 85)
    print("\nSummary of Maximal Clique Extraction:")
    print(f"  - Snapshots containing cliques (>= {min_size} nodes): {snapshots_with_cliques} / {len(snapshots)}")
    print(f"  - Total maximal cliques retained across timeline: {total_cliques_extracted}")
    print(f"  - Peak clique size observed: {max_clique_ever_size} nodes (in snapshot {peak_snapshot_id})")

    if peak_cliques:
        print(f"\nSample Top Cliques from Peak Snapshot #{peak_snapshot_id}:")
        for i, c in enumerate(peak_cliques[:5], start=1):
            print(f"  [{i}] {format_clique_str(c, labels)}")

    print("=" * 85)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and inspect maximal cliques per snapshot.")
    parser.add_argument("--data_dir", type=str, default="Dataset/email-Enron")
    parser.add_argument("--window_days", type=float, default=30.0)
    parser.add_argument("--stride_days", type=float, default=30.0)
    parser.add_argument("--min_size", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=30)
    args = parser.parse_args()

    sys.exit(run_clique_inspection(args.data_dir, args.window_days, args.stride_days, args.min_size, args.top_k))


if __name__ == "__main__":
    main()
