#!/usr/bin/env python3
"""
Dataset Inspection and Validation CLI Script.


Usage:
    python3 src/data/inspect_dataset.py [--data_dir Dataset/email-Enron]

This script performs Stage 1 validation on the higher-order temporal email dataset,
explaining each check, printing decoded events, and providing recommendations
for graph snapshot construction in Stage 2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path if running directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import (
    discover_dataset_files,
    format_event_summary,
    load_node_labels,
    load_raw_vectors,
    parse_simplex_events,
    validate_dataset,
)


def run_inspection(data_dir: str | Path, show_samples: int = 8) -> int:
    """
    Runs full dataset inspection and prints a human-readable, educational report.
    """
    data_path = Path(data_dir)
    print("=" * 80)
    print(" TEMPORAL GNN - DATASET INSPECTION & VALIDATION REPORT (STAGE 1)")
    print(f"  ")
    print(f" Target Directory:   {data_path.resolve()}")
    print("=" * 80)

    # 1. Discover files
    try:
        file_map = discover_dataset_files(data_path)
    except FileNotFoundError as err:
        print(f"\n[CRITICAL ERROR] File discovery failed: {err}")
        return 1

    print("\n1. LOCATED DATASET FILES:")
    for key, path in sorted(file_map.items()):
        size_kb = path.stat().st_size / 1024.0
        print(f"  - {key:<12}: {path.name} ({size_kb:.1f} KB)")

    # 2. Load raw data
    print("\n2. LOADING RAW INTEGER VECTORS...")
    try:
        nverts, simplices, times = load_raw_vectors(data_path)
        labels = load_node_labels(data_path)
    except Exception as err:
        print(f"\n[CRITICAL ERROR] Loading vectors failed: {err}")
        return 1

    print(f"  - Lines in nverts.txt:    {len(nverts):,}")
    print(f"  - Lines in times.txt:     {len(times):,}")
    print(f"  - Lines in simplices.txt: {len(simplices):,}")
    print(f"  - Entries in node-labels: {len(labels):,}")

    # 3. Perform Validation
    print("\n3. RUNNING INTEGRITY & ALIGNMENT CHECKS:")
    report = validate_dataset(nverts, simplices, times, labels, strict=False)

    # Check 1: Count alignment
    c_status = "PASS" if report.counts_match else "FAIL"
    print(f"  [{c_status}] Event Count vs Timestamp Count:")
    print(f"         len(nverts) = {report.num_events:,} == len(times) = {report.num_timestamps:,}")

    # Check 2: Sum alignment
    s_status = "PASS" if report.sums_match else "FAIL"
    print(f"  [{s_status}] Sum of Simplex Vertices vs Simplices File Length:")
    print(f"         sum(nverts) = {report.sum_nverts:,} == len(simplices) = {report.num_simplices_nodes:,}")

    # Check 3: Node IDs
    n_status = "PASS" if report.min_node_id > 0 else "FAIL"
    print(f"  [{n_status}] Node ID Range & Indexing:")
    print(f"         Min Node ID = {report.min_node_id}, Max Node ID = {report.max_node_id}")
    print(f"         Unique Nodes in Events = {report.num_unique_nodes} (out of {report.num_node_labels} labeled employees)")

    # Check 4: Timestamps & Chronology
    t_status = "PASS" if report.is_chronological else "WARNING (Expected in raw data)"
    print(f"  [{t_status}] Timestamp Chronological Monotonicity:")
    print(f"         Chronological: {report.is_chronological} (detected {report.timestamp_inversions} inversions)")
    print(f"         Min Timestamp: {report.min_timestamp:,}")
    print(f"         Max Timestamp: {report.max_timestamp:,}")
    print(f"         Start Date:    {report.start_date}")
    print(f"         End Date:      {report.end_date}")
    print(f"         Time Span:     {report.time_span_days:.2f} days (~{report.time_span_days / 365.25:.2f} years)")

    # Check 5: Duplicate and internal duplicates
    d_status = "PASS" if report.internal_duplicate_node_events == 0 else "WARNING"
    print(f"  [{d_status}] Simplex Internal Duplicate Nodes:")
    print(f"         Events with duplicated nodes within simplex: {report.internal_duplicate_node_events}")
    print(f"         Unique participant sets:                    {report.unique_participant_sets:,}")
    print(f"         Exact duplicate (timestamp, nodes) events:  {report.duplicate_events}")

    # 4. Simplex size distribution
    print("\n4. SIMPLEX PARTICIPANT SIZE DISTRIBUTION:")
    print(f"  {'Size (k)':<10} | {'Count':<8} | {'Percentage':<10} | {'Interpretation'}")
    print("  " + "-" * 65)
    sorted_sizes = sorted(report.size_distribution.items(), key=lambda x: x[0])
    for sz, count in sorted_sizes:
        pct = (count / report.num_events) * 100.0
        if sz == 1:
            desc = "Self-communication / non-Enron recipient"
        elif sz == 2:
            desc = "Direct pairwise 1-on-1 email"
        elif sz == 3:
            desc = "Triadic communication (e.g. 1 sender, 2 recipients)"
        else:
            desc = f"Higher-order group email ({sz} participants)"
        print(f"  {sz:<10} | {count:<8,} | {pct:>6.2f}%    | {desc}")

    # 5. Inactive employees
    if report.missing_nodes_in_events:
        print("\n5. INACTIVE EMPLOYEES (In address book, but 0 events in communication log):")
        for node_id in report.missing_nodes_in_events:
            addr = labels.get(node_id, "unknown")
            print(f"  - Node {node_id:<3}: {addr}")

    # 6. Decoded Sample Events
    events = parse_simplex_events(nverts, simplices, times)
    print(f"\n6. DECODED SAMPLE EMAIL EVENTS (First {min(show_samples, len(events))} events):")
    for ev in events[:show_samples]:
        print("  " + format_event_summary(ev, labels))

    # 7. Recommendations
    print("\n" + "=" * 80)
    print(" RECOMMENDATIONS FOR DOWNSTREAM STAGES (Stage 2 - Graph Snapshots):")
    print("=" * 80)
    print("  1. Timestamp Sorting:")
    print("     The paper specifies (Section III-A.1): 'We assume t_i <= t_j for i < j after sorting by time.'")
    print(f"     Raw data has {report.timestamp_inversions} slight inversions. We should sort events by timestamp")
    print("     before windowing/snapshotting.")
    print("  2. Handling Size-1 Events (k=1):")
    print(f"     There are {report.size_distribution.get(1, 0)} size-1 events. In pairwise projection, size-1 events")
    print("     do not form pairwise edges between distinct employees. We will filter them out for clique")
    print("     conflict analysis, as a clique requires at least 2 distinct nodes (or >= 3 as configured).")
    print("  3. Handling Duplicate Events:")
    print(f"     There are {report.duplicate_events} duplicate events with identical (timestamp, participants).")
    print("     In unweighted pairwise graphs, these naturally deduplicate; in weighted graphs, they represent")
    print("     bursts / repeat communications.")
    print("  4. Inactive Nodes:")
    print(f"     5 employees in the 148-node roster have 0 communications in this period. Graph snapshots")
    print("     can either include them as isolated degree-0 nodes or index active participants dynamically.")
    print("=" * 80)

    if report.errors:
        print(f"\n[STATUS: FAILED] {len(report.errors)} blocking errors detected.")
        for err in report.errors:
            print(f"  - {err}")
        return 1
    else:
        print("\n[STATUS: SUCCESS] Dataset verified and ready for Stage 2.")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and validate temporal higher-order email dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Dataset/email-Enron",
        help="Path to directory containing dataset files (default: Dataset/email-Enron)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="Number of decoded sample events to display (default: 8)",
    )
    args = parser.parse_args()

    sys.exit(run_inspection(args.data_dir, show_samples=args.samples))


if __name__ == "__main__":
    main()
