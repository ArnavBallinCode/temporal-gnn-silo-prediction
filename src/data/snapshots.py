"""
Temporal Graph Snapshots Construction.

 

Purpose of this module:
-----------------------
Raw communication data is a stream of timestamped email events.
To analyze how organizational communication networks evolve over time, we partition
the timeline into discrete time windows (e.g. 30 days) and build a graph snapshot G^(k)
for each window.

Mathematical Operations:
------------------------
1. Higher-Order Simplex to Pairwise Clique Projection:
   An email with k participants {v_1, v_2, ..., v_k} is projected into all k*(k-1)/2
   undirected edges: (v_i, v_j) for all i < j.
   - Why? Standard graph neural networks and clique detection algorithms operate on pairwise graphs.
   - If two people appear in multiple emails within the same window, the edge weight is incremented.

2. Discrete Time Windowing:
   Given window size delta_w (in milliseconds) and stride delta_s:
   Window k covers the time interval [t_start, t_end) where:
     t_start = t_0 + k * delta_s
     t_end   = t_start + delta_w
   All communication events with t_start <= timestamp < t_end belong to snapshot G^(k).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import networkx as nx

from src.data.loader import CORNELL_EPOCH_DATETIME, SimplexEvent

# 1 day in milliseconds = 24 hours * 60 mins * 60 secs * 1000 ms = 86,400,000 ms
MS_PER_DAY = 86_400_000


@dataclass
class GraphSnapshot:
    """
    Represents a single graph snapshot G^(k) over a time window.

    Attributes:
        snapshot_id: Integer index k (0, 1, 2, ...).
        start_time_ms: Start of the window in raw milliseconds.
        end_time_ms: End of the window in raw milliseconds.
        graph: NetworkX undirected Graph (nx.Graph) containing active edges and weights.
        num_events: Total number of email events that occurred in this window.
    """
    snapshot_id: int
    start_time_ms: int
    end_time_ms: int
    graph: nx.Graph
    num_events: int

    @property
    def num_nodes(self) -> int:
        """Number of active employees who communicated in this window."""
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        """Number of unique employee-to-employee communication ties."""
        return self.graph.number_of_edges()

    @property
    def density(self) -> float:
        """
        Graph density: actual edges / possible edges.
        Formula: 2 * |E| / (|V| * (|V| - 1)) for an undirected graph.
        """
        n = self.num_nodes
        if n < 2:
            return 0.0
        possible_edges = n * (n - 1) / 2.0
        return self.num_edges / possible_edges

    @property
    def num_connected_components(self) -> int:
        """Number of isolated connected components in the communication network."""
        if self.num_nodes == 0:
            return 0
        return nx.number_connected_components(self.graph)

    @property
    def start_date_str(self) -> str:
        """Human-readable start date (UTC)."""
        sec = self.start_time_ms / 1000.0
        dt = CORNELL_EPOCH_DATETIME + datetime.timedelta(seconds=sec)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def end_date_str(self) -> str:
        """Human-readable end date (UTC)."""
        sec = self.end_time_ms / 1000.0
        dt = CORNELL_EPOCH_DATETIME + datetime.timedelta(seconds=sec)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def summary(self) -> Dict[str, object]:
        """Returns a dictionary summarizing the key structural metrics of this snapshot."""
        return {
            "snapshot_id": self.snapshot_id,
            "start_date": self.start_date_str,
            "end_date": self.end_date_str,
            "events": self.num_events,
            "active_nodes": self.num_nodes,
            "edges": self.num_edges,
            "density": round(self.density, 4),
            "components": self.num_connected_components,
        }


def project_event_to_edges(
    event: SimplexEvent,
    remove_self_loops: bool = True,
) -> List[Tuple[int, int]]:
    """
    Converts a single higher-order email event into pairwise undirected edges.

    Example:
        An email sent with participants {Node 1, Node 2, Node 3}
        generates 3 pairwise edges:
        (1, 2), (1, 3), and (2, 3).

    Mathematical rationale:
        If k employees are on the same email, they form a k-clique in the
        instantaneous communication event.
    """
    nodes = list(event.participants)
    k = len(nodes)
    edges: List[Tuple[int, int]] = []

    # Size-1 events have only 1 participant (e.g. self-loop or non-Enron recipient).
    if k <= 1:
        if not remove_self_loops and k == 1:
            edges.append((nodes[0], nodes[0]))
        return edges

    # Generate all pairwise combinations u < v
    for i in range(k):
        for j in range(i + 1, k):
            u, v = nodes[i], nodes[j]
            # Standardize order: smaller ID first for undirected representation
            if u < v:
                edges.append((u, v))
            else:
                edges.append((v, u))

    return edges


def build_graph_snapshots(
    events: List[SimplexEvent],
    window_days: float = 30.0,
    stride_days: Optional[float] = None,
    all_node_ids: Optional[List[int]] = None,
    include_isolated_nodes: bool = False,
    sort_events: bool = True,
) -> List[GraphSnapshot]:
    """
    Partitions the event stream into sequential graph snapshots.

    Args:
        events: List of SimplexEvent objects from the dataset loader.
        window_days: Time window length for each snapshot in days (default: 30 days).
        stride_days: Step size to move the window. If None, defaults to window_days
                     (non-overlapping contiguous partitions [t0, t1), [t1, t2), ...).
        all_node_ids: Optional list of all known employee IDs (e.g. 1 to 148).
        include_isolated_nodes: If True, adds all known nodes to every graph even if
                                they had 0 emails in that window (degree 0).
                                If False (default), only active communicating nodes are added.
        sort_events: If True, sorts events by timestamp first to guarantee monotonicity.

    Returns:
        List of GraphSnapshot objects.
    """
    if not events:
        return []

    # Step 1: Ensure events are sorted chronologically
    # The paper (Section III-A.1) assumes t_i <= t_j for i < j.
    if sort_events:
        clean_events = sorted(events, key=lambda e: e.timestamp_raw)
    else:
        clean_events = events

    # Step 2: Determine global time range
    min_time_ms = clean_events[0].timestamp_raw
    max_time_ms = clean_events[-1].timestamp_raw

    # Convert window and stride days into milliseconds
    window_ms = int(window_days * MS_PER_DAY)
    stride_ms = int((stride_days if stride_days is not None else window_days) * MS_PER_DAY)

    snapshots: List[GraphSnapshot] = []
    snapshot_id = 0
    current_start_ms = min_time_ms

    # Step 3: Iterate through timeline in windows
    while current_start_ms <= max_time_ms:
        current_end_ms = current_start_ms + window_ms

        # Create a new empty undirected graph for this window
        G = nx.Graph()

        # If user wants all employees present (including degree-0 isolates)
        if include_isolated_nodes and all_node_ids:
            for node_id in all_node_ids:
                G.add_node(node_id)

        # Collect events within [current_start_ms, current_end_ms)
        event_count_in_window = 0
        for ev in clean_events:
            # Events are sorted, so we can check window boundaries
            if ev.timestamp_raw < current_start_ms:
                continue
            if ev.timestamp_raw >= current_end_ms:
                # Since events are sorted, once we pass current_end_ms we could break,
                # but if sliding windows are used, simple check is safe and fast.
                continue

            event_count_in_window += 1

            # Project simplex to pairwise edges
            pairwise_edges = project_event_to_edges(ev, remove_self_loops=True)
            for u, v in pairwise_edges:
                if G.has_edge(u, v):
                    # Increment interaction weight (number of shared emails)
                    G[u][v]["weight"] += 1
                else:
                    # New communication tie in this window
                    G.add_edge(u, v, weight=1)

        # Record this snapshot
        snapshot = GraphSnapshot(
            snapshot_id=snapshot_id,
            start_time_ms=current_start_ms,
            end_time_ms=current_end_ms,
            graph=G,
            num_events=event_count_in_window,
        )
        snapshots.append(snapshot)

        # Move window forward by stride
        snapshot_id += 1
        current_start_ms += stride_ms

    return snapshots
