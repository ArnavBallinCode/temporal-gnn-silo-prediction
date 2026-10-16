"""
Automated unit tests for graph snapshot construction and simplex projection.

"""

from pathlib import Path
import pytest
import networkx as nx

from src.data.loader import (
    SimplexEvent,
    load_raw_vectors,
    parse_simplex_events,
)
from src.data.snapshots import (
    GraphSnapshot,
    MS_PER_DAY,
    build_graph_snapshots,
    project_event_to_edges,
)


def test_project_event_three_participants():
    """
    Hand-calculated test:
    A 3-participant email {1, 2, 3} must generate exactly 3 undirected edges:
    (1, 2), (1, 3), and (2, 3).
    """
    event = SimplexEvent(event_id=0, timestamp_raw=1000, participants=(1, 2, 3))
    edges = project_event_to_edges(event, remove_self_loops=True)

    assert len(edges) == 3
    assert set(edges) == {(1, 2), (1, 3), (2, 3)}


def test_project_event_size_one():
    """
    A 1-participant event (self-loop) should produce no edges when remove_self_loops=True.
    """
    event = SimplexEvent(event_id=0, timestamp_raw=1000, participants=(5,))
    edges = project_event_to_edges(event, remove_self_loops=True)
    assert edges == []

    # When self-loops are explicitly preserved:
    edges_with_loops = project_event_to_edges(event, remove_self_loops=False)
    assert edges_with_loops == [(5, 5)]


def test_repeated_interactions_weight_aggregation():
    """
    Verifies that multiple emails between the same two people in one window
    increment the edge weight.
    """
    ev1 = SimplexEvent(event_id=0, timestamp_raw=1000, participants=(1, 2))
    ev2 = SimplexEvent(event_id=1, timestamp_raw=2000, participants=(1, 2))
    ev3 = SimplexEvent(event_id=2, timestamp_raw=3000, participants=(2, 3))

    # All 3 events happen within the first 30 days
    snapshots = build_graph_snapshots([ev1, ev2, ev3], window_days=30.0)

    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.num_nodes == 3  # nodes 1, 2, 3
    assert snap.num_edges == 2  # edge (1, 2) and edge (2, 3)
    assert snap.graph[1][2]["weight"] == 2
    assert snap.graph[2][3]["weight"] == 1


def test_time_window_partitioning():
    """
    Tests that events are partitioned into the correct chronological windows.
    Event 1 at 5 days   -> Window 0 [0, 30)
    Event 2 at 25 days  -> Window 0 [0, 30)
    Event 3 at 45 days  -> Window 1 [30, 60)
    Event 4 at 80 days  -> Window 2 [60, 90)
    """
    t0 = 100_000_000  # arbitrary base
    ev1 = SimplexEvent(event_id=0, timestamp_raw=t0 + 5 * MS_PER_DAY, participants=(1, 2))
    ev2 = SimplexEvent(event_id=1, timestamp_raw=t0 + 25 * MS_PER_DAY, participants=(2, 3))
    ev3 = SimplexEvent(event_id=2, timestamp_raw=t0 + 45 * MS_PER_DAY, participants=(3, 4))
    ev4 = SimplexEvent(event_id=3, timestamp_raw=t0 + 80 * MS_PER_DAY, participants=(4, 5))

    snapshots = build_graph_snapshots([ev1, ev2, ev3, ev4], window_days=30.0)

    assert len(snapshots) == 3
    assert snapshots[0].num_events == 2
    assert set(snapshots[0].graph.nodes()) == {1, 2, 3}

    assert snapshots[1].num_events == 1
    assert set(snapshots[1].graph.nodes()) == {3, 4}

    assert snapshots[2].num_events == 1
    assert set(snapshots[2].graph.nodes()) == {4, 5}


def test_graph_snapshot_structural_metrics():
    """
    Hand-calculated test of structural metrics:
    A graph with triangle {1, 2, 3} and separate edge {4, 5}:
    - Nodes: 5
    - Edges: 4 (three in triangle + one outside)
    - Connected components: 2
    - Density: 4 / (5 * 4 / 2) = 4 / 10 = 0.4
    """
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1, 3), (4, 5)])

    snap = GraphSnapshot(
        snapshot_id=0,
        start_time_ms=0,
        end_time_ms=1000,
        graph=G,
        num_events=2,
    )

    assert snap.num_nodes == 5
    assert snap.num_edges == 4
    assert snap.num_connected_components == 2
    assert snap.density == pytest.approx(0.4)


def test_real_snapshots_smoke():
    """
    Smoke test: build snapshots from the actual Enron dataset and check basic integrity.
    """
    real_data_dir = Path("Dataset/email-Enron")
    if not real_data_dir.exists():
        pytest.skip("Dataset/email-Enron directory not found")

    nverts, simplices, times = load_raw_vectors(real_data_dir)
    events = parse_simplex_events(nverts, simplices, times)

    # Build 30-day snapshots
    snapshots = build_graph_snapshots(events, window_days=30.0)

    # Span is 1307 days, so ~44 30-day snapshots
    assert 40 <= len(snapshots) <= 50
    # At least some snapshots must have active edges
    total_edges = sum(s.num_edges for s in snapshots)
    assert total_edges > 1000
