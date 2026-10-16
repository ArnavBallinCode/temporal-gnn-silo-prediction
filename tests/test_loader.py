"""
Automated unit tests for dataset loading, alignment, and validation.
"""

import tempfile
from pathlib import Path
import pytest

from src.data.loader import (
    SimplexEvent,
    ValidationReport,
    DatasetValidationError,
    DataAlignmentError,
    DataFormatError,
    discover_dataset_files,
    load_raw_vectors,
    load_node_labels,
    parse_simplex_events,
    validate_dataset,
    format_event_summary,
)


@pytest.fixture
def synthetic_dataset_dir():
    """
    Creates a temporary directory with the exact synthetic example
    from Cornell's DATA-DESCRIPTION.txt:
    Event 1: {1, 2, 3} at time 10000 ms
    Event 2: {2, 4}    at time 15000 ms
    Event 3: {1, 3, 4, 5} at time 21000 ms
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        (base / "test-nverts.txt").write_text("3\n2\n4\n", encoding="utf-8")
        (base / "test-simplices.txt").write_text("1\n2\n3\n2\n4\n1\n3\n4\n5\n", encoding="utf-8")
        (base / "test-times.txt").write_text("10000\n15000\n21000\n", encoding="utf-8")
        (base / "test-node-labels.txt").write_text(
            "1 alice@corp.com\n2 bob@corp.com\n3 carol@corp.com\n4 dave@corp.com\n5 eve@corp.com\n6 inactive@corp.com\n",
            encoding="utf-8",
        )
        yield base


def test_synthetic_valid_dataset(synthetic_dataset_dir):
    """Verifies that a valid dataset parses cleanly and aligns all vectors."""
    nverts, simplices, times = load_raw_vectors(synthetic_dataset_dir)
    labels = load_node_labels(synthetic_dataset_dir)

    assert nverts == [3, 2, 4]
    assert simplices == [1, 2, 3, 2, 4, 1, 3, 4, 5]
    assert times == [10000, 15000, 21000]
    assert len(labels) == 6
    assert labels[1] == "alice@corp.com"

    events = parse_simplex_events(nverts, simplices, times)
    assert len(events) == 3

    # Check Event 0: {1, 2, 3}
    assert events[0].event_id == 0
    assert events[0].participants == (1, 2, 3)
    assert events[0].size == 3
    assert events[0].timestamp_raw == 10000
    assert events[0].timestamp_seconds == 10.0

    # Undirected pairwise edges for {1, 2, 3} should be (1, 2), (1, 3), (2, 3)
    edges = events[0].to_pairwise_edges(directed=False)
    assert set(edges) == {(1, 2), (1, 3), (2, 3)}

    # Directed pairwise edges for {1, 2, 3} should be 6 directed pairs
    directed_edges = events[0].to_pairwise_edges(directed=True)
    assert len(directed_edges) == 6
    assert (2, 1) in directed_edges and (1, 2) in directed_edges

    # Validation report
    report = validate_dataset(nverts, simplices, times, labels, strict=True)
    assert report.is_valid
    assert report.counts_match
    assert report.sums_match
    assert report.num_unique_nodes == 5
    assert report.missing_nodes_in_events == [6]  # node 6 is inactive
    assert report.is_chronological


def test_malformed_count_mismatch():
    """Checks that a mismatch between len(nverts) and len(times) is caught."""
    nverts = [3, 2]
    times = [10000]  # Only 1 timestamp for 2 events
    simplices = [1, 2, 3, 4, 5]

    with pytest.raises(DataAlignmentError) as exc_info:
        parse_simplex_events(nverts, simplices, times)
    assert "Vector length mismatch" in str(exc_info.value)

    report = validate_dataset(nverts, simplices, times)
    assert not report.is_valid
    assert not report.counts_match
    assert any("Row count mismatch" in err for err in report.errors)


def test_malformed_sum_mismatch():
    """Checks that a mismatch between sum(nverts) and len(simplices) is caught."""
    nverts = [3, 2]  # sum is 5
    times = [10000, 20000]
    simplices = [1, 2, 3, 4]  # only 4 items instead of 5

    with pytest.raises(DataAlignmentError) as exc_info:
        parse_simplex_events(nverts, simplices, times)
    assert "Simplices count mismatch" in str(exc_info.value)

    report = validate_dataset(nverts, simplices, times)
    assert not report.is_valid
    assert not report.sums_match
    assert any("Simplices count mismatch" in err for err in report.errors)


def test_malformed_non_integer_file():
    """Checks that non-integer text values raise DataFormatError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        (base / "test-nverts.txt").write_text("3\nINVALID\n", encoding="utf-8")
        (base / "test-simplices.txt").write_text("1\n2\n3\n", encoding="utf-8")
        (base / "test-times.txt").write_text("10000\n20000\n", encoding="utf-8")

        with pytest.raises(DataFormatError) as exc_info:
            load_raw_vectors(base)
        assert "Non-integer value" in str(exc_info.value)


def test_invalid_node_id():
    """Checks that non-positive node IDs (e.g. 0 or negative) trigger validation errors."""
    nverts = [2]
    times = [10000]
    simplices = [0, 2]  # 0 is invalid for 1-indexed node IDs

    report = validate_dataset(nverts, simplices, times)
    assert not report.is_valid
    assert any("Invalid non-positive node ID" in err for err in report.errors)


def test_timestamp_inversions_detection():
    """Checks that non-chronological events produce a clear warning with inversion count."""
    nverts = [2, 2, 2]
    times = [30000, 10000, 20000]  # Inversion at step 0->1
    simplices = [1, 2, 2, 3, 3, 4]

    report = validate_dataset(nverts, simplices, times)
    assert report.is_valid  # Warnings do not block validity
    assert not report.is_chronological
    assert report.timestamp_inversions == 1
    assert any("not strictly non-decreasing" in w for w in report.warnings)


def test_size_one_events_handling():
    """Checks that size-1 events produce appropriate pairwise edges and warnings."""
    ev = SimplexEvent(event_id=0, timestamp_raw=1000, participants=(5,))
    assert ev.size == 1
    assert ev.to_pairwise_edges(remove_self_loops=True) == []
    assert ev.to_pairwise_edges(remove_self_loops=False) == [(5, 5)]

    nverts = [1, 2]
    times = [1000, 2000]
    simplices = [5, 1, 2]
    report = validate_dataset(nverts, simplices, times)
    assert any("Detected 1 events of size 1" in w for w in report.warnings)


def test_real_dataset_smoke():
    """
    Smoke test running on the actual Dataset/email-Enron directory.
    Verifies that the Enron files load correctly and match the known metrics.
    """
    real_data_dir = Path("Dataset/email-Enron")
    if not real_data_dir.exists():
        pytest.skip("Dataset/email-Enron directory not found")

    nverts, simplices, times = load_raw_vectors(real_data_dir)
    labels = load_node_labels(real_data_dir)

    assert len(nverts) == 10883
    assert len(times) == 10883
    assert len(simplices) == 26841
    assert sum(nverts) == 26841
    assert len(labels) == 148

    report = validate_dataset(nverts, simplices, times, labels)
    assert report.is_valid
    assert report.counts_match
    assert report.sums_match
    assert report.num_unique_nodes == 143
    assert report.min_node_id == 1
    assert report.max_node_id == 148
    assert len(report.missing_nodes_in_events) == 5  # Nodes [10, 16, 31, 35, 124]
    assert report.timestamp_inversions == 115
    assert report.duplicate_events == 2
