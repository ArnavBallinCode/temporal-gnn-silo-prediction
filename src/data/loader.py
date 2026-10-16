"""
Temporal Graph Neural Network - Dataset Loader and Validator.

This module is responsible for loading, aligning, and validating the Cornell higher-order
temporal email communication dataset (e.g., email-Enron).

Mathematical Formulation & File Alignment:
------------------------------------------
In higher-order communication networks, each communication event (e.g., an email with
senders and multiple recipients) is modeled as a simplex or hyperedge among a set of nodes.
The dataset separates this structure into three aligned integer vectors:

1. `dataset-nverts.txt` (length M):
   The vector n = [n_1, n_2, ..., n_M] where n_i is the number of participants in event i.
2. `dataset-simplices.txt` (length S = sum_{i=1}^M n_i):
   The contiguous concatenated sequence of participant node IDs.
   Event i corresponds to the slice: simplices[offset_i : offset_i + n_i],
   where offset_i = sum_{j=1}^{i-1} n_j.
3. `dataset-times.txt` (length M):
   The vector t = [t_1, t_2, ..., t_M] where t_i is the timestamp of event i.
   Timestamps are measured in milliseconds from calendar year 1 AD (1998-2002 period).
4. `dataset-node-labels.txt`:
   Mapping from integer node ID (1-indexed) to employee email address.
"""

from __future__ import annotations

import datetime
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class DatasetValidationError(Exception):
    """Raised when dataset integrity or alignment checks fail."""
    pass


class DataAlignmentError(DatasetValidationError):
    """Raised specifically when file lengths or slice offsets do not align."""
    pass


class DataFormatError(DatasetValidationError):
    """Raised when data lines cannot be parsed as expected types."""
    pass


# Gregorian Year 1 AD epoch used by Cornell dataset
# 63046642020000 ms corresponds to 1998-11-14 12:07:00 UTC
CORNELL_EPOCH_DATETIME = datetime.datetime(1, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)


@dataclass(frozen=True)
class SimplexEvent:
    """
    Represents a single timestamped email communication event (simplex).

    Attributes:
        event_id: 0-indexed sequence index from the raw file.
        timestamp_raw: Timestamp in milliseconds as recorded in dataset-times.txt.
        participants: Tuple of participant employee node IDs (1-indexed integers).
    """
    event_id: int
    timestamp_raw: int
    participants: Tuple[int, ...]

    @property
    def size(self) -> int:
        """Number of participants in this email event."""
        return len(self.participants)

    @property
    def timestamp_seconds(self) -> float:
        """Timestamp converted to seconds."""
        return self.timestamp_raw / 1000.0

    @property
    def datetime(self) -> datetime.datetime:
        """
        Converts the Cornell raw millisecond timestamp to a Python datetime object.
        Raw values count seconds/ms from Gregorian calendar Year 1 (0001-01-01).
        """
        seconds_from_year_one = self.timestamp_raw / 1000.0
        return CORNELL_EPOCH_DATETIME + datetime.timedelta(seconds=seconds_from_year_one)

    def to_pairwise_edges(self, directed: bool = False, remove_self_loops: bool = True) -> List[Tuple[int, int]]:
        """
        Projects this higher-order communication event into pairwise graph edges.

        For a simplex with k participants {v_1, v_2, ..., v_k}:
        - If directed is False: generates all k*(k-1)/2 undirected pairs (u, v) with u < v.
        - If directed is True: generates all ordered pairs (u, v) with u != v.
        - Size-1 simplices (single participant) produce self-loops (v, v) if remove_self_loops=False.
        """
        nodes = self.participants
        edges: List[Tuple[int, int]] = []
        k = len(nodes)

        if k == 1:
            if not remove_self_loops:
                edges.append((nodes[0], nodes[0]))
            return edges

        for i in range(k):
            for j in range(i + 1, k):
                u, v = nodes[i], nodes[j]
                if directed:
                    edges.append((u, v))
                    edges.append((v, u))
                else:
                    edges.append((min(u, v), max(u, v)))
        return edges


@dataclass
class ValidationReport:
    """
    Summary report of dataset inspection and integrity checks.
    """
    num_events: int
    num_timestamps: int
    sum_nverts: int
    num_simplices_nodes: int
    counts_match: bool
    sums_match: bool
    num_unique_nodes: int
    min_node_id: int
    max_node_id: int
    num_node_labels: int
    missing_nodes_in_events: List[int]
    min_timestamp: int
    max_timestamp: int
    time_span_days: float
    start_date: str
    end_date: str
    is_chronological: bool
    timestamp_inversions: int
    size_distribution: Dict[int, int]
    unique_participant_sets: int
    duplicate_events: int
    internal_duplicate_node_events: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if there are zero blocking errors."""
        return len(self.errors) == 0


def discover_dataset_files(data_dir: str | Path) -> Dict[str, Path]:
    """
    Locates the required dataset files within data_dir, supporting standard prefixes.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_path.resolve()}")

    files: Dict[str, Path] = {}

    # Look for nverts
    nverts_candidates = list(data_path.glob("*nverts.txt"))
    if not nverts_candidates:
        raise FileNotFoundError(f"Could not find *nverts.txt in {data_path}")
    files["nverts"] = nverts_candidates[0]

    # Look for simplices
    simplices_candidates = list(data_path.glob("*simplices.txt"))
    if not simplices_candidates:
        raise FileNotFoundError(f"Could not find *simplices.txt in {data_path}")
    files["simplices"] = simplices_candidates[0]

    # Look for times
    times_candidates = list(data_path.glob("*times.txt"))
    if not times_candidates:
        raise FileNotFoundError(f"Could not find *times.txt in {data_path}")
    files["times"] = times_candidates[0]

    # Look for node labels or addresses
    label_candidates = list(data_path.glob("*node-labels.txt")) or list(data_path.glob("*addresses.txt"))
    if label_candidates:
        files["node_labels"] = label_candidates[0]

    return files


def load_raw_vectors(data_dir: str | Path) -> Tuple[List[int], List[int], List[int]]:
    """
    Reads the three raw integer vectors from the dataset directory.

    Returns:
        (nverts, simplices, times)
    Raises:
        FileNotFoundError: If a required file is missing.
        DataFormatError: If any line cannot be parsed as an integer.
    """
    file_map = discover_dataset_files(data_dir)

    def _read_int_file(path: Path) -> List[int]:
        values: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue  # skip empty lines
                try:
                    values.append(int(stripped))
                except ValueError as exc:
                    raise DataFormatError(
                        f"Non-integer value on line {line_no} of {path.name}: '{stripped}'"
                    ) from exc
        return values

    nverts = _read_int_file(file_map["nverts"])
    simplices = _read_int_file(file_map["simplices"])
    times = _read_int_file(file_map["times"])

    return nverts, simplices, times


def load_node_labels(data_dir: str | Path) -> Dict[int, str]:
    """
    Loads the mapping from integer node ID to email address / label.

    Returns:
        Dict[node_id, email_address]
    """
    file_map = discover_dataset_files(data_dir)
    labels: Dict[int, str] = {}

    if "node_labels" not in file_map:
        return labels

    with open(file_map["node_labels"], "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    node_id = int(parts[0])
                    address = parts[1]
                    labels[node_id] = address
                except ValueError:
                    continue  # Ignore header if present
    return labels


def parse_simplex_events(
    nverts: List[int],
    simplices: List[int],
    times: List[int],
) -> List[SimplexEvent]:
    """
    Aligns nverts, simplices, and times into a list of SimplexEvent objects.

    Mathematical verification:
    - len(nverts) must equal len(times)
    - sum(nverts) must equal len(simplices)

    Raises:
        DataAlignmentError: If lengths do not match or slicing fails.
    """
    if len(nverts) != len(times):
        raise DataAlignmentError(
            f"Vector length mismatch: nverts has {len(nverts)} rows, but times has {len(times)} rows."
        )

    expected_simplices_len = sum(nverts)
    if expected_simplices_len != len(simplices):
        raise DataAlignmentError(
            f"Simplices count mismatch: sum(nverts) = {expected_simplices_len}, "
            f"but simplices.txt contains {len(simplices)} entries."
        )

    events: List[SimplexEvent] = []
    offset = 0

    for event_id, (sz, timestamp) in enumerate(zip(nverts, times)):
        if sz < 1:
            raise DataAlignmentError(f"Event {event_id} has invalid participant count sz={sz} (< 1).")

        participants = tuple(simplices[offset : offset + sz])
        offset += sz

        events.append(
            SimplexEvent(
                event_id=event_id,
                timestamp_raw=timestamp,
                participants=participants,
            )
        )

    return events


def validate_dataset(
    nverts: List[int],
    simplices: List[int],
    times: List[int],
    node_labels: Optional[Dict[int, str]] = None,
    strict: bool = False,
) -> ValidationReport:
    """
    Performs comprehensive validation on the dataset components.

    Checks:
    1. Vector lengths match: len(nverts) == len(times).
    2. Sum of event sizes matches total participant entries: sum(nverts) == len(simplices).
    3. All node IDs are valid positive integers.
    4. Chronological order / monotonic timestamps.
    5. Event size distribution (e.g. single-participant self-loops vs pairwise vs higher-order).
    6. Identifies duplicate events (same timestamp and participant set).
    7. Identifies internal duplicate node IDs within any simplex.
    8. Identifies nodes present in node-labels.txt but with 0 communication events.

    Args:
        nverts: List of event participant counts.
        simplices: Concatenated participant node IDs.
        times: List of timestamps.
        node_labels: Optional mapping of node ID to email address.
        strict: If True, raises DatasetValidationError on any error.

    Returns:
        ValidationReport dataclass with all metrics, warnings, and errors.
    """
    errors: List[str] = []
    warnings: List[str] = []

    num_events = len(nverts)
    num_timestamps = len(times)
    counts_match = (num_events == num_timestamps)
    if not counts_match:
        errors.append(
            f"Row count mismatch: nverts has {num_events} rows, times has {num_timestamps} rows."
        )

    sum_nverts = sum(nverts)
    num_simplices_nodes = len(simplices)
    sums_match = (sum_nverts == num_simplices_nodes)
    if not sums_match:
        errors.append(
            f"Simplices count mismatch: sum(nverts) = {sum_nverts}, len(simplices) = {num_simplices_nodes}."
        )

    # Check for empty vectors
    if num_events == 0 or num_simplices_nodes == 0:
        errors.append("Dataset contains 0 events or 0 simplex nodes.")
        return ValidationReport(
            num_events=num_events,
            num_timestamps=num_timestamps,
            sum_nverts=sum_nverts,
            num_simplices_nodes=num_simplices_nodes,
            counts_match=counts_match,
            sums_match=sums_match,
            num_unique_nodes=0,
            min_node_id=0,
            max_node_id=0,
            num_node_labels=len(node_labels) if node_labels else 0,
            missing_nodes_in_events=[],
            min_timestamp=0,
            max_timestamp=0,
            time_span_days=0.0,
            start_date="",
            end_date="",
            is_chronological=False,
            timestamp_inversions=0,
            size_distribution={},
            unique_participant_sets=0,
            duplicate_events=0,
            internal_duplicate_node_events=0,
            warnings=warnings,
            errors=errors,
        )

    # Node ID integrity
    min_node_id = min(simplices)
    max_node_id = max(simplices)
    unique_nodes = set(simplices)
    if min_node_id <= 0:
        errors.append(f"Invalid non-positive node ID detected: min={min_node_id}. Expected 1-indexed IDs.")

    # Inactive nodes comparison with node-labels
    missing_nodes_in_events: List[int] = []
    num_node_labels = len(node_labels) if node_labels else 0
    if node_labels:
        all_label_nodes = set(node_labels.keys())
        missing_nodes = all_label_nodes - unique_nodes
        missing_nodes_in_events = sorted(missing_nodes)
        if missing_nodes_in_events:
            warnings.append(
                f"{len(missing_nodes_in_events)} nodes in node-labels never appear in any communication event: "
                f"{missing_nodes_in_events}"
            )
        extra_nodes = unique_nodes - all_label_nodes
        if extra_nodes:
            errors.append(f"{len(extra_nodes)} node IDs in simplices do not exist in node-labels: {sorted(extra_nodes)}")

    # Timestamp checks
    min_timestamp = min(times)
    max_timestamp = max(times)
    time_span_ms = max_timestamp - min_timestamp
    time_span_days = time_span_ms / (1000.0 * 86400.0)

    # Date formatting from Gregorian epoch
    start_dt = CORNELL_EPOCH_DATETIME + datetime.timedelta(seconds=min_timestamp / 1000.0)
    end_dt = CORNELL_EPOCH_DATETIME + datetime.timedelta(seconds=max_timestamp / 1000.0)
    start_date = start_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    end_date = end_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Timestamp monotonicity
    inversions = 0
    for i in range(len(times) - 1):
        if times[i] > times[i + 1]:
            inversions += 1
    is_chronological = (inversions == 0)
    if not is_chronological:
        warnings.append(
            f"Timestamps are not strictly non-decreasing: detected {inversions} inversions. "
            "Section III-A.1 of the paper specifies sorting events chronologically before temporal modeling."
        )

    # Simplex sizes & internal duplicates
    size_distribution: Dict[int, int] = dict(Counter(nverts))
    if any(sz <= 0 for sz in nverts):
        errors.append("Detected simplex event with size <= 0.")

    if 1 in size_distribution:
        warnings.append(
            f"Detected {size_distribution[1]} events of size 1 (single participant / self-loop). "
            "In communication graphs, size-1 events do not form pairwise edges unless self-loops are retained."
        )

    # Decode events to inspect duplicates
    offset = 0
    seen_events: Set[Tuple[int, Tuple[int, ...]]] = set()
    seen_participant_sets: Set[Tuple[int, ...]] = set()
    internal_duplicate_node_events = 0
    duplicate_events = 0

    for sz, t in zip(nverts, times):
        participants = tuple(simplices[offset : offset + sz])
        offset += sz

        # Check internal duplicate node IDs within single simplex
        if len(participants) != len(set(participants)):
            internal_duplicate_node_events += 1

        sorted_participants = tuple(sorted(participants))
        seen_participant_sets.add(sorted_participants)

        event_key = (t, sorted_participants)
        if event_key in seen_events:
            duplicate_events += 1
        else:
            seen_events.add(event_key)

    if internal_duplicate_node_events > 0:
        warnings.append(f"Detected {internal_duplicate_node_events} events with internal duplicate node IDs.")
    if duplicate_events > 0:
        warnings.append(
            f"Detected {duplicate_events} duplicate events with identical timestamp and participant sets."
        )

    report = ValidationReport(
        num_events=num_events,
        num_timestamps=num_timestamps,
        sum_nverts=sum_nverts,
        num_simplices_nodes=num_simplices_nodes,
        counts_match=counts_match,
        sums_match=sums_match,
        num_unique_nodes=len(unique_nodes),
        min_node_id=min_node_id,
        max_node_id=max_node_id,
        num_node_labels=num_node_labels,
        missing_nodes_in_events=missing_nodes_in_events,
        min_timestamp=min_timestamp,
        max_timestamp=max_timestamp,
        time_span_days=time_span_days,
        start_date=start_date,
        end_date=end_date,
        is_chronological=is_chronological,
        timestamp_inversions=inversions,
        size_distribution=size_distribution,
        unique_participant_sets=len(seen_participant_sets),
        duplicate_events=duplicate_events,
        internal_duplicate_node_events=internal_duplicate_node_events,
        warnings=warnings,
        errors=errors,
    )

    if strict and not report.is_valid:
        raise DatasetValidationError(f"Dataset validation failed with {len(errors)} error(s): {errors}")

    return report


def format_event_summary(
    event: SimplexEvent,
    labels: Optional[Dict[int, str]] = None,
    max_participants: int = 5,
) -> str:
    """
    Formats an event into a clear, educational string display.
    """
    dt_str = event.datetime.strftime("%Y-%m-%d %H:%M:%S")
    participants_str = []
    for pid in event.participants[:max_participants]:
        if labels and pid in labels:
            participants_str.append(f"Node {pid} ({labels[pid]})")
        else:
            participants_str.append(f"Node {pid}")
    if len(event.participants) > max_participants:
        participants_str.append(f"... +{len(event.participants) - max_participants} more")

    return (
        f"Event #{event.event_id:<5} | {dt_str} UTC (raw: {event.timestamp_raw}) | "
        f"Size {event.size}: [{', '.join(participants_str)}]"
    )
