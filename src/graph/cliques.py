"""
Maximal Clique Extraction for Organizational Silo Detection.

 
Conceptual Background for Beginners:
-------------------------------------
1. What is a Clique?
   In graph theory, a clique is a subset of nodes in an undirected graph where
   EVERY pair of nodes is directly connected by an edge (a complete subgraph).
   In organizational terms, a clique represents a cohesive group of employees
   where everyone talks to everyone else.

2. Maximal Clique vs. Maximum Clique:
   - A MAXIMAL clique is a clique that CANNOT be extended by adding any other node
     from the graph. It is locally complete. A graph often has many maximal cliques
     of different sizes.
   - A MAXIMUM clique is the single largest clique (or cliques) in the entire graph
     (the global maximum size, also known as the clique number omega(G)).
   *In this research, we extract ALL MAXIMAL CLIQUES* because an organization has
   multiple distinct teams and project pods of various sizes, not just one giant team.

3. Algorithm:
   We use the classic Bron–Kerbosch algorithm (implemented via NetworkX's `find_cliques`),
   which efficiently enumerates all maximal cliques using recursive backtracking with
   pivot vertex optimization.
   Following the paper (Section V-A), we retain the top K' largest cliques (e.g. K' = 30)
   with minimum size >= 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx


def extract_maximal_cliques(
    G: nx.Graph,
    min_size: int = 3,
    top_k: Optional[int] = 30,
) -> List[Tuple[int, ...]]:
    """
    Extracts maximal cliques from an undirected graph G.

    Steps:
    1. Runs the Bron-Kerbosch algorithm via `nx.find_cliques(G)`.
    2. Filters out cliques with fewer than `min_size` members (e.g. min_size=3).
    3. Sorts cliques deterministically:
       - Primary sort: size descending (largest teams first).
       - Secondary sort: sorted node IDs ascending (for 100% reproducible ordering).
    4. Retains top_k largest cliques if top_k is specified (e.g. top_k=30, as in the paper).

    Args:
        G: NetworkX undirected communication graph for a snapshot.
        min_size: Minimum number of members for a clique to be considered (default: 3).
        top_k: Optional maximum number of cliques to return (default: 30).

    Returns:
        List of cliques, where each clique is represented as a sorted tuple of node IDs.
    """
    if G.number_of_nodes() < min_size:
        return []

    # 1. Enumerate all maximal cliques
    raw_cliques = list(nx.find_cliques(G))

    # 2. Filter by minimum size
    filtered_cliques: List[Tuple[int, ...]] = []
    for c in raw_cliques:
        if len(c) >= min_size:
            # Sort node IDs within the clique for consistent representation
            filtered_cliques.append(tuple(sorted(c)))

    # 3. Sort deterministically:
    # - Negative length for descending size
    # - Tuple of node IDs for deterministic tie-breaking
    sorted_cliques = sorted(filtered_cliques, key=lambda c: (-len(c), c))

    # 4. Retain top K' cliques (as specified in paper Section III-B and V-A)
    if top_k is not None and len(sorted_cliques) > top_k:
        return sorted_cliques[:top_k]

    return sorted_cliques


def filter_and_rank_cliques(
    cliques: List[Tuple[int, ...]],
    top_k: Optional[int] = 30,
    min_size: int = 3,
) -> List[Tuple[int, ...]]:
    """
    Utility function to filter and rank a pre-existing list of cliques.
    """
    filtered = [tuple(sorted(c)) for c in cliques if len(c) >= min_size]
    sorted_cliques = sorted(filtered, key=lambda c: (-len(c), c))
    if top_k is not None and len(sorted_cliques) > top_k:
        return sorted_cliques[:top_k]
    return sorted_cliques


def format_clique_str(clique: Tuple[int, ...], labels: Optional[Dict[int, str]] = None) -> str:
    """
    Formats a clique tuple into a readable string with member emails if available.
    """
    if not labels:
        return f"Clique(size={len(clique)}, nodes={list(clique)})"

    members = [f"{nid} ({labels.get(nid, 'unknown').split('@')[0]})" for nid in clique]
    return f"Clique(size={len(clique)}): [{', '.join(members)}]"


def compute_clique_jaccard(
    clique_a: Tuple[int, ...] | Set[int],
    clique_b: Tuple[int, ...] | Set[int],
) -> float:
    """
    Computes Jaccard node similarity between two cliques:
        J(A, B) = |A intersect B| / |A union B|
    """
    set_a = set(clique_a)
    set_b = set(clique_b)
    union_len = len(set_a.union(set_b))
    if union_len == 0:
        return 0.0
    return len(set_a.intersection(set_b)) / float(union_len)


@dataclass
class CliqueMatchResult:
    """
    Detailed results of matching cliques across consecutive snapshots.

    Attributes:
        prev_to_curr: 1-to-1 mapping from previous clique index to current clique index.
        curr_to_prev: 1-to-1 mapping from current clique index to previous clique index.
        matched_similarities: Mapping of (prev_idx, curr_idx) -> Jaccard similarity score.
        unmatched_prev: List of previous clique indices with no valid match.
        unmatched_curr: List of current clique indices with no valid match (new groups).
        splits: Mapping of prev_idx -> list of (curr_idx, jaccard) for non-1-to-1 partial overlaps.
        merges: Mapping of curr_idx -> list of (prev_idx, jaccard) for non-1-to-1 partial overlaps.
    """
    prev_to_curr: Dict[int, int]
    curr_to_prev: Dict[int, int]
    matched_similarities: Dict[Tuple[int, int], float]
    unmatched_prev: List[int]
    unmatched_curr: List[int]
    splits: Dict[int, List[Tuple[int, float]]]
    merges: Dict[int, List[Tuple[int, float]]]


def match_cliques_bipartite(
    prev_cliques: List[Tuple[int, ...]],
    curr_cliques: List[Tuple[int, ...]],
    threshold: float = 0.5,
    method: str = "greedy",
    overlap_threshold: float = 0.3,
) -> CliqueMatchResult:
    """
    Matches cliques across consecutive snapshots using 1-to-1 bipartite matching.

    Guarantees:
    - At most ONE current clique is matched to any previous clique.
    - At most ONE previous clique is matched to any current clique.
    - Detects splits (one previous clique overlapping multiple current cliques).
    - Detects merges (multiple previous cliques overlapping one current clique).

    Args:
        prev_cliques: Cliques from snapshot k.
        curr_cliques: Cliques from snapshot k + 1.
        threshold: Minimum Jaccard similarity required for a valid 1-to-1 match.
        method: "greedy" (highest-similarity first) or "hungarian" (maximum weight matching).
        overlap_threshold: Threshold to detect partial splits/merges.

    Returns:
        CliqueMatchResult dataclass.
    """
    num_prev = len(prev_cliques)
    num_curr = len(curr_cliques)

    if num_prev == 0 or num_curr == 0:
        return CliqueMatchResult(
            prev_to_curr={},
            curr_to_prev={},
            matched_similarities={},
            unmatched_prev=list(range(num_prev)),
            unmatched_curr=list(range(num_curr)),
            splits={},
            merges={},
        )

    prev_sets = [set(c) for c in prev_cliques]
    curr_sets = [set(c) for c in curr_cliques]

    # Compute similarity matrix and identify all candidate edges
    candidates: List[Tuple[float, int, int]] = []
    sim_matrix: Dict[Tuple[int, int], float] = {}
    splits_candidates: Dict[int, List[Tuple[int, float]]] = {p: [] for p in range(num_prev)}
    merges_candidates: Dict[int, List[Tuple[int, float]]] = {c: [] for c in range(num_curr)}

    for p_idx, p_set in enumerate(prev_sets):
        for c_idx, c_set in enumerate(curr_sets):
            jaccard = compute_clique_jaccard(p_set, c_set)
            sim_matrix[(p_idx, c_idx)] = jaccard

            if jaccard >= threshold:
                candidates.append((jaccard, p_idx, c_idx))

            if jaccard >= overlap_threshold:
                splits_candidates[p_idx].append((c_idx, jaccard))
                merges_candidates[c_idx].append((p_idx, jaccard))

    prev_to_curr: Dict[int, int] = {}
    curr_to_prev: Dict[int, int] = {}
    matched_sims: Dict[Tuple[int, int], float] = {}

    if method == "hungarian":
        try:
            from scipy.optimize import linear_sum_assignment
            import numpy as np

            cost_matrix = np.zeros((num_prev, num_curr))
            for p_idx in range(num_prev):
                for c_idx in range(num_curr):
                    cost_matrix[p_idx, c_idx] = -sim_matrix[(p_idx, c_idx)]

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                sim = -cost_matrix[r, c]
                if sim >= threshold:
                    prev_to_curr[int(r)] = int(c)
                    curr_to_prev[int(c)] = int(r)
                    matched_sims[(int(r), int(c))] = round(float(sim), 4)
        except ImportError:
            method = "greedy"

    if method == "greedy":
        # Sort candidates descending by Jaccard similarity.
        # Tie-break deterministically by prev_idx ascending, then curr_idx ascending.
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

        used_prev: Set[int] = set()
        used_curr: Set[int] = set()

        for jaccard, p_idx, c_idx in candidates:
            if p_idx not in used_prev and c_idx not in used_curr:
                used_prev.add(p_idx)
                used_curr.add(c_idx)
                prev_to_curr[p_idx] = c_idx
                curr_to_prev[c_idx] = p_idx
                matched_sims[(p_idx, c_idx)] = round(jaccard, 4)

    unmatched_prev = [p for p in range(num_prev) if p not in prev_to_curr]
    unmatched_curr = [c for c in range(num_curr) if c not in curr_to_prev]

    # Filter splits to only those with >= 2 overlaps or overlapping non-matched
    splits = {p: clist for p, clist in splits_candidates.items() if len(clist) > 1}
    merges = {c: plist for c, plist in merges_candidates.items() if len(plist) > 1}

    return CliqueMatchResult(
        prev_to_curr=prev_to_curr,
        curr_to_prev=curr_to_prev,
        matched_similarities=matched_sims,
        unmatched_prev=unmatched_prev,
        unmatched_curr=unmatched_curr,
        splits=splits,
        merges=merges,
    )


def categorize_clique_disappearance(
    prev_clique: Tuple[int, ...],
    next_retained_cliques: List[Tuple[int, ...]],
    next_all_cliques: Optional[List[Tuple[int, ...]]] = None,
    threshold: float = 0.5,
) -> Dict[str, any]:
    """
    Categorizes why a clique in snapshot k did not match a retained clique in snapshot k+1.

    Distinguishes:
    1. 'omitted_by_truncation': The clique persisted into snapshot k+1, but the matching
       clique was ranked outside the top_k retained cliques.
    2. 'split_or_reorganized': The clique broke into smaller subsets or reorganized (overlap >= 0.3).
    3. 'dissolved_structural': Genuine structural dissolution (no clique in k+1 with J >= 0.3).

    Args:
        prev_clique: Clique from snapshot k.
        next_retained_cliques: Retained cliques (e.g. top_k=30) in snapshot k+1.
        next_all_cliques: All maximal cliques (size >= 3) in snapshot k+1 before top_k truncation.
        threshold: Jaccard persistence threshold (default: 0.5).

    Returns:
        Dict with 'category', 'best_retained_jaccard', 'best_full_jaccard', 'matching_discarded_clique'.
    """
    prev_set = set(prev_clique)

    # Check best match in retained cliques
    best_retained_j = 0.0
    for c in next_retained_cliques:
        j = compute_clique_jaccard(prev_set, c)
        if j > best_retained_j:
            best_retained_j = j

    # Check if match exists in full list of cliques (beyond top_k)
    best_full_j = best_retained_j
    discarded_match = None

    if next_all_cliques is not None and len(next_all_cliques) > len(next_retained_cliques):
        retained_set = set(tuple(sorted(c)) for c in next_retained_cliques)
        for c in next_all_cliques:
            c_tup = tuple(sorted(c))
            if c_tup not in retained_set:
                j = compute_clique_jaccard(prev_set, c)
                if j > best_full_j:
                    best_full_j = j
                    if j >= threshold:
                        discarded_match = c_tup

    if discarded_match is not None:
        category = "omitted_by_truncation"
    elif best_retained_j >= threshold:
        category = "matched_retained"
    elif best_full_j >= 0.3 or best_retained_j >= 0.3:
        category = "split_or_reorganized"
    else:
        category = "dissolved_structural"

    return {
        "category": category,
        "best_retained_jaccard": round(best_retained_j, 4),
        "best_full_jaccard": round(best_full_j, 4),
        "matching_discarded_clique": discarded_match,
    }

