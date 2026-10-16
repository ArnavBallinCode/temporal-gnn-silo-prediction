"""
Structural Conflict Scores: ICCS, Clique Tension, and Silo Risk Index (SRI).
Mathematical Definitions & Formulations:
-----------------------------------------
1. External Neighbor Set:
   For a clique S in graph G:
       N_ext(S) = { v in V \\ S | exists u in S such that (u, v) in E }
   Plain English:
       All employees OUTSIDE the clique who communicate with at least ONE person inside the clique.

2. Boundary Overlap:
   For two cliques A and B:
       overlap(A, B) = (N_ext(A) intersect B) union (N_ext(B) intersect A)
   Plain English:
       The set of boundary employees in A and B who bridge communication across group boundaries.

3. Inter-Clique Conflict Score (ICCS):
       ICCS(A, B) = |overlap(A, B)| / (|A| + |B|)
   Plain English:
       The fraction of the combined group membership that is pulled across group boundaries.
       Normalized between 0.0 and 1.0.

4. Instantaneous Clique Tension T(C):
       T(C) = sum_{C' != C} ICCS(C, C')
   Plain English:
       The cumulative boundary tension experienced by clique C across all other cohesive teams.

5. Instantaneous Silo Risk Index (SRI) for employee v:
       SRI(v) = sum_{C contains v} T(C)
   Plain English:
       Aggregates clique tension across all cliques that employee v belongs to.
       High-SRI individuals occupy fragile, overburdened bridging positions between competing cliques.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import networkx as nx


def compute_external_neighbors(G: nx.Graph, clique: Tuple[int, ...] | Set[int]) -> Set[int]:
    """
    Computes N_ext(S): external neighbors of clique S outside S.

    Equation:
        N_ext(S) = { v in V \\ S | exists u in S with (u, v) in E }

    Args:
        G: NetworkX graph.
        clique: Tuple or set of node IDs representing the clique.

    Returns:
        Set of node IDs outside the clique that have at least one edge into the clique.
    """
    clique_nodes = set(clique)
    ext_neighbors: Set[int] = set()

    for u in clique_nodes:
        if G.has_node(u):
            for v in G.neighbors(u):
                if v not in clique_nodes:
                    ext_neighbors.add(v)

    return ext_neighbors


def compute_boundary_overlap(
    G: nx.Graph,
    clique_a: Tuple[int, ...],
    clique_b: Tuple[int, ...],
) -> Set[int]:
    """
    Computes overlap(A, B): boundary nodes pulled across the two cliques.

    Equation:
        overlap(A, B) = (N_ext(A) intersect B) union (N_ext(B) intersect A)
    """
    set_a = set(clique_a)
    set_b = set(clique_b)

    # If both cliques are identical, there is no boundary conflict between them
    if set_a == set_b:
        return set()

    n_ext_a = compute_external_neighbors(G, set_a)
    n_ext_b = compute_external_neighbors(G, set_b)

    overlap_nodes = (n_ext_a.intersection(set_b)).union(n_ext_b.intersection(set_a))
    return overlap_nodes


def compute_iccs(
    G: nx.Graph,
    clique_a: Tuple[int, ...],
    clique_b: Tuple[int, ...],
) -> float:
    """
    Computes the Inter-Clique Conflict Score ICCS(A, B).

    Equation:
        ICCS(A, B) = |overlap(A, B)| / (|A| + |B|)

    Edge cases handled:
    - Same clique: returns 0.0
    - Empty cliques or zero denominator: returns 0.0
    """
    if clique_a == clique_b:
        return 0.0

    denom = len(clique_a) + len(clique_b)
    if denom == 0:
        return 0.0

    overlap_nodes = compute_boundary_overlap(G, clique_a, clique_b)
    return len(overlap_nodes) / float(denom)


def compute_clique_tensions(
    G: nx.Graph,
    cliques: List[Tuple[int, ...]],
) -> Dict[Tuple[int, ...], float]:
    """
    Computes instantaneous clique tension T(C) for all cliques in a snapshot.

    Equation:
        T(C) = sum_{C' in Cliques, C' != C} ICCS(C, C')

    Returns:
        Dict mapping each clique tuple to its float tension score T(C).
    """
    tensions: Dict[Tuple[int, ...], float] = {c: 0.0 for c in cliques}
    num_cliques = len(cliques)

    # Compute pairwise ICCS (symmetric: ICCS(A, B) == ICCS(B, A))
    for i in range(num_cliques):
        c_a = cliques[i]
        for j in range(i + 1, num_cliques):
            c_b = cliques[j]
            score = compute_iccs(G, c_a, c_b)
            if score > 0.0:
                tensions[c_a] += score
                tensions[c_b] += score

    return tensions


def compute_silo_risk_index(
    cliques: List[Tuple[int, ...]],
    tensions: Dict[Tuple[int, ...], float],
    all_nodes: Optional[List[int]] = None,
) -> Dict[int, float]:
    """
    Computes the instantaneous Silo Risk Index (SRI) for each employee node.

    Equation:
        SRI(v) = sum_{C contains v} T(C)

    Args:
        cliques: List of maximal cliques.
        tensions: Pre-computed tension scores for each clique.
        all_nodes: Optional complete list of all known employee node IDs.
                   If provided, nodes that do not belong to any clique get SRI = 0.0.

    Returns:
        Dict mapping integer node ID -> float SRI score.
    """
    sri_scores: Dict[int, float] = {}

    # Initialize all requested nodes with 0.0
    if all_nodes is not None:
        for nid in all_nodes:
            sri_scores[nid] = 0.0

    # Aggregate clique tensions into node SRI
    for c in cliques:
        c_tension = tensions.get(c, 0.0)
        for nid in c:
            sri_scores[nid] = sri_scores.get(nid, 0.0) + c_tension

    return sri_scores


def compute_temporal_smoothed_sri(
    sri_history: List[Dict[int, float]],
    decay_rate: float = 0.8,
    window_length: int = 3,
) -> Dict[int, float]:
    """
    Computes windowed temporal SRI smoothing over the past L snapshots.

    Equation (Section III-C.2):
        SRI_temp^(k)(v) = (1 / Z) * sum_{l = k - L + 1}^k w_l * SRI^(l)(v)
        where w_l = decay_rate^(k - l) and Z = sum w_l.

    Args:
        sri_history: List of SRI dicts from recent snapshots up to current time k.
        decay_rate: Exponential decay factor alpha in [0, 1] (default: 0.8).
        window_length: Number of past snapshots L to include (default: 3).

    Returns:
        Dict mapping node ID -> smoothed float SRI score.
    """
    if not sri_history:
        return {}

    recent_history = sri_history[-window_length:]
    L = len(recent_history)

    # Compute weights w_l = decay_rate^(L - 1 - idx)
    # Most recent snapshot (idx = L - 1) gets weight = decay_rate^0 = 1.0
    weights = [decay_rate ** (L - 1 - idx) for idx in range(L)]
    Z = sum(weights)

    smoothed: Dict[int, float] = {}
    all_nodes: Set[int] = set()
    for s_dict in recent_history:
        all_nodes.update(s_dict.keys())

    for nid in all_nodes:
        weighted_sum = sum(
            weights[idx] * recent_history[idx].get(nid, 0.0)
            for idx in range(L)
        )
        smoothed[nid] = weighted_sum / Z if Z > 0 else 0.0

    return smoothed
