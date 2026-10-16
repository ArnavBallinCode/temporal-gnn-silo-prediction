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
