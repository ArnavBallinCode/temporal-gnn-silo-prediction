"""
Graph operations module: maximal clique extraction, structural conflict scores, and meta-graph construction.

"""

from .cliques import (
    extract_maximal_cliques,
    filter_and_rank_cliques,
    format_clique_str,
    compute_clique_jaccard,
    CliqueMatchResult,
    match_cliques_bipartite,
    categorize_clique_disappearance,
)
from .scores import (
    compute_external_neighbors,
    compute_boundary_overlap,
    compute_iccs,
    compute_clique_tensions,
    compute_silo_risk_index,
    compute_temporal_smoothed_sri,
    compute_boundary_bridging_index,
    audit_sri_coverage,
)
from .meta_graph import (
    CliqueMetaGraph,
    build_clique_meta_graph,
)

__all__ = [
    "extract_maximal_cliques",
    "filter_and_rank_cliques",
    "format_clique_str",
    "compute_clique_jaccard",
    "CliqueMatchResult",
    "match_cliques_bipartite",
    "categorize_clique_disappearance",
    "compute_external_neighbors",
    "compute_boundary_overlap",
    "compute_iccs",
    "compute_clique_tensions",
    "compute_silo_risk_index",
    "compute_temporal_smoothed_sri",
    "compute_boundary_bridging_index",
    "audit_sri_coverage",
    "CliqueMetaGraph",
    "build_clique_meta_graph",
]
