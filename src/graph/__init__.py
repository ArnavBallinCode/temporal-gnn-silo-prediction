"""
Graph operations module: maximal clique extraction, structural conflict scores, and meta-graph construction.

"""

from .cliques import (
    extract_maximal_cliques,
    filter_and_rank_cliques,
    format_clique_str,
)
from .scores import (
    compute_external_neighbors,
    compute_boundary_overlap,
    compute_iccs,
    compute_clique_tensions,
    compute_silo_risk_index,
    compute_temporal_smoothed_sri,
)
from .meta_graph import (
    CliqueMetaGraph,
    build_clique_meta_graph,
)

__all__ = [
    "extract_maximal_cliques",
    "filter_and_rank_cliques",
    "format_clique_str",
    "compute_external_neighbors",
    "compute_boundary_overlap",
    "compute_iccs",
    "compute_clique_tensions",
    "compute_silo_risk_index",
    "compute_temporal_smoothed_sri",
    "CliqueMetaGraph",
    "build_clique_meta_graph",
]
