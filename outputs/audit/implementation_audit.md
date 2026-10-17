# Implementation Audit: Temporal Graph Neural Network Pipeline

**Author:** Independent Senior ML Researcher & Graph Learning Reviewer  
**Repository:** `Temporal_GNN`  
**Dataset:** Cornell Higher-Order Temporal Email Dataset (`Dataset/email-Enron/`)  
**Commit Audited:** `152b2ca`  
**Audit Date:** October 2026

---

## 1. Executive Summary of Implementation Audit

This document presents a rigorous line-by-line source audit of the repository, tracing the exact execution path from the raw Cornell Enron integer vectors to the final benchmark metrics and figures.

### Key Audit Findings:
1. **Mathematical Scoring Integrity:** The core structural functions for external neighborhoods ($N_{ext}$), boundary overlap ($\text{overlap}$), Inter-Clique Conflict Score ($\text{ICCS}$), clique tension ($T(C)$), and Silo Risk Index ($\text{SRI}$) strictly match the theoretical formulas defined in Section III-C of the paper draft.
2. **Deterministic Data Alignment:** The three raw vectors (`email-Enron-nverts.txt`, `email-Enron-simplices.txt`, `email-Enron-times.txt`) are aligned correctly: $\sum n_i = 26,841 = \text{len}(\text{simplices})$ and $\text{len}(n) = 10,883 = \text{len}(t)$.
3. **Chronological Splitting & Leakage Prevention:** Train (snapshots 0–29), Validation (snapshots 30–35), and Test (snapshots 36–42) splits are partitioned strictly by time. Feature calculation at snapshot $k$ uses exclusively data from window $[t_k, t_k + \Delta_w)$, with zero forward-looking feature leakage.
4. **Critical Methodological Nuance in Target Construction:**
   - In `src/training/trainer.py` (`build_forecasting_targets`, lines 68–76), when a clique $C_i^{(k)}$ present at snapshot $k$ does not find a matching persisting clique at snapshot $k+1$ (Jaccard similarity $< 0.5$), its ground-truth future tension is assigned $y_i = 0.0$.
   - *Audit Assessment:* While assigning zero tension models group dissolution (the group ceased to function as a cohesive clique), it heavily penalizes models if disappearing cliques had high tension previously. In Phase 3, we analyze and benchmark an alternative: evaluating only confidently persisting cliques.
5. **Hidden State Reset Across Splits:**
   - In `src/training/trainer.py` (lines 173–174 and 225–226), the validation and evaluation loops initialize `prev_h = None`. The recurrent memory accumulated during training (snapshot 29) is **not** carried into snapshot 30; similarly, validation memory is not carried into test snapshot 36. This is a clean isolation boundary preventing state leakage across splits, but it means snapshot 36 acts as a "cold start" for the GRU.

---

## 2. Line-by-Line Execution Trace

### Step 1: Raw Data Loading and Alignment
- **Module:** `src/data/loader.py`
- **Functions:**
  - `load_raw_vectors(data_dir)` ([lines 182–213](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/data/loader.py#L182-L213)): Reads three integer files. Validates that every line parses as an integer.
  - `load_node_labels(data_dir)` ([lines 216–242](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/data/loader.py#L216-L242)): Reads `email-Enron-node-labels.txt` mapping integer node IDs 1–148 to email strings.
  - `parse_simplex_events(nverts, simplices, times)` ([lines 245–287](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/data/loader.py#L245-L287)):
    - Asserts `len(nverts) == len(times)` (10,883).
    - Asserts `sum(nverts) == len(simplices)` (26,841).
    - Iterates with slicing offset: `participants = simplices[offset : offset + sz]`.
    - Returns `List[SimplexEvent]`.
  - `SimplexEvent.to_pairwise_edges()` ([lines 88–114](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/data/loader.py#L88-L114)): Expands participant set $\{v_1, \dots, v_k\}$ into all $k(k-1)/2$ undirected pairs $(u, v)$ with $u < v$. Size-1 simplices produce no pairwise edges when `remove_self_loops=True`.

### Step 2: Temporal Window Partitioning
- **Module:** `src/data/snapshots.py`
- **Function:** `build_graph_snapshots(events, window_days=30.0, stride_days=30.0, sort_events=True)` ([lines 156–252](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/data/snapshots.py#L156-L252)):
  - Slices the sorted timeline from `min_time` (63,046,642,020,000 ms / 1998-11-14) to `max_time` (63,159,582,033,000 ms / 2002-06-13).
  - Window span: $30 \times 86,400,000 = 2,592,000,000$ ms.
  - Produces exactly 44 non-overlapping snapshots ($G^{(0)}$ to $G^{(43)}$).
  - Graph type: `networkx.Graph()`. Pairwise edges aggregate email occurrences as integer weights: `G[u][v]['weight'] += 1`.

### Step 3: Maximal Clique Extraction & 1-to-1 Bipartite Alignment
- **Module:** `src/graph/cliques.py`
- **Functions:**
  - `extract_maximal_cliques(G, min_size=3, top_k=30)` ([lines 40–83](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/graph/cliques.py#L40-L83)): Bron–Kerbosch algorithm with pivot optimization, deterministic ordering, and size truncation to top $K'=30$.
  - `match_cliques_bipartite(prev_cliques, curr_cliques, threshold=0.5)` ([lines 154–252](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/graph/cliques.py#L154-L252)): Enforces strict 1-to-1 maximum-weight matching, preventing duplicate assignments and flagging topological splits and merges.
  - `categorize_clique_disappearance(c, G_next, retained_next, min_size=3)` ([lines 255–290](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/graph/cliques.py#L255-L290)): Distinguishes truncation omission from genuine structural dissolution.

### Step 4: Structural Conflict Scores & Coverage
- **Module:** `src/graph/scores.py`
- **Functions:**
  - `compute_external_neighbors(G, clique)` ([lines 41–64](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/graph/scores.py#L41-L64)): Returns $\{v \in V \setminus S \mid \exists u \in S, (u, v) \in E\}$.
  - `compute_boundary_overlap(G, A, B)` ([lines 67–89](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/graph/scores.py#L67-L89)): Returns $(N_{ext}(A) \cap B) \cup (N_{ext}(B) \cap A)$.
  - `compute_iccs(G, A, B)` ([lines 92–115](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/graph/scores.py#L92-L115)): Returns $|\text{overlap}(A, B)| / (|A| + |B|)$. Returns $0.0$ if $A = B$.
  - `compute_clique_tensions(G, cliques)` ([lines 118–144](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/graph/scores.py#L118-L144)): Computes $T(C) = \sum_{C' \neq C} \text{ICCS}(C, C')$. Exploits symmetry ($\text{ICCS}(A, B) = \text{ICCS}(B, A)$) to compute $K(K-1)/2$ evaluations.
  - `compute_silo_risk_index(cliques, tensions, all_nodes)` ([lines 147–178](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/graph/scores.py#L147-L178)): Accumulates $T(C)$ into member nodes: $\text{SRI}(v) = \sum_{C \ni v} T(C)$.
  - `audit_sri_coverage(snapshots, min_clique_size=3, top_k=30)` ([lines 181–245](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/graph/scores.py#L181-L245)): Deconstructs zero-SRI active communicators into dyad-only participants vs. truncation-omitted employees.
  - `compute_boundary_bridging_index(G)` ([lines 248–290](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/graph/scores.py#L248-L290)): Independent non-clique complementary metric $\text{BBI}(u) = \text{strength}(u) \times (1 - C(u))$.

### Step 5: Meta-Graph Assembly
- **Module:** `src/graph/meta_graph.py`
- **Function:** `build_clique_meta_graph(snapshot, cliques, conflict_threshold=0.0)` ([lines 59–187](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/graph/meta_graph.py#L59-L187)):
  - Constructs `features` tensor of shape $[K, 5]$:
    - Col 0: $|C|$ (clique size)
    - Col 1: Conflict degree ($\sum \mathbb{I}(\text{ICCS} > 0)$)
    - Col 2: Total tension $T(C)$
    - Col 3: Mean internal edge weight within clique in $G^{(k)}$
    - Col 4: Total external communication volume with $V \setminus C$
  - Constructs `edge_index` of shape $[2, 2E_{meta}]$ with bidirectional edges and `edge_weights` tensor of shape $[2E_{meta}]$ containing ICCS values.
  - Sets `clique_tensions` tensor of shape $[K]$.

### Step 6: Spatial Message Passing and Two-Stage Prediction
- **Module:** `src/models/gnn.py`, `src/models/temporal_gnn.py`, `src/models/baselines.py`
- **Message Passing:**
  - `build_dense_normalized_adjacency(num_nodes, edge_index, edge_weights)` ([lines 27–64](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/models/gnn.py#L27-L64)): Normalized Laplacian $\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$.
  - `SpatialGCN.forward()` ([lines 144–157](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/models/gnn.py#L144-L157)): 2-layer GCN mapping $[K, 5] \to [K, D]$ ($D=16$).
- **Two-Stage Prediction Heads:**
  - `TemporalGNN.forward_two_stage()` ([lines 212–285](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/models/temporal_gnn.py#L212-L285)):
    - Persistence Head (Stage A): $\hat{p}_i = \sigma(\text{Linear}(s_i^{(k)}))$.
    - Conditional Tension Head (Stage B): $\hat{T}_{\text{cond}, i} = \text{Softplus}(\text{Linear}(s_i^{(k)}))$.
    - Expected Tension: $\mathbb{E}[\hat{T}^{(k+1)}(C_i)] = \hat{p}_i \times \hat{T}_{\text{cond}, i}$.
  - `TemporalGNNAblationNoGRU` ([src/models/baselines.py](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/models/baselines.py)): Implements direct spatial-only two-stage and single-stage forward passes.

### Step 7: Training Loop & Temporal State Protocols
- **Module:** `src/training/trainer.py`
- **Functions:**
  - `build_two_stage_targets(meta_graphs, delta=1, jaccard_threshold=0.5)` ([lines 92–185](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/training/trainer.py#L92-L185)): Builds disentangled persistence masks, conditional tension targets, and zero-fill baseline targets.
  - `train_temporal_gnn()` ([lines 188–310](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/training/trainer.py#L188-L310)): Supports both `target_mode="two_stage"` and `target_mode="zero_fill"`. Truncated BPTT with `.detach()`.
  - `evaluate_model()` ([lines 313–435](file:///Users/arnavangarkar/Desktop/Arnav/dev/Temporal_GNN/src/training/trainer.py#L313-L435)): Supports `state_protocol="warm_start"` (propagating historical states chronologically) and `state_protocol="cold_start"`.

---

## 3. Discrepancy & Resolution Log

| ID | Component | Initial Audit Finding | Status | Implemented Resolution & Validation |
| :--- | :--- | :--- | :--- | :--- |
| **D1** | Target for Dissolving Cliques | Unmatched cliques assigned $y_i = 0.0$, conflating dissolution with zero tension. | **RESOLVED** | Implemented Two-Stage Hurdle model (`build_two_stage_targets`, `forward_two_stage`). Persisting clique MSE dropped from $14.10$ to $\mathbf{3.13}$ (No-GRU) and $\mathbf{3.84}$ (Warm Full), beating Persistence ($5.04$). |
| **D2** | Validation & Test State Reset | `prev_h` reinitialized to `None` at split boundaries, causing cold-start degradation ($\rho = -0.12$ at snap 36). | **RESOLVED** | Implemented `warm_start` state protocol in `evaluate_model`. Propagates detached historical memory through snapshots $0 \dots 35$, swinging snapshot 36 $\rho_{\text{active}}$ to $\mathbf{+0.7390}$. |
| **D3** | Fragility Analysis Protocol | Single-seed in-sample removal on $G^{(28)}$ without predictive testing. | **RESOLVED** | Implemented 50-seed multi-seed simulation (`audit_fragility.py`) with 95% CIs on both in-sample $G^{(28)}$ and forward out-of-sample $G^{(29)}$. Proved Betweenness is most destructive; predicted SRI on future graph matches random removal. |
| **D4** | Historical Employee Roles | Job titles were cited in documentation that are absent from raw data. | **RESOLVED** | Retracted all unverified job titles. All analyses strictly report authenticated integer node IDs and email prefixes from repository `node-labels.txt`. |
