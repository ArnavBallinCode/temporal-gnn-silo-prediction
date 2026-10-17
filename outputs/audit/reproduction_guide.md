# Scientific Audit: Complete Reproduction Instructions

This guide provides exact step-by-step instructions to reproduce the complete scientific audit, multi-seed benchmarks, ranking evaluations, network fragility simulations, and publication figures for the Temporal GNN Organizational Silo Prediction research.

---

## 1. Environment & Dependencies

- **OS:** macOS (Apple Silicon arm64) or Linux x86_64
- **Python:** $\ge 3.10$ (Tested on Python 3.14.7 and PyTorch 2.7.0)
- **Dependencies:**
  ```bash
  pip install torch networkx numpy scipy matplotlib pandas pytest
  ```

---

## 2. Directory Structure Verification

Ensure the dataset is located at `Dataset/email-Enron/`:
```
Dataset/email-Enron/
├── email-Enron-nverts.txt
├── email-Enron-simplices.txt
├── email-Enron-times.txt
└── node-labels.txt
```

---

## 3. Step-by-Step Reproduction Commands

### Step 1: Run the Automated Invariant and Regression Test Suite
```bash
pytest -v
```
*Expected outcome:* 49 passed tests across unit tests, two-stage target tests, temporal state tests, SRI coverage tests, and ranking evaluation tests.

---

### Step 2: Verify Raw Dataset & Build Graph Snapshots
```bash
python3 src/data/build_snapshots.py --data_dir Dataset/email-Enron --window_days 30 --stride_days 30
```
*Outputs generated:*
- `outputs/audit/data_validation.json`
- `outputs/audit/snapshot_statistics_verified.csv`

---

### Step 3: Run Multi-Seed Benchmarks & Model Ablations (5 Seeds)
```bash
python3 src/training/benchmark_audit.py
```
*Parameters:*
- Seeds: `[42, 123, 456, 789, 999]`
- Epochs: 35
- Learning rate: 0.01
- 11 Models evaluated:
  - Baselines: Persistence Baseline, Historical Mean Baseline
  - Single-Stage (Zero-Fill): Feature-Only MLP, Static GCN, Temporal GNN (No GRU Ablation), Temporal GNN (Cold-Start GRU), Temporal GNN (Warm-Start GRU)
  - Two-Stage Hurdle: Feature-Only MLP (Two-Stage), Static GCN (Two-Stage), Temporal GNN No-GRU (Two-Stage), Temporal GNN Full Warm-Start (Two-Stage)
- Evaluates both **Protocol A** (Unconditional, $N = 109$) and **Protocol B** (Conditional, $N = 35$ under 1-to-1 bipartite matching).

*Outputs generated:*
- `outputs/audit/benchmark_results.json`
- `outputs/audit/benchmark_results.csv`

---

### Step 4: Audit Node SRI Ranking Performance
```bash
python3 src/training/audit_ranking.py
```
*Evaluates:*
- Snapshot-by-snapshot Spearman rank correlation $\rho$, Top-5 overlap, and Top-10 overlap across test snapshots 36–42.
- Compares Temporal GNN Predicted SRI against Persistence Observed SRI, Degree Centrality, and Betweenness Centrality.
- Reports metrics on all 148 nodes vs. active clique members only.

*Outputs generated:*
- `outputs/audit/ranking_evaluation.csv`

---

### Step 5: Audit Network Fragility Simulation
```bash
python3 src/training/audit_fragility.py
```
*Simulates:*
- In-sample targeted node removal on $G^{(28)}$ (105 nodes) and out-of-sample forward disruption on $G^{(29)}$ (107 nodes).
- 50 independent random removal seeds with 95% confidence intervals.
- Strategies: Betweenness Centrality, Degree Centrality, Observed SRI, Predicted SRI, Random.

*Outputs generated:*
- `outputs/audit/fragility_results.csv`

---

### Step 6: Generate All 7 Audited Publication Figures
```bash
python3 src/training/generate_audit_figures.py
```
*Outputs generated:*
- `outputs/audit/figures/fig1_network_dynamics.png`
- `outputs/audit/figures/fig2_loss_curves.png`
- `outputs/audit/figures/fig3_model_benchmarks.png`
- `outputs/audit/figures/fig4_predicted_vs_true.png`
- `outputs/audit/figures/fig5_silo_risk_trajectories.png`
- `outputs/audit/figures/fig6_peak_conflict_metagraph.png`
- `outputs/audit/figures/fig7_network_fragility.png`
- `outputs/audit/figure_captions.md`

---

## 4. Key Artifacts Summary

| Artifact File | Description |
| :--- | :--- |
| `outputs/audit/implementation_audit.md` | Source-level implementation trace, tensor shapes, and line-by-line review. |
| `outputs/audit/data_validation.json` | Verification of event counts, simplex vector alignment, timestamps, and active nodes. |
| `outputs/audit/snapshot_statistics_verified.csv` | Graph metrics (nodes, edges, density, components, cliques) across all 44 snapshots. |
| `outputs/audit/methodology_audit.md` | Mathematical formulations, target definition analysis, Jaccard matching sensitivity, Protocol A vs. B dilemma. |
| `outputs/audit/benchmark_results.json` | 5-seed benchmark means and standard deviations for all 6 models under Protocol A and Protocol B. |
| `outputs/audit/benchmark_results.csv` | Tabular summary of multi-seed benchmark metrics. |
| `outputs/audit/ranking_evaluation.csv` | Snapshot-by-snapshot node SRI ranking evaluation comparing GNN to centrality baselines. |
| `outputs/audit/fragility_results.csv` | 50-seed network disruption simulation data for in-sample and forward graphs. |
| `outputs/audit/results_analysis.md` | Comprehensive scientific synthesis, claim categorization, and submission action items. |
| `outputs/audit/figure_captions.md` | Publication-ready, audited captions for Figures 1 through 7. |
