# Audited Publication Figure Captions and Methodological Descriptions

This document provides scientifically rigorous, fully audited captions and analytical descriptions for all seven figures generated for the research on Temporal Graph Neural Networks for Organizational Silo Prediction.

---

### Figure 1: Longitudinal Network Evolution and Subgroup Dynamics (`fig1_network_dynamics.png`)

**Caption:**
**Figure 1: Longitudinal communication network dynamics and cohesive subgroup evolution across 44 discrete 30-day temporal snapshots (November 1998 – June 2002).**
**(a)** Total active participating employees ($|V|$) and pairwise email communication edges ($|E|$) per 30-day window. Chronological reference markers denote Snapshot 28 (March 2001, exhibiting peak clique complexity) and Snapshot 37 (December 2001, coinciding with Enron's Chapter 11 bankruptcy filing).
**(b)** Network connectivity metrics: graph density $\rho = 2|E|/(|V|(|V|-1))$ (left axis) and number of connected components (right axis).
**(c)** Cohesive subgroup structure: count of extracted maximal cliques of size $\ge 3$ (top 30 by size) and maximum clique size ($\omega(G)$). Communication volume and clique density peaked during Spring 2001 before sharply contracting following late-2001 corporate restructuring.

---

### Figure 2: Temporal GNN Convergence Curves (`fig2_loss_curves.png`)

**Caption:**
**Figure 2: Empirical convergence of the Temporal GNN model.**
Training Mean Squared Error (MSE) computed across training snapshots 0–29 (70% chronological split, November 1998 – May 2001) and validation MSE computed across validation snapshots 30–35 (15% chronological split, May 2001 – October 2001) over 35 training epochs with Adam optimizer ($\text{lr} = 0.01$, weight decay $10^{-4}$). The validation loss reaches minimum at epoch 32, which was selected as the checkpoint for held-out evaluation on test snapshots 36–42 (October 2001 – April 2002).

---

### Figure 3: Benchmark Model Comparisons Across Independent Seeds (`fig3_model_benchmarks.png`)

**Caption:**
**Figure 3: Multi-seed forecasting error comparison across model architectures under Protocol A and Protocol B.**
Values report mean Mean Squared Error (MSE) and standard deviations across five independent random seeds (`[42, 123, 456, 789, 999]`).
**(Left - Protocol A: Unconditional Evaluation, $N = 109$ test cliques):** Evaluates all cliques present in test snapshots 36–42. Cliques that dissolve in the subsequent snapshot ($J < 0.5$, 67.9% of test cliques, $N = 74$) receive target tension $y = 0.0$. Persistence Baseline achieves high error ($\text{MSE} = 23.47 \pm 0.00$) because it blindly projects past tension into dissolved cliques. Neural baselines predict conservative shrinkage towards zero: Static GCN ($\text{MSE} = 5.33 \pm 0.26$), Temporal GNN (Warm-Start GRU) ($\text{MSE} = 5.47 \pm 0.24$), Feature-Only MLP ($\text{MSE} = 5.53 \pm 0.04$), and Two-Stage Temporal GNN ($\text{MSE} = 6.18 \pm 0.98$).
**(Right - Protocol B: Conditional on Survival, $N = 35$ persisting cliques under 1-to-1 bipartite matching):** Evaluates strictly the 35 cliques that survive into $t+1$ with Jaccard overlap $J \ge 0.5$. Compares single-stage zero-fill models (orange bars) against the newly developed two-stage hurdle models (teal bars). Under single-stage training, every neural architecture was heavily biased by zeros, losing decisively to Persistence ($\text{MSE} = 5.04 \pm 0.00$, $\text{Pearson } r = 0.7206$). Under two-stage hurdle modeling (training the tension regressor conditionally on surviving cliques), error drops dramatically: Feature MLP drops from $6.44 \pm 0.18$ to $4.50 \pm 0.65$; Static GCN drops from $12.92 \pm 3.85$ to $6.48 \pm 6.15$; Temporal GNN No-GRU drops from $13.26 \pm 4.71$ to $\mathbf{3.13 \pm 0.59}$ ($\mathbf{r = 0.7140}$); and Full Temporal GNN Warm-Start drops from $8.00 \pm 0.91$ to $\mathbf{3.84 \pm 0.47}$ ($\mathbf{r = 0.7066}$). Both Two-Stage Temporal GNN models decisively outperform the Persistence Baseline ($5.04$).

---

### Figure 4: Predicted vs. Ground-Truth Future Tension Scatter Plot (`fig4_predicted_vs_true.png`)

**Caption:**
**Figure 4: Scatter distribution of predicted future clique tension ($\hat{T}^{(t+1)}$) versus ground truth ($T^{(t+1)}$) on held-out test snapshots 36–42 ($N = 109$) under warm-start temporal state propagation.**
Green circles indicate persisting cliques ($N = 35$, $J \ge 0.5$ under 1-to-1 bipartite matching) where ground-truth future tension is strictly positive. Red crosses denote dissolved cliques ($N = 74$, $J < 0.5$) assigned target zero under Protocol A. The dashed black line represents perfect forecasting ($y = x$). The cluster of dissolved cliques along the $y=0$ axis reflects Enron's late-2001 organizational collapse (67.9% structural dissolution). In the single-stage model, predictions for persisting cliques are dragged toward zero, whereas the two-stage formulation isolates conditional tension regression on valid survivors.

---

### Figure 5: Longitudinal Structural Silo Risk Index (SRI) Trajectories (`fig5_silo_risk_trajectories.png`)

**Caption:**
**Figure 5: Evolving longitudinal Silo Risk Index ($\text{SRI}(v) = \sum_{C \ni v} T(C)$) for the five employees with highest cumulative structural tension over the 44 snapshots.**
Trajectories are labeled by authenticated email prefixes from repository `node-labels.txt` (Node 0: `sara.shackleton`, Node 1: `mark.taylor`, Node 2: `tana.jones`, Node 3: `jeff.dasovich`, Node 4: `steven.kean`). Note that SRI is a mathematically defined graph-theoretic tension metric reflecting simultaneous membership in high-conflict, cohesive cliques; it does not constitute empirical evidence of personal wrongdoing, workplace conflict, or verified administrative titles. Peak structural risk concentrated during the 2000–2001 operational crisis before declining rapidly during post-filing organizational dissolution.

---

### Figure 6: Peak Conflict Meta-Graph Topology (`fig6_peak_conflict_metagraph.png`)

**Caption:**
**Figure 6: Topological structure of the clique-conflict meta-graph $\mathcal{G}_{\mathcal{C}}^{(28)}$ corresponding to Snapshot 28 (March 3, 2001 – April 2, 2001).**
Snapshot 28 was selected by reproducible quantitative criteria: it contained the maximum simultaneous number of active maximal cliques ($K = 30$) and inter-clique communication edges ($|E| = 449$). Meta-nodes represent maximal cliques, with node diameter scaled proportionally to clique size $|C|$ and node color mapped to observed clique tension $T(C)$ (yellow to dark red). Edges between meta-nodes indicate non-zero Inter-Clique Communication Sparsity ($\text{ICCS} > 0$), with edge width proportional to conflict weight. High-tension cliques cluster tightly at the core of the organizational communication structure.

---

### Figure 7: Network Fragility Analysis Under Targeted Node Removal (`fig7_network_fragility.png`)

**Caption:**
**Figure 7: Audited network collapse simulation under targeted employee removal comparing structural metrics.**
Relative size of the Giant Connected Component ($|\text{GCC}| / |\text{GCC}_{\text{initial}}|$) plotted against the fraction of employees removed (0% to 50%).
**(a) In-Sample Disruption on Graph $G^{(28)}$ (Initial GCC = 103 / 105 nodes, 98.1%):** Betweenness centrality causes the most rapid structural fragmentation (reducing GCC to 38.8% at 20% removal), substantially outperforming Observed SRI (55.3%), Degree Centrality (67.0%), and Predicted SRI (72.8%). Random removal (50 independent seeds, shaded band = 95% confidence interval) results in 75.5% $\pm$ 2.3% GCC size at 20% removal.
**(b) Forward Out-of-Sample Disruption on Graph $G^{(29)}$ (Initial GCC = 105 / 107 nodes):** Node removal order determined at snapshot $t=28$ applied to subsequent graph $G^{(29)}$. Betweenness Centrality remains the most potent disruptor (64.0% GCC at 20% removal), followed by Observed SRI Persistence (68.4%) and Degree Centrality (73.7%). Predicted SRI from the Temporal GNN (75.4% at 20% removal) is statistically indistinguishable from random removal (75.4% $\pm$ 3.8% at 50 seeds), disproving the claim that the model forecasts future network fragility points.
