"""
Revision experiments (response to reviewer report, Oct 2026).

Everything reported in the revised manuscript is produced by this one script so
that tables cannot drift from the saved results again.

What it adds over benchmark_audit.py:
  1. Missing controls: conditional-only single-stage regressors (trained on
     survivors, no classifier), recalibrated persistence (a*T + b), ridge AR on
     tension history, gradient-boosted trees, and survival-weighted persistence
     (p_hat * T) for Protocol A.
  2. Survival-classifier evaluation: AUROC, AUPRC, Brier score and base rates
     per split. Protocol A composition is stated explicitly: E[T] = p_hat * T_cond.
  3. Statistics: block bootstrap over snapshots with paired differences, and a
     rolling-origin (expanding window) evaluation that includes pre-collapse
     windows.
  4. Measurement checks: share of "dissolutions" caused by top-K truncation,
     broadcast-email cliques, tension vs. clique count, trivially dissolved
     final snapshots, sensitivity to tau, K', window/stride and recipient caps,
     and survival under relaxed group definitions (Louvain communities).
  5. Construct checks on SRI (correlation with degree and clique count) and an
     area-under-curve summary of the fragility experiment.

Model hyperparameters are identical to benchmark_audit.py (hidden 16, head 8,
dropout 0.1, Adam lr 0.01, weight decay 1e-4, 35 full-batch epochs, model
selection on validation loss, lambda = 1). "GCN" here is the architecture that
benchmark_audit.py ran twice under two names ("Static GCN" and "Temporal GNN
No-GRU"); the two classes are identical.

Usage:
    python3 src/training/revision_experiments.py            # everything
    python3 src/training/revision_experiments.py --quick    # fewer seeds
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(1)

import networkx as nx
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

import src.models.gnn as gnn_module
from src.data.loader import load_node_labels, load_raw_vectors, parse_simplex_events, SimplexEvent
from src.data.snapshots import build_graph_snapshots
from src.graph.cliques import extract_maximal_cliques, match_cliques_bipartite
from src.graph.meta_graph import build_clique_meta_graph
from src.graph.scores import compute_silo_risk_index
from src.models.baselines import FeatureOnlyMLP, StaticGCNModel
from src.models.temporal_gnn import TemporalGNN
from src.training.trainer import build_two_stage_targets

# ---------------------------------------------------------------------------
# Speed-up: cache the dense normalized adjacency per meta-graph. The math is
# unchanged; benchmark_audit.py rebuilt it with a Python loop on every call.
# ---------------------------------------------------------------------------
_orig_adj = gnn_module.build_dense_normalized_adjacency
_adj_cache: Dict[Tuple[int, int], torch.Tensor] = {}


def _cached_adj(num_nodes, edge_index, edge_weights=None, add_self_loops=True):
    key = (id(edge_index), num_nodes)
    if key not in _adj_cache:
        _adj_cache[key] = _orig_adj(num_nodes, edge_index, edge_weights, add_self_loops)
    return _adj_cache[key]


gnn_module.build_dense_normalized_adjacency = _cached_adj

DATA_DIR = "Dataset/email-Enron"
OUT_DIR = Path("outputs/revision")
HIDDEN = 16
EPOCHS = 35
LR = 0.01
WD = 1e-4

ORIG_FEATS = ["clique_size", "conflict_degree", "tension", "avg_internal_comm", "external_comm_vol"]
EXTRA_FEATS = ["age", "has_prev", "lag_tension", "delta_tension", "num_cliques_in_snapshot"]


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------
@dataclass
class Pipeline:
    config: Dict
    snapshots: list
    events_by_snapshot: List[List[Tuple[int, ...]]]
    all_cliques: List[List[Tuple[int, ...]]]
    metas: list
    p: list
    c: list
    z: list
    mask: list
    records: list
    tab: Dict[int, np.ndarray] = field(default_factory=dict)  # k -> [n_cliques, n_feats]
    valid: List[int] = field(default_factory=list)


_EVENTS_CACHE: Dict[str, List[SimplexEvent]] = {}


def load_events() -> List[SimplexEvent]:
    if "ev" not in _EVENTS_CACHE:
        nverts, simplices, times = load_raw_vectors(DATA_DIR)
        _EVENTS_CACHE["ev"] = parse_simplex_events(nverts, simplices, times)
    return _EVENTS_CACHE["ev"]


def build_pipeline(window_days=30.0, stride_days=None, top_k: Optional[int] = 30, tau=0.5,
                   max_participants: Optional[int] = None) -> Pipeline:
    events = load_events()
    if max_participants is not None:
        events = [e for e in events if len(e.participants) <= max_participants]
    snapshots = build_graph_snapshots(events, window_days=window_days, stride_days=stride_days, sort_events=True)

    # participant sets per snapshot (for the broadcast-email check)
    ev_sorted = sorted(events, key=lambda e: e.timestamp_raw)
    ev_times = np.array([e.timestamp_raw for e in ev_sorted])
    events_by_snapshot = []
    for s in snapshots:
        lo = np.searchsorted(ev_times, s.start_time_ms, side="left")
        hi = np.searchsorted(ev_times, s.end_time_ms, side="left")
        events_by_snapshot.append([tuple(sorted(set(ev_sorted[i].participants))) for i in range(lo, hi)])

    all_cliques = [extract_maximal_cliques(s.graph, min_size=3, top_k=None) for s in snapshots]
    retained = [extract_maximal_cliques(s.graph, min_size=3, top_k=top_k) for s in snapshots]
    metas = [build_clique_meta_graph(s, r) for s, r in zip(snapshots, retained)]
    tb = build_two_stage_targets(metas, all_snapshots_cliques=all_cliques, delta=1,
                                 jaccard_threshold=tau, matching_method="greedy")
    pl = Pipeline(
        config=dict(window_days=window_days, stride_days=stride_days or window_days, top_k=top_k, tau=tau,
                    max_participants=max_participants),
        snapshots=snapshots, events_by_snapshot=events_by_snapshot, all_cliques=all_cliques, metas=metas,
        p=tb.persistence_targets, c=tb.conditional_targets, z=tb.zero_fill_targets, mask=tb.persisting_masks,
        records=tb.disappearance_records,
    )
    pl.valid = [i for i, t in enumerate(pl.z) if t is not None]
    build_tabular(pl, tau)
    return pl


def build_tabular(pl: Pipeline, tau: float) -> None:
    """Original 5 features plus age / tension-history features (backward matching only, no leakage)."""
    prev_age = None
    prev_T = None
    prev_meta = None
    for k, m in enumerate(pl.metas):
        n = m.num_cliques
        if n == 0:
            pl.tab[k] = np.zeros((0, len(ORIG_FEATS) + len(EXTRA_FEATS)))
            m.features_hist = torch.zeros((0, len(ORIG_FEATS) + len(EXTRA_FEATS)))
            prev_age, prev_T, prev_meta = None, None, m
            continue
        X = m.features.numpy().astype(float)
        T = m.clique_tensions.numpy().astype(float)
        age = np.zeros(n)
        has_prev = np.zeros(n)
        lag = T.copy()
        if prev_meta is not None and prev_meta.num_cliques > 0:
            res = match_cliques_bipartite(prev_meta.cliques, m.cliques, threshold=tau, method="greedy")
            for ci, pi in res.curr_to_prev.items():
                age[ci] = prev_age[pi] + 1
                has_prev[ci] = 1
                lag[ci] = prev_T[pi]
        extra = np.column_stack([age, has_prev, lag, T - lag, np.full(n, n)])
        pl.tab[k] = np.column_stack([X, extra])
        # same rows as a tensor, for the "+hist" neural ablations
        m.features_hist = torch.tensor(pl.tab[k], dtype=torch.float32)
        prev_age, prev_T, prev_meta = age, T, m


def split_indices(pl: Pipeline):
    v = pl.valid
    n_train = int(len(v) * 0.7)
    n_val = int(len(v) * 0.15)
    return v[:n_train], v[n_train:n_train + n_val], v[n_train + n_val:]


def stack_rows(pl: Pipeline, idxs: List[int]):
    """Return dict of arrays over all cliques at transitions idxs (rows ordered by k, then clique index)."""
    X, p, c, z, mask, blk = [], [], [], [], [], []
    for k in idxs:
        if pl.metas[k].num_cliques == 0 or pl.z[k] is None:
            continue
        X.append(pl.tab[k])
        p.append(pl.p[k].numpy())
        c.append(pl.c[k].numpy())
        z.append(pl.z[k].numpy())
        mask.append(pl.mask[k].numpy())
        blk.append(np.full(pl.metas[k].num_cliques, k))
    if not X:
        e = np.zeros(0)
        return dict(X=np.zeros((0, len(ORIG_FEATS) + len(EXTRA_FEATS))), p=e, c=e, z=e, mask=e.astype(bool), blk=e)
    return dict(X=np.vstack(X), p=np.concatenate(p), c=np.concatenate(c), z=np.concatenate(z),
                mask=np.concatenate(mask).astype(bool), blk=np.concatenate(blk))


# ---------------------------------------------------------------------------
# Tabular models
# ---------------------------------------------------------------------------
T_COL = ORIG_FEATS.index("tension")
AGE_COL = len(ORIG_FEATS) + EXTRA_FEATS.index("age")
AR_COLS = [T_COL] + [len(ORIG_FEATS) + EXTRA_FEATS.index(f) for f in ["lag_tension", "has_prev", "delta_tension"]]
GBDT_COLS = list(range(len(ORIG_FEATS))) + [AGE_COL]
ALL_COLS = list(range(len(ORIG_FEATS) + len(EXTRA_FEATS)))


def _log_feats(X):
    # heavy-tailed volume features; log1p is monotone so trees are unaffected
    X = X.copy()
    for j in [ORIG_FEATS.index("avg_internal_comm"), ORIG_FEATS.index("external_comm_vol")]:
        X[:, j] = np.log1p(X[:, j])
    return X


def fit_tabular(tr, seed=0) -> Dict[str, object]:
    """Fit all tabular models on a training row set. Returns callables X -> dict of outputs."""
    out = {}
    Xs = tr["X"][tr["mask"]]
    ys = tr["c"][tr["mask"]]

    # Recalibrated persistence a*T + b on survivors
    lr = LinearRegression().fit(Xs[:, [T_COL]], ys) if len(ys) > 1 else None
    out["recal"] = lr

    # Ridge AR on tension history (survivors)
    sc_ar = StandardScaler().fit(Xs[:, AR_COLS]) if len(ys) > 1 else None
    ar = Ridge(alpha=1.0).fit(sc_ar.transform(Xs[:, AR_COLS]), ys) if len(ys) > 1 else None
    out["ar"] = (sc_ar, ar)

    # Ridge on all features (survivors)
    Xl = _log_feats(Xs)
    sc_all = StandardScaler().fit(Xl) if len(ys) > 1 else None
    rall = Ridge(alpha=1.0).fit(sc_all.transform(Xl), ys) if len(ys) > 1 else None
    out["ridge_all"] = (sc_all, rall)

    # GBDT regressor on 5 features + age (survivors)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
                                   random_state=seed).fit(Xs[:, GBDT_COLS], ys) if len(ys) > 5 else None
    out["gbdt"] = gb

    # Survival classifiers on all rows
    Xa = _log_feats(tr["X"])
    sc_cls = StandardScaler().fit(Xa)
    pa = tr["p"]
    if len(np.unique(pa)) == 2:
        logit = LogisticRegression(C=1.0, max_iter=2000).fit(sc_cls.transform(Xa), pa)
        gbc = GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
                                         random_state=seed).fit(tr["X"][:, GBDT_COLS], pa)
    else:
        logit, gbc = None, None
    out["cls"] = (sc_cls, logit, gbc)
    out["base_rate"] = float(pa.mean()) if len(pa) else 0.0

    # Direct zero-fill GBDT (single-stage, all rows)
    out["gbdt_zero"] = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
                                                 random_state=seed).fit(tr["X"][:, GBDT_COLS], tr["z"]) if len(pa) > 5 else None
    return out


def predict_tabular(models, X) -> Dict[str, np.ndarray]:
    n = len(X)
    T = X[:, T_COL]
    pred = {"persistence": T.copy()}
    pred["recal"] = models["recal"].predict(X[:, [T_COL]]) if models["recal"] is not None else T
    sc, ar = models["ar"]
    pred["ar"] = ar.predict(sc.transform(X[:, AR_COLS])) if ar is not None else T
    sc, ra = models["ridge_all"]
    pred["ridge_all"] = ra.predict(sc.transform(_log_feats(X))) if ra is not None else T
    pred["gbdt"] = models["gbdt"].predict(X[:, GBDT_COLS]) if models["gbdt"] is not None else T
    for kk in ["recal", "ar", "ridge_all", "gbdt"]:
        pred[kk] = np.clip(pred[kk], 0, None)
    sc, logit, gbc = models["cls"]
    pred["p_logit"] = logit.predict_proba(sc.transform(_log_feats(X)))[:, 1] if logit is not None else np.full(n, models["base_rate"])
    pred["p_gbdt"] = gbc.predict_proba(X[:, GBDT_COLS])[:, 1] if gbc is not None else np.full(n, models["base_rate"])
    pred["p_base"] = np.full(n, models["base_rate"])
    pred["gbdt_zero"] = np.clip(models["gbdt_zero"].predict(X[:, GBDT_COLS]), 0, None) if models["gbdt_zero"] is not None else np.zeros(n)
    return pred


# ---------------------------------------------------------------------------
# Neural models (same classes and training loop as benchmark_audit.py)
# ---------------------------------------------------------------------------
N_HIST = len(ORIG_FEATS) + len(EXTRA_FEATS)


class MLPEncoder(nn.Module):
    """Two-layer MLP with the same shape as SpatialGCN but no message passing (ignores edges)."""

    def __init__(self, in_features=5, hidden_dim=HIDDEN, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_features, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index=None, edge_weights=None):
        return self.lin2(self.dropout(torch.relu(self.lin1(x))))


def make_model(kind: str):
    if kind == "mlph":
        return FeatureOnlyMLP(in_features=N_HIST, hidden_dim=HIDDEN)
    if kind == "gcnh":
        return StaticGCNModel(in_features=N_HIST, hidden_dim=HIDDEN, dropout=0.1)
    if kind == "mlpgru":
        model = TemporalGNN(in_features=5, hidden_dim=HIDDEN, dropout=0.1, jaccard_threshold=0.5, one_to_one_matching=True)
        model.spatial_gnn = MLPEncoder(5, HIDDEN, 0.1)
        return model
    if kind == "mlp":
        return FeatureOnlyMLP(in_features=5, hidden_dim=HIDDEN)
    if kind == "gcn":
        return StaticGCNModel(in_features=5, hidden_dim=HIDDEN, dropout=0.1)
    if kind == "gru":
        return TemporalGNN(in_features=5, hidden_dim=HIDDEN, dropout=0.1, jaccard_threshold=0.5, one_to_one_matching=True)
    raise ValueError(kind)


def nn_forward(model, kind, m, prev_m, prev_h):
    """Returns (persist_logits, cond_pred, hidden)."""
    if m.num_cliques == 0:
        e = torch.zeros(0)
        return e, e, torch.zeros((0, HIDDEN))
    if kind in ("mlp", "mlph"):
        h = model.encoder(m.features if kind == "mlp" else m.features_hist)
        return model.persistence_head(h).squeeze(-1), torch.nn.functional.softplus(model.net(h).squeeze(-1)), h
    if kind in ("gcn", "gcnh"):
        h = model.spatial_gnn(m.features if kind == "gcn" else m.features_hist, m.edge_index, m.edge_weights)
        return model.persistence_head(h).squeeze(-1), torch.nn.functional.softplus(model.head(h).squeeze(-1)), h
    _, _, cond, h = model.forward_two_stage(m, prev_m, prev_h)
    return model.persistence_head(h).squeeze(-1), cond, h


def nn_loss(mode, logits, cond, p, c, z, mask, mse, bce):
    if mode == "zero":
        return mse(cond, z)
    lc = mse(cond[mask], c[mask]) if mask.sum() > 0 else None
    if mode == "cond":
        return lc
    lp = bce(logits, p)
    return lp + lc if lc is not None else lp


def train_nn(pl: Pipeline, kind: str, mode: str, train_idx, val_idx, seed: int, epochs=EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(kind)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    recurrent = kind in ("gru", "mlpgru")
    last_needed = max(val_idx) if val_idx else max(train_idx)
    best, best_state = float("inf"), None
    train_set, val_set = set(train_idx), set(val_idx)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pm, ph = None, None
        for k in range(0, max(train_idx) + 1):
            m = pl.metas[k]
            if not recurrent and k not in train_set:
                continue
            logits, cond, h = nn_forward(model, kind, m, pm, ph)
            if k in train_set and m.num_cliques > 0 and pl.z[k] is not None:
                loss = nn_loss(mode, logits, cond, pl.p[k], pl.c[k], pl.z[k], pl.mask[k], mse, bce)
                if loss is not None:
                    loss.backward()
            pm, ph = m, (h.detach() if h is not None and h.numel() > 0 else None)
        opt.step()
        if not val_idx:
            best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
            continue
        model.eval()
        vl, vc = 0.0, 0
        with torch.no_grad():
            pm, ph = None, None
            for k in range(0, last_needed + 1):
                m = pl.metas[k]
                if not recurrent and k not in val_set:
                    continue
                logits, cond, h = nn_forward(model, kind, m, pm, ph)
                if k in val_set and m.num_cliques > 0 and pl.z[k] is not None:
                    loss = nn_loss(mode, logits, cond, pl.p[k], pl.c[k], pl.z[k], pl.mask[k], mse, bce)
                    if loss is not None:
                        vl += loss.item()
                        vc += 1
                pm, ph = m, h
        avg = vl / vc if vc else float("inf")
        if avg < best or best_state is None:
            best = avg
            best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_nn(pl: Pipeline, model, kind: str, eval_idx) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Warm-start roll-forward from snapshot 0; returns k -> (p_hat, cond_hat)."""
    out = {}
    eval_set = set(eval_idx)
    with torch.no_grad():
        pm, ph = None, None
        for k in range(0, max(eval_idx) + 1):
            m = pl.metas[k]
            if kind not in ("gru", "mlpgru") and k not in eval_set:
                continue
            logits, cond, h = nn_forward(model, kind, m, pm, ph)
            if k in eval_set and m.num_cliques > 0:
                out[k] = (torch.sigmoid(logits).numpy(), cond.numpy())
            pm, ph = m, h
    return out


NN_VARIANTS = [
    # (name, kind, mode)
    ("MLP zero-fill", "mlp", "zero"),
    ("GCN zero-fill", "gcn", "zero"),
    ("GCN+GRU zero-fill", "gru", "zero"),
    ("MLP cond-only", "mlp", "cond"),
    ("GCN cond-only", "gcn", "cond"),
    ("GCN+GRU cond-only", "gru", "cond"),
    ("MLP two-stage", "mlp", "two"),
    ("GCN two-stage", "gcn", "two"),
    ("GCN+GRU two-stage", "gru", "two"),
    # ablation ladder: history as features instead of recurrence, and recurrence without the graph
    ("MLP+hist cond-only", "mlph", "cond"),
    ("GCN+hist cond-only", "gcnh", "cond"),
    ("MLP+GRU cond-only", "mlpgru", "cond"),
    ("MLP+hist two-stage", "mlph", "two"),
    ("GCN+hist two-stage", "gcnh", "two"),
    ("MLP+GRU two-stage", "mlpgru", "two"),
]


# ---------------------------------------------------------------------------
# Metrics and bootstrap
# ---------------------------------------------------------------------------
def reg_metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    mse = float(np.mean((yhat - y) ** 2))
    mae = float(np.mean(np.abs(yhat - y)))
    r = float(stats.pearsonr(y, yhat)[0]) if len(y) > 2 and np.std(yhat) > 1e-9 and np.std(y) > 1e-9 else float("nan")
    rho = float(stats.spearmanr(y, yhat)[0]) if len(y) > 2 and np.std(yhat) > 1e-9 and np.std(y) > 1e-9 else float("nan")
    r2 = 1 - mse / float(np.var(y)) if np.var(y) > 0 else float("nan")
    return dict(mse=mse, mae=mae, r=r, rho=rho, r2=r2)


def block_bootstrap(blocks: np.ndarray, y: np.ndarray, preds: Dict[str, np.ndarray], ref: str,
                    B=5000, seed=0) -> Dict[str, Dict[str, float]]:
    """
    Resample whole snapshots with replacement. preds values are arrays [n_seeds, N] or [N].
    Returns, per model, the 95% CI of MSE and of the paired MSE difference vs `ref`,
    plus the share of resamples where the model beats ref.
    """
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    idx_by_b = {b: np.where(blocks == b)[0] for b in ub}
    P = {k: (v if v.ndim == 2 else v[None, :]) for k, v in preds.items()}
    res = {k: {"mse": [], "diff": []} for k in P}
    for _ in range(B):
        draw = rng.choice(ub, size=len(ub), replace=True)
        ii = np.concatenate([idx_by_b[b] for b in draw])
        ref_mse = np.mean((P[ref][:, ii] - y[ii]) ** 2, axis=1).mean()
        for k, v in P.items():
            m = np.mean((v[:, ii] - y[ii]) ** 2, axis=1).mean()
            res[k]["mse"].append(m)
            res[k]["diff"].append(m - ref_mse)
    out = {}
    for k, d in res.items():
        mse_a, diff_a = np.array(d["mse"]), np.array(d["diff"])
        out[k] = dict(mse_lo=float(np.percentile(mse_a, 2.5)), mse_hi=float(np.percentile(mse_a, 97.5)),
                      diff_lo=float(np.percentile(diff_a, 2.5)), diff_hi=float(np.percentile(diff_a, 97.5)),
                      p_better=float(np.mean(diff_a < 0)))
    return out


def per_block_paired_test(blocks, y, a, b):
    """Wilcoxon signed-rank on per-snapshot MSE (a vs b)."""
    ub = np.unique(blocks)
    da, db = [], []
    for u in ub:
        ii = blocks == u
        da.append(np.mean((a[ii] - y[ii]) ** 2))
        db.append(np.mean((b[ii] - y[ii]) ** 2))
    da, db = np.array(da), np.array(db)
    if len(ub) < 5 or np.allclose(da, db):
        return dict(n_blocks=int(len(ub)), wins=int(np.sum(da < db)), p=float("nan"))
    return dict(n_blocks=int(len(ub)), wins=int(np.sum(da < db)), p=float(stats.wilcoxon(da, db).pvalue))


# ---------------------------------------------------------------------------
# Experiment 1: main chronological split (repro + controls + classifier eval)
# ---------------------------------------------------------------------------
def _nn_job(args):
    cfg, name, kind, mode, seed, train_idx, val_idx, eval_idx = args
    pl = _get_pipeline(cfg)
    model = train_nn(pl, kind, mode, train_idx, val_idx, seed)
    return name, seed, predict_nn(pl, model, kind, eval_idx)


_PIPE_CACHE: Dict[str, Pipeline] = {}


def _get_pipeline(cfg: Dict) -> Pipeline:
    key = json.dumps(cfg, sort_keys=True)
    if key not in _PIPE_CACHE:
        _PIPE_CACHE[key] = build_pipeline(**cfg)
    return _PIPE_CACHE[key]


_POOL = None


def run_parallel(jobs, workers):
    # one pool for the whole run: forking a second time on macOS can crash once
    # the Metal backend has been touched in the parent
    global _POOL
    if _POOL is None:
        _POOL = mp.get_context("fork").Pool(workers)
    return _POOL.map(_nn_job, jobs, chunksize=1)


def experiment_main(cfg, seeds, workers) -> Dict:
    pl = _get_pipeline(cfg)
    tr_i, va_i, te_i = split_indices(pl)
    tr, va, te = stack_rows(pl, tr_i), stack_rows(pl, va_i), stack_rows(pl, te_i)

    # target stats and base rates
    info = dict(
        train_transitions=[tr_i[0], tr_i[-1]], val_transitions=[va_i[0], va_i[-1]], test_transitions=[te_i[0], te_i[-1]],
        n_rows=dict(train=int(len(tr["p"])), val=int(len(va["p"])), test=int(len(te["p"]))),
        survival_rate=dict(train=float(tr["p"].mean()), val=float(va["p"].mean()), test=float(te["p"].mean())),
        test_cliques_by_snapshot={int(k): int(pl.metas[k].num_cliques) for k in te_i},
        test_survivors_by_snapshot={int(k): int(pl.mask[k].sum()) for k in te_i if pl.mask[k] is not None},
        target_B_test=dict(mean=float(te["c"][te["mask"]].mean()), var=float(te["c"][te["mask"]].var())),
        target_A_test=dict(mean=float(te["z"].mean()), var=float(te["z"].var())),
        target_B_train=dict(mean=float(tr["c"][tr["mask"]].mean()), var=float(tr["c"][tr["mask"]].var())),
    )

    # tabular models (deterministic except GBDT subsampling -> seed)
    tab_B = {k: [] for k in ["persistence", "recal", "ar", "ridge_all", "gbdt"]}
    tab_A = {k: [] for k in ["zero", "train_mean", "persist_x_base", "persist_x_plogit", "recal_x_plogit",
                             "gbdt_hurdle", "gbdt_zero", "persistence"]}
    cls_scores = {"logit": [], "gbdt": []}
    for s in seeds:
        mdl = fit_tabular(tr, seed=s)
        pr = predict_tabular(mdl, te["X"])
        for k in tab_B:
            tab_B[k].append(pr[k][te["mask"]])
        tab_A["zero"].append(np.zeros(len(te["z"])))
        tab_A["train_mean"].append(np.full(len(te["z"]), tr["z"].mean()))
        tab_A["persistence"].append(pr["persistence"])
        tab_A["persist_x_base"].append(pr["persistence"] * pr["p_base"])
        tab_A["persist_x_plogit"].append(pr["persistence"] * pr["p_logit"])
        tab_A["recal_x_plogit"].append(pr["recal"] * pr["p_logit"])
        tab_A["gbdt_hurdle"].append(pr["gbdt"] * pr["p_gbdt"])
        tab_A["gbdt_zero"].append(pr["gbdt_zero"])
        cls_scores["logit"].append(pr["p_logit"])
        cls_scores["gbdt"].append(pr["p_gbdt"])

    # historical mean baseline exactly as benchmark_audit.py (exact clique key, default 0.5)
    hist: Dict[tuple, List[float]] = {}
    for k in tr_i:
        for cl, t in zip(pl.metas[k].cliques, pl.metas[k].clique_tensions.tolist()):
            hist.setdefault(cl, []).append(t)
    hm = []
    n_seen = 0
    for k in te_i:
        for cl in pl.metas[k].cliques:
            h = hist.get(cl)
            n_seen += int(bool(h))
            hm.append(np.mean(h) if h else 0.5)
    hm = np.array(hm)
    tab_A["hist_mean"] = [hm for _ in seeds]
    tab_B["hist_mean"] = [hm[te["mask"]] for _ in seeds]
    info["hist_mean_test_cliques_with_history"] = n_seen

    # neural models
    jobs = [(cfg, name, kind, mode, s, tr_i, va_i, te_i) for (name, kind, mode) in NN_VARIANTS for s in seeds]
    results = run_parallel(jobs, workers)
    nn_A, nn_B, nn_P = {}, {}, {}
    for name, seed, out in results:
        cond = np.concatenate([out[k][1] for k in te_i if k in out])
        ph = np.concatenate([out[k][0] for k in te_i if k in out])
        mode = [m for (n, _, m) in NN_VARIANTS if n == name][0]
        nn_B.setdefault(name, []).append(cond[te["mask"]])
        nn_A.setdefault(name, []).append(cond if mode in ("zero",) else (ph * cond if mode == "two" else cond))
        if mode == "two":
            nn_P.setdefault(name, []).append(ph)
            nn_A.setdefault(name + " (persist x p_hat)", []).append(te["X"][:, T_COL] * ph)

    yB, yA = te["c"][te["mask"]], te["z"]
    blkB, blkA = te["blk"][te["mask"]], te["blk"]

    def summarize(pred_lists, y):
        rows = {}
        for k, lst in pred_lists.items():
            ms = [reg_metrics(y, p) for p in lst]
            agg = {}
            for met in ms[0]:
                vals = np.array([m[met] for m in ms], float)
                agg[met + "_mean"] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
                agg[met + "_std"] = float(np.nanstd(vals)) if not np.all(np.isnan(vals)) else float("nan")
            agg["skill_vs_persistence"] = 1 - agg["mse_mean"] / reg_metrics(y, pred_lists["persistence"][0])["mse"]
            # metrics of the seed-averaged (ensemble) prediction
            ens = reg_metrics(y, np.mean(np.vstack(lst), axis=0))
            agg["ensemble_mse"] = ens["mse"]
            agg["ensemble_r"] = ens["r"]
            rows[k] = agg
        return rows

    allB = {**tab_B, **nn_B}
    allA = {**tab_A, **{k: v for k, v in nn_A.items()}}
    sumB, sumA = summarize(allB, yB), summarize(allA, yA)
    bootB = block_bootstrap(blkB, yB, {k: np.vstack(v) for k, v in allB.items()}, ref="persistence")
    bootA = block_bootstrap(blkA, yA, {k: np.vstack(v) for k, v in allA.items()}, ref="persist_x_base")

    # Protocol A without the trivially-dissolved last non-empty snapshot (40 -> 41 has no cliques)
    trivial = [k for k in te_i if pl.metas[k].num_cliques > 0 and k + 1 < len(pl.metas) and pl.metas[k + 1].num_cliques == 0]
    keepA = ~np.isin(blkA, trivial)
    sumA_nontrivial = summarize({k: [p[keepA] for p in v] for k, v in allA.items()}, yA[keepA])
    info["trivially_dissolved_snapshots"] = trivial
    info["trivially_dissolved_rows"] = int((~keepA).sum())

    # Classifier evaluation
    cls_eval = {}
    yP = te["p"]
    for name, lst in {**cls_scores, **nn_P}.items():
        au, ap, br = [], [], []
        for p in lst:
            au.append(roc_auc_score(yP, p))
            ap.append(average_precision_score(yP, p))
            br.append(brier_score_loss(yP, p))
        cls_eval[name] = dict(auroc_mean=float(np.mean(au)), auroc_std=float(np.std(au)),
                              auprc_mean=float(np.mean(ap)), auprc_std=float(np.std(ap)),
                              brier_mean=float(np.mean(br)), brier_std=float(np.std(br)),
                              mean_p_hat=float(np.mean(lst)))
    cls_eval["base_rate (train)"] = dict(auroc_mean=0.5, auprc_mean=float(yP.mean()),
                                         brier_mean=float(brier_score_loss(yP, np.full(len(yP), tr["p"].mean()))),
                                         mean_p_hat=float(tr["p"].mean()))
    # reliability bins for the logistic model (seed-independent)
    p_l = cls_scores["logit"][0]
    bins = np.linspace(0, 1, 6)
    rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        ii = (p_l >= lo) & (p_l < hi if hi < 1 else p_l <= hi)
        if ii.sum():
            rel.append(dict(bin=[float(lo), float(hi)], n=int(ii.sum()), mean_p=float(p_l[ii].mean()), frac_pos=float(yP[ii].mean())))
    cls_eval["logit_reliability"] = rel

    # Paired per-snapshot Wilcoxon (only 4 test snapshots carry survivors, so these are reported but weak)
    paired_B = {k: per_block_paired_test(blkB, yB, np.mean(np.vstack(v), 0), allB["persistence"][0]) for k, v in allB.items()}

    # Shrinkage check: regress prediction on persistence in Protocol B
    shrink = {}
    for k in ["persistence", "recal", "GCN two-stage", "GCN+GRU two-stage", "MLP two-stage", "GCN cond-only", "MLP cond-only"]:
        if k in allB:
            pmean = np.mean(np.vstack(allB[k]), 0)
            slope, intercept = np.polyfit(yB, pmean, 1)
            shrink[k] = dict(pred_std=float(pmean.std()), slope_on_truth=float(slope), mean_pred=float(pmean.mean()))
    shrink["truth"] = dict(std=float(yB.std()), mean=float(yB.mean()))

    return dict(info=info, protocol_B=sumB, protocol_A=sumA, protocol_A_nontrivial=sumA_nontrivial,
                bootstrap_B=bootB, bootstrap_A=bootA, classifier=cls_eval, paired_B=paired_B, shrinkage=shrink)


# ---------------------------------------------------------------------------
# Experiment 2: rolling-origin evaluation
# ---------------------------------------------------------------------------
ROLL_NN = [("MLP cond-only", "mlp", "cond"), ("GCN cond-only", "gcn", "cond"),
           ("GCN two-stage", "gcn", "two"), ("GCN+GRU two-stage", "gru", "two"), ("MLP two-stage", "mlp", "two"),
           ("GCN+GRU cond-only", "gru", "cond"), ("MLP+hist two-stage", "mlph", "two"),
           ("GCN+hist two-stage", "gcnh", "two"), ("MLP+GRU two-stage", "mlpgru", "two")]


def experiment_rolling(cfg, seeds, workers, min_origin=16) -> Dict:
    pl = _get_pipeline(cfg)
    origins = [k for k in pl.valid if k >= min_origin and pl.mask[k] is not None and pl.mask[k].sum() >= 1]
    jobs, tab_rows = [], []
    origin_split = {}
    for o in origins:
        hist = [k for k in pl.valid if k < o]
        n_val = max(2, int(round(0.15 * len(hist))))
        tr_i, va_i = hist[:-n_val], hist[-n_val:]
        origin_split[o] = (tr_i, va_i)
        for name, kind, mode in ROLL_NN:
            for s in seeds:
                jobs.append((cfg, name, kind, mode, s, tr_i, va_i, [o]))
    results = run_parallel(jobs, workers)
    nn_out: Dict[Tuple[str, int], Dict[int, Tuple]] = {}
    for name, seed, out in results:
        nn_out.setdefault((name, seed), {}).update(out)

    yB, blkB, yA, blkA = [], [], [], []
    predB: Dict[str, List[np.ndarray]] = {}
    predA: Dict[str, List[np.ndarray]] = {}
    clsP, clsY = {}, []

    def add(d, key, arr_by_seed):
        d.setdefault(key, []).append(arr_by_seed)

    for o in origins:
        tr_i, va_i = origin_split[o]
        tr = stack_rows(pl, tr_i + va_i)  # tabular models have no early stopping, use all history
        te = stack_rows(pl, [o])
        mdl_by_seed = [fit_tabular(tr, seed=s) for s in seeds]
        prs = [predict_tabular(m, te["X"]) for m in mdl_by_seed]
        msk = te["mask"]
        yB.append(te["c"][msk]); blkB.append(np.full(msk.sum(), o))
        yA.append(te["z"]); blkA.append(np.full(len(te["z"]), o))
        clsY.append(te["p"])
        for k in ["persistence", "recal", "ar", "ridge_all", "gbdt"]:
            add(predB, k, np.vstack([p[k][msk] for p in prs]))
        add(predA, "zero", np.zeros((len(seeds), len(te["z"]))))
        add(predA, "persist_x_base", np.vstack([p["persistence"] * p["p_base"] for p in prs]))
        add(predA, "persist_x_plogit", np.vstack([p["persistence"] * p["p_logit"] for p in prs]))
        add(predA, "gbdt_hurdle", np.vstack([p["gbdt"] * p["p_gbdt"] for p in prs]))
        add(predA, "persistence", np.vstack([p["persistence"] for p in prs]))
        add(clsP, "logit", np.vstack([p["p_logit"] for p in prs]))
        add(clsP, "gbdt", np.vstack([p["p_gbdt"] for p in prs]))
        for name, kind, mode in ROLL_NN:
            ph = np.vstack([nn_out[(name, s)][o][0] for s in seeds])
            cd = np.vstack([nn_out[(name, s)][o][1] for s in seeds])
            add(predB, name, cd[:, msk])
            if mode == "two":
                add(predA, name, ph * cd)
                add(clsP, name, ph)

    yB, blkB, yA, blkA = map(np.concatenate, (yB, blkB, yA, blkA))
    clsY = np.concatenate(clsY)
    PB = {k: np.hstack(v) for k, v in predB.items()}
    PA = {k: np.hstack(v) for k, v in predA.items()}
    CP = {k: np.hstack(v) for k, v in clsP.items()}

    def summ(P, y, blk, ref):
        out = {}
        ref_mse = np.mean((P[ref].mean(0) - y) ** 2)
        for k, v in P.items():
            per_seed = [reg_metrics(y, v[i]) for i in range(v.shape[0])]
            out[k] = dict(mse_mean=float(np.mean([m["mse"] for m in per_seed])), mse_std=float(np.std([m["mse"] for m in per_seed])),
                          mae_mean=float(np.mean([m["mae"] for m in per_seed])),
                          r_mean=float(np.nanmean([m["r"] for m in per_seed])),
                          skill=float(1 - np.mean([m["mse"] for m in per_seed]) / ref_mse))
        boot = block_bootstrap(blk, y, P, ref=ref)
        for k in out:
            out[k].update(boot[k])
            out[k].update({"wilcoxon_" + kk: vv for kk, vv in per_block_paired_test(blk, y, P[k].mean(0), P[ref].mean(0)).items()})
        return out

    # split pre-collapse vs collapse
    pre = blkB < 36
    res = dict(
        origins=[int(o) for o in origins],
        n_B=int(len(yB)), n_A=int(len(yA)),
        protocol_B=summ(PB, yB, blkB, "persistence"),
        protocol_B_pre_collapse=summ({k: v[:, pre] for k, v in PB.items()}, yB[pre], blkB[pre], "persistence"),
        protocol_B_collapse=summ({k: v[:, ~pre] for k, v in PB.items()}, yB[~pre], blkB[~pre], "persistence") if (~pre).sum() else {},
        protocol_A=summ(PA, yA, blkA, "persist_x_base"),
        classifier={k: dict(auroc=float(np.mean([roc_auc_score(clsY, v[i]) for i in range(v.shape[0])])),
                            auprc=float(np.mean([average_precision_score(clsY, v[i]) for i in range(v.shape[0])])),
                            brier=float(np.mean([brier_score_loss(clsY, v[i]) for i in range(v.shape[0])])))
                    for k, v in CP.items()},
        survival_rate=float(clsY.mean()),
    )
    # per-origin survival rate / persistence MSE (for the figure)
    res["per_origin"] = []
    for o in origins:
        ii, jj = blkB == o, blkA == o
        res["per_origin"].append(dict(origin=int(o), n_cliques=int(jj.sum()), n_surv=int(ii.sum()),
                                      mse_persist=float(np.mean((PB["persistence"][0, ii] - yB[ii]) ** 2)),
                                      mse_recal=float(np.mean((PB["recal"][0, ii] - yB[ii]) ** 2)),
                                      mse_gcn2=float(np.mean((PB["GCN two-stage"][:, ii].mean(0) - yB[ii]) ** 2)),
                                      mse_gru2=float(np.mean((PB["GCN+GRU two-stage"][:, ii].mean(0) - yB[ii]) ** 2))))
    return res


# ---------------------------------------------------------------------------
# Experiment 2b: node-level SRI forecasts against the SRI actually observed at k+1
# ---------------------------------------------------------------------------
def experiment_node_level(cfg, seeds, workers) -> Dict:
    """
    Predicted node risk = sum over the cliques a node belongs to at k of the clique's
    predicted value (expected tension p_hat * T_cond for two-stage models). Truth is
    the observed SRI(v) at k+1, recomputed from the cliques of snapshot k+1, so nodes
    whose cliques dissolve and nodes joining new cliques are both scored. Spearman rho
    is computed over the active set: nodes in a retained clique at k or at k+1.
    """
    pl = _get_pipeline(cfg)
    tr_i, va_i, te_i = split_indices(pl)
    eval_k = [k for k in te_i if pl.metas[k].num_cliques > 0 and pl.metas[k + 1].num_cliques > 0]
    variants = [("GCN two-stage", "gcn", "two"), ("GCN+GRU two-stage", "gru", "two"), ("MLP two-stage", "mlp", "two"),
                ("GCN+hist two-stage", "gcnh", "two"), ("MLP+GRU two-stage", "mlpgru", "two")]
    jobs = [(cfg, n, kd, md, s, tr_i, va_i, te_i) for (n, kd, md) in variants for s in seeds]
    outs = run_parallel(jobs, workers)

    def node_scores(m, vals):
        sc = {}
        for c_idx, cl in enumerate(m.cliques):
            for v in cl:
                sc[v] = sc.get(v, 0.0) + float(vals[c_idx])
        return sc

    res: Dict[str, List[float]] = {}
    for k in eval_k:
        m0, m1 = pl.metas[k], pl.metas[k + 1]
        truth = node_scores(m1, m1.clique_tensions.numpy())
        active = sorted(set(truth) | set(n for cl in m0.cliques for n in cl))
        y = np.array([truth.get(v, 0.0) for v in active])
        G = pl.snapshots[k].graph
        cand = {
            "Persistence (observed SRI at k)": node_scores(m0, m0.clique_tensions.numpy()),
            "Degree at k": {v: G.degree(v) if G.has_node(v) else 0 for v in active},
            "Clique count at k": {v: len(m0.node_to_clique_map.get(v, [])) for v in active},
        }
        for name, s_, out in outs:
            if k in out:
                ph, cd = out[k]
                cand.setdefault(name, [])
                cand[name].append(node_scores(m0, ph * cd))
        for name, sc in cand.items():
            if isinstance(sc, list):
                rhos = [stats.spearmanr(y, [d.get(v, 0.0) for v in active])[0] for d in sc]
                res.setdefault(name, []).append(float(np.nanmean(rhos)))
            else:
                res.setdefault(name, []).append(float(stats.spearmanr(y, [sc.get(v, 0.0) for v in active])[0]))
    return dict(eval_snapshots=eval_k, active_set="nodes in a retained clique at k or k+1",
                spearman_by_snapshot=res, spearman_mean={k: float(np.mean(v)) for k, v in res.items()})


# ---------------------------------------------------------------------------
# Experiment 3: measurement checks and sensitivity
# ---------------------------------------------------------------------------
def measurement_checks(pl: Pipeline) -> Dict:
    # categories over cliques that are unmatched under the 1-to-1 rule (p == 0); an
    # unmatched clique whose best J >= tau partner was taken by another clique is
    # labelled "lost_one_to_one"
    cats = {}
    for k in pl.valid:
        for r, pv in zip(pl.records[k] or [], (pl.p[k].tolist() if pl.p[k] is not None else [])):
            if pv == 1.0:
                continue
            c = "lost_one_to_one" if r["category"] == "matched_retained" else r["category"]
            cats[c] = cats.get(c, 0) + 1
    n_dissolved = sum(cats.values())

    # broadcast cliques: clique contained in a single email's participant set
    n_cl, n_bcast, n_exact = 0, 0, 0
    for k, m in enumerate(pl.metas):
        evs = [set(e) for e in pl.events_by_snapshot[k] if len(e) >= 3]
        for cl in m.cliques:
            s = set(cl)
            n_cl += 1
            if any(s <= e for e in evs):
                n_bcast += 1
            if any(s == e for e in evs):
                n_exact += 1

    # tension vs clique count
    Ks, meanT = [], []
    for m in pl.metas:
        if m.num_cliques > 1:
            Ks.append(m.num_cliques)
            meanT.append(float(m.clique_tensions.mean()))
    rho_KT = stats.spearmanr(Ks, meanT)

    # truncation saturation
    top_k = pl.config["top_k"]
    saturated = [i for i, (m, a) in enumerate(zip(pl.metas, pl.all_cliques)) if top_k and len(a) > top_k]

    # J for a 4-clique with two members swapped
    return dict(
        disappearance_categories=cats, n_dissolved=n_dissolved,
        share_truncation=cats.get("omitted_by_truncation", 0) / max(1, n_dissolved),
        share_split=cats.get("split_or_reorganized", 0) / max(1, n_dissolved),
        share_structural=cats.get("dissolved_structural", 0) / max(1, n_dissolved),
        share_lost_one_to_one=cats.get("lost_one_to_one", 0) / max(1, n_dissolved),
        n_retained_cliques=n_cl, share_inside_single_email=n_bcast / max(1, n_cl), share_equal_single_email=n_exact / max(1, n_cl),
        spearman_K_vs_meanT=dict(rho=float(rho_KT[0]), p=float(rho_KT[1]), n=len(Ks)),
        saturated_snapshots=saturated,
        all_cliques_per_snapshot=[len(a) for a in pl.all_cliques],
    )


def survival_rate_simple(pl: Pipeline, idxs=None):
    idxs = idxs if idxs is not None else pl.valid
    p = np.concatenate([pl.p[k].numpy() for k in idxs if pl.p[k] is not None and len(pl.p[k])] or [np.zeros(0)])
    return float(p.mean()) if len(p) else float("nan"), int(len(p))


def louvain_survival(pl: Pipeline, tau=0.5, seed=0) -> Dict:
    """Relaxed group definition: Louvain communities (size >= 3) tracked with the same 1-to-1 Jaccard rule."""
    comms = []
    for s in pl.snapshots:
        G = s.graph
        if G.number_of_edges() == 0:
            comms.append([])
            continue
        cs = nx.community.louvain_communities(G, weight="weight", seed=seed)
        comms.append([tuple(sorted(c)) for c in cs if len(c) >= 3])
    surv, tot = 0, 0
    sizes = []
    for k in range(len(comms) - 1):
        if not comms[k]:
            continue
        res = match_cliques_bipartite(comms[k], comms[k + 1], threshold=tau, method="greedy")
        surv += len(res.prev_to_curr)
        tot += len(comms[k])
        sizes += [len(c) for c in comms[k]]
    return dict(survival_rate=surv / max(1, tot), n_groups=tot, mean_size=float(np.mean(sizes)) if sizes else float("nan"))


def kcore_survival(pl: Pipeline, k_core=3, tau=0.5) -> Dict:
    """Relaxed definition: connected components of the k-core."""
    groups = []
    for s in pl.snapshots:
        G = s.graph
        if G.number_of_nodes() == 0:
            groups.append([])
            continue
        H = nx.k_core(G, k_core)
        groups.append([tuple(sorted(c)) for c in nx.connected_components(H) if len(c) >= 3])
    surv, tot = 0, 0
    for k in range(len(groups) - 1):
        if not groups[k]:
            continue
        res = match_cliques_bipartite(groups[k], groups[k + 1], threshold=tau, method="greedy")
        surv += len(res.prev_to_curr)
        tot += len(groups[k])
    return dict(survival_rate=surv / max(1, tot), n_groups=tot)


def quick_B_scores(pl: Pipeline, seeds=(0,)) -> Dict:
    """Tabular Protocol B comparison on the 70/15/15 split and under rolling origin (cheap; for sensitivity tables)."""
    tr_i, va_i, te_i = split_indices(pl)
    tr, te = stack_rows(pl, tr_i + va_i), stack_rows(pl, te_i)
    out = {}
    if te["mask"].sum() >= 3 and tr["mask"].sum() >= 6:
        pr = predict_tabular(fit_tabular(tr, seed=0), te["X"])
        y = te["c"][te["mask"]]
        for k in ["persistence", "recal", "ar", "gbdt"]:
            out["split_" + k] = reg_metrics(y, pr[k][te["mask"]])["mse"]
        out["split_n_test_survivors"] = int(te["mask"].sum())
    # rolling origin over the second half of the series
    half = pl.valid[len(pl.valid) // 3]
    ys, P = [], {k: [] for k in ["persistence", "recal", "ar", "gbdt"]}
    for o in [k for k in pl.valid if k >= half and pl.mask[k] is not None and pl.mask[k].sum() > 0]:
        trr = stack_rows(pl, [k for k in pl.valid if k < o])
        if trr["mask"].sum() < 6:
            continue
        te_o = stack_rows(pl, [o])
        pr = predict_tabular(fit_tabular(trr, seed=0), te_o["X"])
        ys.append(te_o["c"][te_o["mask"]])
        for k in P:
            P[k].append(pr[k][te_o["mask"]])
    if ys:
        y = np.concatenate(ys)
        for k in P:
            out["roll_" + k] = reg_metrics(y, np.concatenate(P[k]))["mse"]
        out["roll_n"] = int(len(y))
    return out


def experiment_sensitivity() -> Dict:
    base = dict(window_days=30.0, stride_days=None, top_k=30, tau=0.5, max_participants=None)
    grid = []
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7]:
        grid.append(("tau", tau, {**base, "tau": tau}))
    for kk in [10, 20, 30, 50, None]:
        grid.append(("top_k", kk, {**base, "top_k": kk}))
    for w, s in [(14.0, None), (30.0, None), (60.0, None), (30.0, 15.0), (60.0, 30.0)]:
        grid.append(("window/stride", f"{int(w)}/{int(s or w)}", {**base, "window_days": w, "stride_days": s}))
    for mp_ in [3, 5, 10, None]:
        grid.append(("max_participants", mp_, {**base, "max_participants": mp_}))
    rows = []
    for param, val, cfg in grid:
        t0 = time.time()
        pl = build_pipeline(**cfg)
        tr_i, va_i, te_i = split_indices(pl)
        mc = measurement_checks(pl)
        row = dict(param=param, value=str(val), n_snapshots=len(pl.snapshots),
                   n_cliques=int(sum(m.num_cliques for m in pl.metas)),
                   survival_all=survival_rate_simple(pl)[0],
                   survival_train=survival_rate_simple(pl, tr_i)[0], survival_test=survival_rate_simple(pl, te_i)[0],
                   share_truncation=mc["share_truncation"], share_inside_single_email=mc["share_inside_single_email"])
        row.update(quick_B_scores(pl))
        rows.append(row)
        print(f"  sensitivity {param}={val}: surv={row['survival_all']:.3f} ({time.time() - t0:.1f}s)", flush=True)
    return dict(rows=rows)


# ---------------------------------------------------------------------------
# Experiment 4: SRI construct checks and fragility AUC
# ---------------------------------------------------------------------------
def sri_construct(pl: Pipeline) -> Dict:
    rhos_deg, rhos_cnt, rhos_str = [], [], []
    for k, (s, m) in enumerate(zip(pl.snapshots, pl.metas)):
        if m.num_cliques < 2:
            continue
        tens = {c: float(t) for c, t in zip(m.cliques, m.clique_tensions.tolist())}
        sri = compute_silo_risk_index(m.cliques, tens)
        nodes = [n for n in sri if sri[n] > 0]
        if len(nodes) < 5:
            continue
        G = s.graph
        a = [sri[n] for n in nodes]
        rhos_deg.append(stats.spearmanr(a, [G.degree(n) for n in nodes])[0])
        rhos_cnt.append(stats.spearmanr(a, [len(m.node_to_clique_map.get(n, [])) for n in nodes])[0])
        rhos_str.append(stats.spearmanr(a, [G.degree(n, weight="weight") for n in nodes])[0])
    f = lambda x: dict(mean=float(np.nanmean(x)), median=float(np.nanmedian(x)), n_snapshots=len(x))
    return dict(sri_vs_degree=f(rhos_deg), sri_vs_clique_count=f(rhos_cnt), sri_vs_strength=f(rhos_str))


def fragility_auc() -> Dict:
    path = Path("outputs/audit/fragility_results.csv")
    if not path.exists():
        return {}
    data: Dict[str, Dict[str, List[Tuple[float, float, float]]]] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            data.setdefault(r["target_graph"], {}).setdefault(r["strategy"], []).append(
                (float(r["fraction_removed"]), float(r["relative_gcc_size"]), float(r["std_err"])))
    out = {}
    for g, strat in data.items():
        out[g] = {}
        for s, pts in strat.items():
            pts.sort()
            x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
            sel = x <= 0.2 + 1e-9
            out[g][s] = dict(auc_0_20=float(np.trapezoid(y[sel], x[sel]) / 0.2), gcc_at_20=float(y[sel][-1]),
                             max_fraction=float(x.max()))
    return out


# ---------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return str(o)


def save(results, only):
    name = "revision_results.json" if only == "all" else f"revision_results_{only}.json"
    with open(OUT_DIR / name, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--only", default="all")
    args = ap.parse_args()
    seeds = [42, 123, 456] if args.quick else [42, 123, 456, 789, 999, 7, 2024, 31337, 11, 5]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(window_days=30.0, stride_days=None, top_k=30, tau=0.5, max_participants=None)
    pl = _get_pipeline(cfg)  # build before fork so workers inherit it
    run_parallel([], args.workers)  # start the worker pool now
    results = {"seeds": seeds, "config": cfg,
               "hyperparameters": dict(hidden=HIDDEN, head=HIDDEN // 2, dropout=0.1, lr=LR, weight_decay=WD, epochs=EPOCHS,
                                       optimizer="Adam, one full-batch step per epoch", model_selection="min validation loss",
                                       lambda_cond=1.0, composition="E[T]=p_hat*T_cond (no threshold)",
                                       features="raw (unnormalized) 5 meta-node features, as in the original runs")}
    t0 = time.time()
    if args.only in ("all", "main"):
        print("[1/4] main split ...", flush=True)
        results["main"] = experiment_main(cfg, seeds, args.workers)
        print(f"      done {time.time() - t0:.0f}s", flush=True)
        save(results, args.only)
    if args.only in ("all", "rolling"):
        print("[2/4] rolling origin ...", flush=True)
        results["rolling"] = experiment_rolling(cfg, seeds[:5], args.workers)
        print(f"      done {time.time() - t0:.0f}s", flush=True)
        save(results, args.only)
    if args.only in ("all", "node"):
        print("[2b] node-level SRI ...", flush=True)
        results["node_level"] = experiment_node_level(cfg, seeds, args.workers)
        save(results, args.only)
    if args.only in ("all", "measure"):
        print("[3/4] measurement + sensitivity ...", flush=True)
        results["measurement"] = measurement_checks(pl)
        results["relaxed_groups"] = dict(maximal_cliques_top30=survival_rate_simple(pl)[0],
                                         louvain=louvain_survival(pl), kcore3=kcore_survival(pl, 3), kcore2=kcore_survival(pl, 2))
        results["sensitivity"] = experiment_sensitivity()
        print(f"      done {time.time() - t0:.0f}s", flush=True)
        save(results, args.only)
    if args.only in ("all", "sri"):
        print("[4/4] SRI construct + fragility AUC ...", flush=True)
        results["sri_construct"] = sri_construct(pl)
        results["fragility_auc"] = fragility_auc()
    save(results, args.only)
    print(f"saved in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
