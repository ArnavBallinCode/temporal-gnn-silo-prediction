"""
Second-round revision experiments.

Addresses the second review of the manuscript:
  1. Neural models are trained properly: standardized inputs, early stopping on
     validation loss (up to 300 epochs), a small hyperparameter search, and a
     residual output head  T_hat = softplus(T_now + g(z))  that starts at persistence.
  2. Graph encoders that keep the clique's own signal: GraphSAGE-style layers
     (separate self and neighbour weights) besides the original GCN, and GBDT with
     ICCS-weighted neighbour aggregates as features.
  3. Five group constructions (maximal cliques, cliques not contained in a single
     email, cliques on edges seen in >= 2 emails, clique-percolation communities (k = 3), Louvain
     communities), each with the raw target and a normalized target T / (K_k - 1).
  4. Dependent-data statistics on the rolling origin: Diebold-Mariano tests with a
     Newey-West (HAC) variance and the Harvey-Leybourne-Newbold correction, Holm
     adjustment, circular moving-block bootstrap intervals for MSE ratios, MAE,
     median absolute error and a scale-free relative MAE.
  5. Stronger survival baselines (last month's survival fraction, a discrete-time
     hazard model with age bins), a Tweedie-loss GBDT for the zero-heavy target, a
     decomposition of Protocol A error, and calibration on the rolling origin.
  6. Identity-free node-level evaluation across all rolling origins, with intervals.

All numbers in the second revision of the paper come from this script:
    python3 src/training/revision2_experiments.py
Output: outputs/revision2/results.json (+ training curves).
"""

from __future__ import annotations

import argparse
import json
import math
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
import torch.nn.functional as F

torch.set_num_threads(1)

import networkx as nx
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.graph.cliques import extract_maximal_cliques, match_cliques_bipartite
from src.graph.meta_graph import build_clique_meta_graph
from src.models.gnn import build_dense_normalized_adjacency
from src.training.revision_experiments import build_pipeline as build_base_pipeline
from src.training.trainer import build_two_stage_targets

OUT_DIR = Path("outputs/revision2")
TAU = 0.5
TOP_K = 30
CONSTRUCTIONS = ["clique", "clique_multi", "clique_w2", "cpm3", "louvain"]


# ---------------------------------------------------------------------------
# Data: groups, targets, features
# ---------------------------------------------------------------------------
@dataclass
class Snap:
    k: int
    groups: List[Tuple[int, ...]]
    X: np.ndarray            # [n, 5] raw meta-node features (size, conflict_degree, tension, avg_internal, ext_vol)
    T: np.ndarray            # current tension (target scale of the mode)
    K: int
    A_hat: torch.Tensor      # GCN propagation (with self loops, symmetric normalisation)
    A_row: torch.Tensor      # row-normalised ICCS adjacency without self loops (neighbour mean)
    nbr_mean: np.ndarray     # ICCS-weighted neighbour mean tension
    nbr_max: np.ndarray
    age: np.ndarray
    lag: np.ndarray
    prev_match: Dict[int, int]  # current group index -> group index at k-1
    p: Optional[np.ndarray] = None
    c: Optional[np.ndarray] = None
    node_map: Dict[int, List[int]] = field(default_factory=dict)


@dataclass
class Data:
    construction: str
    mode: str                # "raw" or "norm"
    snaps: List[Snap]
    valid: List[int]         # transitions k with targets
    true_sri_next: Dict[int, Dict[int, float]]
    graphs: list


_BASE = {}


def base_pipeline():
    if "pl" not in _BASE:
        _BASE["pl"] = build_base_pipeline(window_days=30.0, stride_days=None, top_k=TOP_K, tau=TAU)
    return _BASE["pl"]


def build_groups(construction: str, top_k: int = TOP_K):
    pl = base_pipeline()
    out = []
    for k, s in enumerate(pl.snapshots):
        G = s.graph
        if G.number_of_edges() == 0:
            out.append([])
            continue
        if construction == "clique":
            g = extract_maximal_cliques(G, 3, top_k)
        elif construction == "clique_multi":
            evs = [set(e) for e in pl.events_by_snapshot[k] if len(e) >= 3]
            allc = extract_maximal_cliques(G, 3, None)
            keep = [c for c in allc if not any(set(c) <= e for e in evs)]
            g = sorted(keep, key=lambda c: (-len(c), c))[:top_k]
        elif construction == "clique_w2":
            H = nx.Graph()
            H.add_edges_from((u, v) for u, v, w in G.edges(data="weight") if w >= 2)
            g = extract_maximal_cliques(H, 3, top_k) if H.number_of_edges() else []
        elif construction == "cpm3":
            # clique percolation (k = 3): overlapping communities of adjacent triangles (Palla et al.)
            comps = [tuple(sorted(c)) for c in nx.community.k_clique_communities(G, 3)]
            g = sorted(comps, key=lambda c: (-len(c), c))[:top_k]
        elif construction == "louvain":
            cs = nx.community.louvain_communities(G, weight="weight", seed=0)
            comps = [tuple(sorted(c)) for c in cs if len(c) >= 3]
            g = sorted(comps, key=lambda c: (-len(c), c))[:top_k]
        else:
            raise ValueError(construction)
        out.append(g)
    return pl, out


_DATA: Dict[Tuple[str, str], Data] = {}


def get_data(construction: str, mode: str) -> Data:
    key = (construction, mode)
    if key in _DATA:
        return _DATA[key]
    pl, groups = build_groups(construction)
    metas = [build_clique_meta_graph(s, g) for s, g in zip(pl.snapshots, groups)]
    tb = build_two_stage_targets(metas, all_snapshots_cliques=None, delta=1, jaccard_threshold=TAU, matching_method="greedy")
    snaps: List[Snap] = []
    prev = None
    for k, m in enumerate(metas):
        n = m.num_cliques
        K = n
        scale = max(K - 1, 1) if mode == "norm" else 1.0
        if n == 0:
            snaps.append(Snap(k, [], np.zeros((0, 5)), np.zeros(0), 0, torch.zeros((0, 0)), torch.zeros((0, 0)),
                              np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), {}))
            prev = snaps[-1]
            continue
        X = m.features.numpy().astype(float).copy()
        Traw = m.clique_tensions.numpy().astype(float)
        T = Traw / scale
        X[:, 2] = T
        if mode == "norm":
            X[:, 1] = X[:, 1] / scale
        A = np.zeros((n, n))
        ei = m.edge_index.numpy()
        ew = m.edge_weights.numpy()
        for (a, b), w in zip(ei.T, ew):
            A[a, b] = w
        A_hat = build_dense_normalized_adjacency(n, m.edge_index, m.edge_weights)
        rs = A.sum(1, keepdims=True)
        A_row = np.divide(A, rs, out=np.zeros_like(A), where=rs > 0)
        nbr_mean = A_row @ T
        nbr_max = np.array([T[A[i] > 0].max() if (A[i] > 0).any() else 0.0 for i in range(n)])
        age = np.zeros(n)
        lag = T.copy()
        pm: Dict[int, int] = {}
        if prev is not None and len(prev.groups) > 0:
            res = match_cliques_bipartite(prev.groups, m.cliques, threshold=TAU, method="greedy")
            for ci, pi in res.curr_to_prev.items():
                age[ci] = prev.age[pi] + 1
                lag[ci] = prev.T[pi]
                pm[ci] = pi
        s = Snap(k, list(m.cliques), X, T, K, A_hat, torch.tensor(A_row, dtype=torch.float32), nbr_mean, nbr_max,
                 age, lag, pm, node_map=dict(m.node_to_clique_map))
        snaps.append(s)
        prev = s
    valid = []
    for k, s in enumerate(snaps):
        if tb.zero_fill_targets[k] is None or len(s.groups) == 0:
            continue
        nxt = snaps[k + 1]
        scale_next = max(nxt.K - 1, 1) if mode == "norm" else 1.0
        s.p = tb.persistence_targets[k].numpy().astype(float)
        s.c = tb.conditional_targets[k].numpy().astype(float) / scale_next
        valid.append(k)
    # observed node SRI at k+1 in the same scale
    true_sri = {}
    for k in valid:
        nxt = snaps[k + 1]
        d = {}
        for gi, g in enumerate(nxt.groups):
            for v in g:
                d[v] = d.get(v, 0.0) + float(nxt.T[gi])
        true_sri[k] = d
    data = Data(construction, mode, snaps, valid, true_sri, [s.graph for s in pl.snapshots])
    _DATA[key] = data
    return data


# tabular feature matrix (no redundant has-history flag)
FEAT_NAMES = ["size", "conflict_degree", "tension", "avg_internal", "ext_vol", "age", "lag_tension", "delta_tension",
              "num_groups", "nbr_mean_tension", "nbr_max_tension"]
T_COL = 2


def tab_X(s: Snap) -> np.ndarray:
    n = len(s.groups)
    return np.column_stack([s.X, s.age, s.lag, s.T - s.lag, np.full(n, s.K), s.nbr_mean, s.nbr_max])


def rows(d: Data, idxs: List[int]):
    X, p, c, blk = [], [], [], []
    for k in idxs:
        s = d.snaps[k]
        if s.p is None:
            continue
        X.append(tab_X(s)); p.append(s.p); c.append(s.c); blk.append(np.full(len(s.p), k))
    if not X:
        return dict(X=np.zeros((0, len(FEAT_NAMES))), p=np.zeros(0), c=np.zeros(0), blk=np.zeros(0))
    return dict(X=np.vstack(X), p=np.concatenate(p), c=np.concatenate(c), blk=np.concatenate(blk))


def logf(X):
    X = X.copy()
    for j in [1, 3, 4, 8]:
        X[:, j] = np.log1p(np.maximum(X[:, j], 0))
    return X


# ---------------------------------------------------------------------------
# Tabular models
# ---------------------------------------------------------------------------
AR_COLS = [2, 6, 7]                 # tension, lag, delta
GB_COLS = [0, 1, 2, 3, 4, 5]        # five features + age
GBN_COLS = GB_COLS + [9, 10]        # + neighbour aggregates


def age_bins(a):
    return np.column_stack([(a == 0), (a == 1), (a == 2), (a >= 3)]).astype(float)


def fit_predict_tabular(tr, te, prev_surv_rate: float, seed: int = 0) -> Dict[str, np.ndarray]:
    out = {}
    m = tr["p"] == 1
    Xs, ys = tr["X"][m], tr["c"][m]
    T = te["X"][:, T_COL]
    n = len(T)
    out["persistence"] = T.copy()
    if len(ys) > 2:
        lr = LinearRegression().fit(Xs[:, [T_COL]], ys)
        out["recal"] = np.clip(lr.predict(te["X"][:, [T_COL]]), 0, None)
        sc = StandardScaler().fit(Xs[:, AR_COLS])
        out["ar"] = np.clip(Ridge(1.0).fit(sc.transform(Xs[:, AR_COLS]), ys).predict(sc.transform(te["X"][:, AR_COLS])), 0, None)
    else:
        out["recal"] = out["ar"] = T.copy()
    if len(ys) > 8:
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=seed)
        out["gbdt"] = np.clip(gb.fit(Xs[:, GB_COLS], ys).predict(te["X"][:, GB_COLS]), 0, None)
        gbn = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=seed)
        out["gbdt_nbr"] = np.clip(gbn.fit(Xs[:, GBN_COLS], ys).predict(te["X"][:, GBN_COLS]), 0, None)
    else:
        out["gbdt"] = out["gbdt_nbr"] = T.copy()
    # survival probabilities
    base = float(tr["p"].mean()) if len(tr["p"]) else 0.4
    out["p_base"] = np.full(n, base)
    out["p_last"] = np.full(n, prev_surv_rate if not math.isnan(prev_surv_rate) else base)
    if len(np.unique(tr["p"])) == 2:
        sc = StandardScaler().fit(logf(tr["X"]))
        lg = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(logf(tr["X"])), tr["p"])
        out["p_logit"] = lg.predict_proba(sc.transform(logf(te["X"])))[:, 1]
        hz_cols = [0, 2, 8]  # size, tension, number of groups (log where heavy) + age bins
        Hx = lambda X: np.column_stack([age_bins(X[:, 5]), logf(X)[:, hz_cols]])
        sch = StandardScaler().fit(Hx(tr["X"])[:, 4:])
        Htr = np.column_stack([Hx(tr["X"])[:, :4], sch.transform(Hx(tr["X"])[:, 4:])])
        Hte = np.column_stack([Hx(te["X"])[:, :4], sch.transform(Hx(te["X"])[:, 4:])])
        hz = LogisticRegression(C=1.0, max_iter=3000, fit_intercept=False).fit(Htr, tr["p"])
        out["p_hazard"] = hz.predict_proba(Hte)[:, 1]
        gbc = GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=seed)
        out["p_gbdt"] = gbc.fit(tr["X"][:, GB_COLS], tr["p"]).predict_proba(te["X"][:, GB_COLS])[:, 1]
    else:
        out["p_logit"] = out["p_hazard"] = out["p_gbdt"] = np.full(n, base)
    # Tweedie-family GBDT (Poisson loss, power 1) on the zero-heavy Protocol A target (all rows).
    # LightGBM's Tweedie objective segfaults when loaded next to PyTorch on macOS, so we use scikit-learn.
    yA = tr["p"] * tr["c"]
    if len(tr["p"]) > 20 and yA.sum() > 0:
        from sklearn.ensemble import HistGradientBoostingRegressor
        tw = HistGradientBoostingRegressor(loss="poisson", max_iter=200, learning_rate=0.05, max_leaf_nodes=7,
                                           min_samples_leaf=5, random_state=seed)
        out["tweedie_gbdt"] = tw.fit(tr["X"][:, GB_COLS], yA).predict(te["X"][:, GB_COLS])
    else:
        out["tweedie_gbdt"] = np.zeros(n)
    return out


# ---------------------------------------------------------------------------
# Neural models (revision 2)
# ---------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, enc: str, recurrent: bool, residual: bool, hidden: int, dropout: float, base_rate: float, mean_c: float):
        super().__init__()
        self.enc, self.recurrent, self.residual = enc, recurrent, residual
        d, H = 5, hidden
        self.l1 = nn.Linear(d, H)
        self.l2 = nn.Linear(H, H)
        if enc == "sage":
            self.n1 = nn.Linear(d, H, bias=False)
            self.n2 = nn.Linear(H, H, bias=False)
        self.drop = nn.Dropout(dropout)
        if recurrent:
            self.gru = nn.GRUCell(H, H)
        self.reg = nn.Sequential(nn.Linear(H, H // 2), nn.ReLU(), nn.Linear(H // 2, 1))
        self.surv = nn.Sequential(nn.Linear(H, H // 2), nn.ReLU(), nn.Linear(H // 2, 1))
        with torch.no_grad():
            self.reg[2].weight.zero_()
            if residual:
                self.reg[2].bias.zero_()          # starts exactly at persistence
            else:
                self.reg[2].bias.fill_(math.log(math.expm1(max(mean_c, 1e-3))))
            self.surv[2].weight.zero_()
            self.surv[2].bias.fill_(math.log(base_rate / (1 - base_rate)))

    def encode(self, x, A_hat, A_row):
        if self.enc == "mlp":
            h = self.drop(F.relu(self.l1(x)))
            return self.l2(h)
        if self.enc == "gcn":
            h = self.drop(F.relu(A_hat @ self.l1(x)))
            return A_hat @ self.l2(h)
        h = self.drop(F.relu(self.l1(x) + self.n1(A_row @ x)))
        return self.l2(h) + self.n2(A_row @ h)

    def step(self, s: Snap, xs: torch.Tensor, Tt: torch.Tensor, prev_h):
        h = self.encode(xs, s.A_hat, s.A_row)
        if self.recurrent and prev_h is not None and s.prev_match:
            h = h.clone()
            ci = torch.tensor(list(s.prev_match.keys()))
            pi = torch.tensor(list(s.prev_match.values()))
            h[ci] = self.gru(h[ci], prev_h[pi])
        g = self.reg(h).squeeze(-1)
        cond = F.softplus(Tt + g) if self.residual else F.softplus(g)
        logit = self.surv(h).squeeze(-1)
        return logit, cond, h


MODEL_SPECS = {
    "MLP-res": ("mlp", False, True),
    "GCN-res": ("gcn", False, True),
    "SAGE-res": ("sage", False, True),
    "GCN+GRU-res": ("gcn", True, True),
    "GCN-plain": ("gcn", False, False),
}


def make_inputs(d: Data, train_idx: List[int]):
    """Standardise log-transformed features with training statistics."""
    Xtr = np.vstack([d.snaps[k].X for k in train_idx if len(d.snaps[k].groups)])
    L = Xtr.copy()
    for j in [1, 3, 4]:
        L[:, j] = np.log1p(np.maximum(L[:, j], 0))
    mu, sd = L.mean(0), L.std(0) + 1e-6
    xs = {}
    for s in d.snaps:
        if len(s.groups) == 0:
            continue
        Z = s.X.copy()
        for j in [1, 3, 4]:
            Z[:, j] = np.log1p(np.maximum(Z[:, j], 0))
        xs[s.k] = (torch.tensor((Z - mu) / sd, dtype=torch.float32), torch.tensor(s.T, dtype=torch.float32))
    return xs


def train_net(d: Data, name: str, mode: str, train_idx, val_idx, seed: int, hidden: int, lr: float,
              max_epochs: int = 300, patience: int = 30, record: bool = False):
    enc, rec, res = MODEL_SPECS[name]
    torch.manual_seed(seed)
    np.random.seed(seed)
    xs = make_inputs(d, train_idx)
    tr_p = np.concatenate([d.snaps[k].p for k in train_idx])
    tr_c = np.concatenate([d.snaps[k].c[d.snaps[k].p == 1] for k in train_idx])
    base = float(np.clip(tr_p.mean(), 0.05, 0.95))
    var_c = max(float(tr_c.var()), 1e-3) if len(tr_c) > 1 else 1.0
    net = Net(enc, rec, res, hidden, 0.1, base, float(tr_c.mean()) if len(tr_c) else 1.0)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    trs, vas = set(train_idx), set(val_idx)
    last = max(val_idx) if val_idx else max(train_idx)
    tens = {k: (torch.tensor(d.snaps[k].p, dtype=torch.float32), torch.tensor(d.snaps[k].c, dtype=torch.float32),
                torch.tensor(d.snaps[k].p == 1)) for k in d.valid}

    def loss_at(k, logit, cond):
        p, c, msk = tens[k]
        if mode == "zero":
            return F.mse_loss(cond, p * c) / var_c
        lc = F.mse_loss(cond[msk], c[msk]) / var_c if msk.any() else torch.tensor(0.0)
        if mode == "cond":
            return lc if msk.any() else None
        return F.binary_cross_entropy_with_logits(logit, p) + lc

    def run(train: bool, upto: int, which: set):
        net.train(train)
        total, cnt, ph = 0.0, 0, None
        for k in range(0, upto + 1):
            s = d.snaps[k]
            if len(s.groups) == 0:
                ph = None
                continue
            if not rec and k not in which:
                continue
            x, Tt = xs[k]
            logit, cond, h = net.step(s, x, Tt, ph)
            if k in which and k in tens:
                l = loss_at(k, logit, cond)
                if l is not None:
                    total = total + l
                    cnt += 1
            ph = h.detach() if rec else None
        return total, cnt

    best, best_state, bad, curve = float("inf"), None, 0, []
    for ep in range(max_epochs):
        opt.zero_grad()
        tl, tc = run(True, max(train_idx), trs)
        if tc:
            (tl / tc).backward()
            opt.step()
        with torch.no_grad():
            vl, vc = run(False, last, vas)
        v = float(vl / vc) if vc else float(tl / max(tc, 1))
        if record:
            curve.append((ep, float(tl.detach() if torch.is_tensor(tl) else tl) / max(tc, 1), v))
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {kk: t.clone() for kk, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:          # None only if every validation loss was NaN
        net.load_state_dict(best_state)
    net.eval()
    return net, xs, curve


def predict_net(d: Data, net: Net, xs, eval_idx) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    out, ph = {}, None
    es = set(eval_idx)
    with torch.no_grad():
        for k in range(0, max(eval_idx) + 1):
            s = d.snaps[k]
            if len(s.groups) == 0:
                ph = None
                continue
            if not net.recurrent and k not in es:
                continue
            x, Tt = xs[k]
            logit, cond, h = net.step(s, x, Tt, ph)
            if k in es:
                out[k] = (torch.sigmoid(logit).numpy(), cond.numpy())
            ph = h
    return out


# ---------------------------------------------------------------------------
# Parallel jobs
# ---------------------------------------------------------------------------
HP: Dict[str, Tuple[int, float]] = {}   # filled by tuning


def _job(args):
    construction, tmode, name, mode, seed, train_idx, val_idx, eval_idx, hidden, lr, record = args
    d = get_data(construction, tmode)
    net, xs, curve = train_net(d, name, mode, train_idx, val_idx, seed, hidden, lr, record=record)
    out = predict_net(d, net, xs, eval_idx)
    fit = {}
    if record:
        pr = predict_net(d, net, xs, train_idx)
        yt, yp, yp0 = [], [], []
        for k in train_idx:
            s = d.snaps[k]
            msk = s.p == 1
            yt.append(s.c[msk]); yp.append(pr[k][1][msk]); yp0.append(s.T[msk])
        yt, yp, yp0 = map(np.concatenate, (yt, yp, yp0))
        fit = dict(train_mse=float(np.mean((yp - yt) ** 2)), train_mse_persistence=float(np.mean((yp0 - yt) ** 2)))
    return (construction, tmode, name, mode, seed, eval_idx[0] if len(eval_idx) == 1 else -1), out, curve, fit


_POOL = None


def pmap(jobs, workers):
    global _POOL
    if _POOL is None:
        _POOL = mp.get_context("fork").Pool(workers)
    return _POOL.map(_job, jobs, chunksize=1)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def dm_test(d: np.ndarray):
    """Diebold-Mariano with Newey-West variance and HLN small-sample correction (h = 1)."""
    d = np.asarray(d, float)
    T = len(d)
    if T < 4 or np.allclose(d, d[0]):
        return float("nan"), float("nan")
    dbar = d.mean()
    L = int(np.floor(T ** (1 / 3)))
    g0 = np.mean((d - dbar) ** 2)
    var = g0
    for l in range(1, L + 1):
        gl = np.mean((d[l:] - dbar) * (d[:-l] - dbar))
        var += 2 * (1 - l / (L + 1)) * gl
    if var <= 0:
        var = g0
    stat = dbar / math.sqrt(var / T) * math.sqrt((T - 1) / T)
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return float(stat), float(p)


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted([(p, k) for k, p in pvals.items() if not math.isnan(p)])
    m, out, run = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    for k, p in pvals.items():
        out.setdefault(k, float("nan"))
    return out


def mbb_ratio_ci(per_origin_se_m: List[np.ndarray], per_origin_se_r: List[np.ndarray], block=3, B=4000, seed=0, alpha=0.05):
    """Circular moving-block bootstrap over origins for pooled-MSE ratio model/ref."""
    rng = np.random.default_rng(seed)
    n = len(per_origin_se_m)
    sm = np.array([a.sum() for a in per_origin_se_m]); nm = np.array([len(a) for a in per_origin_se_m])
    sr = np.array([a.sum() for a in per_origin_se_r])
    ratios = []
    nb = int(math.ceil(n / block))
    for _ in range(B):
        starts = rng.integers(0, n, nb)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        den = sr[idx].sum()
        if den > 0:
            ratios.append(sm[idx].sum() / den)
    ratios = np.array(ratios)
    if len(ratios) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(ratios, 100 * alpha / 2)), float(np.percentile(ratios, 100 * (1 - alpha / 2)))


def compare(per_origin: Dict[str, List[Tuple[np.ndarray, np.ndarray]]], ref: str, origins: List[int]) -> Dict:
    """per_origin[model] = list over origins of (y, yhat_mean_over_seeds)."""
    out, pv = {}, {}
    se = {m: [(yh - y) ** 2 for y, yh in v] for m, v in per_origin.items()}
    ae = {m: [np.abs(yh - y) for y, yh in v] for m, v in per_origin.items()}
    for m in per_origin:
        mse_o = np.array([a.mean() for a in se[m]])
        ref_o = np.array([a.mean() for a in se[ref]])
        stat, p = dm_test(mse_o - ref_o) if m != ref else (float("nan"), float("nan"))
        pv[m] = p
        allse, allae = np.concatenate(se[m]), np.concatenate(ae[m])
        rel = [np.log(a.mean() / b.mean()) for a, b in zip(ae[m], ae[ref]) if a.mean() > 0 and b.mean() > 0]
        lo, hi = mbb_ratio_ci(se[m], se[ref]) if m != ref else (1.0, 1.0)
        lo90, hi90 = mbb_ratio_ci(se[m], se[ref], alpha=0.10) if m != ref else (1.0, 1.0)
        out[m] = dict(mse=float(allse.mean()), mae=float(allae.mean()), median_ae=float(np.median(allae)),
                      rel_mae=float(np.exp(np.mean(rel))) if rel else float("nan"),
                      mse_ratio=float(allse.sum() / np.concatenate(se[ref]).sum()),
                      ratio_ci95=[lo, hi], ratio_ci90=[lo90, hi90], dm_stat=stat, dm_p=p,
                      wins=int(np.sum(mse_o < ref_o)), n_origins=len(origins))
    adj = holm({m: p for m, p in pv.items() if m != ref})
    for m in out:
        out[m]["dm_p_holm"] = adj.get(m, float("nan"))
    return out


def pair_equivalence(per_origin, a, b, margin=0.10):
    se_a = [(yh - y) ** 2 for y, yh in per_origin[a]]
    se_b = [(yh - y) ** 2 for y, yh in per_origin[b]]
    lo, hi = mbb_ratio_ci(se_a, se_b, alpha=0.10)
    ratio = float(np.concatenate(se_a).sum() / np.concatenate(se_b).sum())
    verdict = "equivalent" if (lo >= 1 - margin and hi <= 1 / (1 - margin)) else ("a better" if hi < 1 else ("b better" if lo > 1 else "inconclusive"))
    return dict(a=a, b=b, ratio=ratio, ci90=[lo, hi], margin=margin, verdict=verdict)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def auroc_ci(y, p, blk, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    ub = np.unique(blk)
    idx = {b: np.where(blk == b)[0] for b in ub}
    vals = []
    for _ in range(B):
        ii = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])
        if len(np.unique(y[ii])) == 2:
            vals.append(roc_auc_score(y[ii], p[ii]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def origins_for(d: Data, min_origin=16):
    return [k for k in d.valid if k >= min_origin and d.snaps[k].p is not None and d.snaps[k].p.sum() >= 1]


def split_hist(d: Data, o: int):
    hist = [k for k in d.valid if k < o]
    nv = max(2, int(round(0.15 * len(hist))))
    return hist[:-nv], hist[-nv:]


def tune(workers, seeds=(42,)):
    """Small grid on the chronological split of the default construction (validation loss)."""
    d = get_data("clique", "raw")
    n = len(d.valid)
    tr, va = d.valid[:int(0.7 * n)], d.valid[int(0.7 * n):int(0.85 * n)]
    grid = [(h, lr) for h in (16, 32) for lr in (1e-3, 3e-3, 1e-2)]
    jobs = [("clique", "raw", m, "two", s, tr, va, va, h, lr, True) for m in MODEL_SPECS for (h, lr) in grid for s in seeds]
    res = pmap(jobs, workers)
    best = {}
    for (key, out, curve, fit), job in zip(res, jobs):
        m, h, lr = job[2], job[8], job[9]
        v = min(c[2] for c in curve)
        if m not in best or v < best[m][0]:
            best[m] = (v, h, lr)
    for m, (v, h, lr) in best.items():
        HP[m] = (h, lr)
    return {m: dict(val_loss=v, hidden=h, lr=lr) for m, (v, h, lr) in best.items()}


def chrono_fit(workers, seed=42):
    """Training curves and train-set fit on the chronological split (default construction)."""
    d = get_data("clique", "raw")
    n = len(d.valid)
    tr, va, te = d.valid[:int(0.7 * n)], d.valid[int(0.7 * n):int(0.85 * n)], d.valid[int(0.85 * n):]
    jobs = [("clique", "raw", m, "two", seed, tr, va, te, HP[m][0], HP[m][1], True) for m in MODEL_SPECS]
    res = pmap(jobs, workers)
    out = {}
    for (key, preds, curve, fit), job in zip(res, jobs):
        out[job[2]] = dict(curve=curve, epochs_run=len(curve), best_epoch=int(np.argmin([c[2] for c in curve])), **fit)
    return out


def rolling(construction, tmode, models, modes, seeds, workers, with_node=False):
    d = get_data(construction, tmode)
    orgs = origins_for(d)
    jobs = []
    for o in orgs:
        tr, va = split_hist(d, o)
        for m in models:
            for md in modes:
                for s in seeds:
                    jobs.append((construction, tmode, m, md, s, tr, va, [o], HP[m][0], HP[m][1], False))
    res = pmap(jobs, workers)
    nn_out = {}
    for (key, out, curve, fit) in res:
        _, _, m, md, s, o = key
        nn_out[(m, md, s, o)] = out[o]
    perB, perA, cls, cls_y, cls_blk = {}, {}, {}, [], []
    node = {}
    decomp = {}
    for o in orgs:
        tr_i, va_i = split_hist(d, o)
        tr, te = rows(d, tr_i + va_i), rows(d, [o])
        prev = d.snaps[o - 1]
        prev_rate = float(prev.p.mean()) if prev.p is not None and len(prev.p) else float("nan")
        tab = fit_predict_tabular(tr, te, prev_rate)
        y, p = te["c"], te["p"]
        msk = p == 1
        yA = p * y
        def addB(name, yh):
            perB.setdefault(name, []).append((y[msk], yh[msk]))
        def addA(name, yh):
            perA.setdefault(name, []).append((yA, yh))
        for nm in ["persistence", "recal", "ar", "gbdt", "gbdt_nbr"]:
            addB(nm, tab[nm])
        T = tab["persistence"]
        addA("zeros", np.zeros_like(T))
        for pn in ["p_base", "p_last", "p_logit", "p_hazard", "p_gbdt"]:
            addA("persistence x " + pn[2:], T * tab[pn])
            cls.setdefault(pn[2:], []).append(tab[pn])
        addA("recal x hazard", tab["recal"] * tab["p_hazard"])
        addA("GBDT hurdle", tab["gbdt"] * tab["p_gbdt"])
        addA("Poisson GBDT", tab["tweedie_gbdt"])
        for m in models:
            for md in modes:
                P = np.mean([nn_out[(m, md, s, o)][0] for s in seeds], 0)
                C = np.mean([nn_out[(m, md, s, o)][1] for s in seeds], 0)
                tag = m + ("" if md == "two" else " cond-only")
                addB(tag, C)
                if md == "two":
                    addA(tag, P * C)
                    cls.setdefault(m, []).append(P)
        cls_y.append(p); cls_blk.append(np.full(len(p), o))
        if with_node:
            s0 = d.snaps[o]
            truth = d.true_sri_next[o]
            active = sorted(set(truth) | {v for g in s0.groups for v in g})
            yt = np.array([truth.get(v, 0.0) for v in active])
            G = d.graphs[o]
            def node_scores(vals):
                sc = {}
                for gi, g in enumerate(s0.groups):
                    for v in g:
                        sc[v] = sc.get(v, 0.0) + float(vals[gi])
                return np.array([sc.get(v, 0.0) for v in active])
            cand = {"persistence (SRI at k)": node_scores(s0.T),
                    "degree": np.array([G.degree(v) if G.has_node(v) else 0 for v in active]),
                    "group count": node_scores(np.ones(len(s0.groups))),
                    "persistence x hazard": node_scores(T * tab["p_hazard"]),
                    "GBDT hurdle": node_scores(tab["gbdt"] * tab["p_gbdt"])}
            for m in models:
                P = np.mean([nn_out[(m, "two", s, o)][0] for s in seeds], 0)
                C = np.mean([nn_out[(m, "two", s, o)][1] for s in seeds], 0)
                cand[m] = node_scores(P * C)
            for nm, sc in cand.items():
                r = stats.spearmanr(yt, sc)[0] if np.std(sc) > 0 and np.std(yt) > 0 else float("nan")
                node.setdefault(nm, []).append(float(r))
    resB = compare(perB, "persistence", orgs)
    resA = compare(perA, "persistence x base", orgs)
    # decomposition of Protocol A squared error into survivor / dissolved parts
    for nm, lst in perA.items():
        sv = sum(float(((yh - yA) ** 2)[yA_mask].sum()) for (yA, yh), yA_mask in zip(lst, [c > 0.5 for c in cls_y]))
        tot = sum(float(((yh - yA) ** 2).sum()) for (yA, yh) in lst)
        N = sum(len(yA) for yA, _ in lst)
        decomp[nm] = dict(survivor_part=sv / N, dissolved_part=(tot - sv) / N)
    yall, blk = np.concatenate(cls_y), np.concatenate(cls_blk)
    clsres = {}
    for nm, lst in cls.items():
        pp = np.concatenate(lst)
        lo, hi = auroc_ci(yall, pp, blk)
        clsres[nm] = dict(auroc=float(roc_auc_score(yall, pp)) if len(np.unique(yall)) == 2 else float("nan"), auroc_ci=[lo, hi], brier=float(brier_score_loss(yall, pp)),
                          ece=ece(yall, pp))
    out = dict(origins=orgs, n_B=int(sum(len(a) for a, _ in perB["persistence"])), n_A=int(len(yall)),
               survival_rate=float(yall.mean()), protocol_B=resB, protocol_A=resA, decomposition_A=decomp,
               classifier=clsres)
    if "MLP-res" in models and "GCN-res" in models:
        eq = [pair_equivalence(perB, "GCN-res", "MLP-res"), pair_equivalence(perB, "SAGE-res", "MLP-res")]
        if "cond" in modes:
            eq += [pair_equivalence(perB, "MLP-res", "MLP-res cond-only"), pair_equivalence(perB, "GCN-res", "GCN-res cond-only")]
        eq += [pair_equivalence(perB, "MLP-res", "persistence"), pair_equivalence(perB, "GCN+GRU-res", "GCN-res")]
        out["equivalence"] = eq
    if with_node:
        nres = {}
        for nm, v in node.items():
            v = np.array(v)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                nres[nm] = dict(mean_rho=float("nan"), ci95=[float("nan"), float("nan")], n_origins=0)
                continue
            rng = np.random.default_rng(0)
            n = len(v)
            bs = []
            for _ in range(4000):
                starts = rng.integers(0, n, int(math.ceil(n / 3)))
                idx = np.concatenate([(np.arange(s, s + 3) % n) for s in starts])[:n]
                bs.append(v[idx].mean())
            nres[nm] = dict(mean_rho=float(v.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n_origins=n)
        out["node_level"] = nres
    return out


def chrono_eval(workers, seeds=(42, 123, 456, 789, 999)):
    """Original chronological split (train 0-29, val 30-35, test 36-42) with the retrained models."""
    d = get_data("clique", "raw")
    tr = [k for k in d.valid if k <= 29]
    va = [k for k in d.valid if 30 <= k <= 35]
    te = [k for k in d.valid if k >= 36]
    combos = [(m, "two") for m in MODEL_SPECS] + [(m, "cond") for m in MODEL_SPECS] + [("GCN-plain", "zero"), ("MLP-res", "zero")]
    jobs = [("clique", "raw", m, md, s, tr, va, te, HP[m][0], HP[m][1], False) for (m, md) in combos for s in seeds]
    res = pmap(jobs, workers)
    preds = {}
    for (key, out, curve, fit) in res:
        _, _, m, md, s, _ = key
        preds.setdefault((m, md), []).append(out)
    yB, yA, TB, TA, blkB, blkA = [], [], [], [], [], []
    for k in te:
        sn = d.snaps[k]
        msk = sn.p == 1
        yB.append(sn.c[msk]); TB.append(sn.T[msk]); blkB.append(np.full(msk.sum(), k))
        yA.append(sn.p * sn.c); TA.append(sn.T); blkA.append(np.full(len(sn.p), k))
    yB, yA, TB, TA = map(np.concatenate, (yB, yA, TB, TA))
    out = {"test_snapshots": te, "n_B": int(len(yB)), "n_A": int(len(yA)),
           "persistence_B": float(np.mean((TB - yB) ** 2)),
           "persistence_x_base_A": float(np.mean((TA * np.concatenate([d.snaps[k].p for k in tr]).mean() - yA) ** 2)),
           "models": {}}
    for (m, md), lst in preds.items():
        mB, mA = [], []
        for o in lst:
            cB = np.concatenate([o[k][1][d.snaps[k].p == 1] for k in te])
            mB.append(float(np.mean((cB - yB) ** 2)))
            if md == "two":
                eA = np.concatenate([o[k][0] * o[k][1] for k in te])
            elif md == "zero":
                eA = np.concatenate([o[k][1] for k in te])
            else:
                eA = None
            if eA is not None:
                mA.append(float(np.mean((eA - yA) ** 2)))
        out["models"][f"{m} {md}"] = dict(mse_B_mean=float(np.mean(mB)), mse_B_std=float(np.std(mB)),
                                          mse_A_mean=float(np.mean(mA)) if mA else None,
                                          mse_A_std=float(np.std(mA)) if mA else None)
    return out


def construction_stats():
    out = {}
    for c in CONSTRUCTIONS:
        d = get_data(c, "raw")
        p = np.concatenate([d.snaps[k].p for k in d.valid])
        sizes = [len(g) for s in d.snaps for g in s.groups]
        out[c] = dict(n_groups=int(sum(len(s.groups) for s in d.snaps)), mean_size=float(np.mean(sizes)) if sizes else 0,
                      survival=float(p.mean()), n_origins=len(origins_for(d)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="all", help="all | chrono")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for c in CONSTRUCTIONS:
        for tm in ("raw", "norm"):
            get_data(c, tm)          # build before forking
    pmap([], args.workers)
    if args.only == "chrono":
        R = json.load(open(OUT_DIR / "results.json"))
        for m, v in R["tuning"].items():
            HP[m] = (v["hidden"], v["lr"])
        R["chrono_eval"] = chrono_eval(args.workers)
        json.dump(R, open(OUT_DIR / "results.json", "w"), indent=1)
        print(f"chrono done {time.time() - t0:.0f}s")
        return
    R = {"constructions": construction_stats()}
    print(f"data ready {time.time() - t0:.0f}s", flush=True)
    R["tuning"] = tune(args.workers)
    print("tuning", R["tuning"], f"{time.time() - t0:.0f}s", flush=True)
    R["chrono_fit"] = chrono_fit(args.workers)
    print(f"curves {time.time() - t0:.0f}s", flush=True)
    models = list(MODEL_SPECS)
    main_seeds = [42, 123, 456] if args.quick else [42, 123, 456, 789, 999]
    R["main_raw"] = rolling("clique", "raw", models, ["two", "cond"], main_seeds, args.workers, with_node=True)
    json.dump(R, open(OUT_DIR / "results.json", "w"), indent=1)
    print(f"main raw {time.time() - t0:.0f}s", flush=True)
    sweep_models = ["MLP-res", "GCN-res", "SAGE-res", "GCN+GRU-res"]
    sweep_seeds = [42, 123] if args.quick else [42, 123, 456]
    R["sweep"] = {}
    for c in CONSTRUCTIONS:
        for tm in ("raw", "norm"):
            if c == "clique" and tm == "raw":
                continue
            R["sweep"][f"{c}/{tm}"] = rolling(c, tm, sweep_models, ["two"], sweep_seeds, args.workers)
            json.dump(R, open(OUT_DIR / "results.json", "w"), indent=1)
            print(f"sweep {c}/{tm} {time.time() - t0:.0f}s", flush=True)
    json.dump(R, open(OUT_DIR / "results.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
