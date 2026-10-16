#!/usr/bin/env python3
"""Dataset generator v1 for the graph-coloring heuristic project.

    python generate_dataset.py --preset sample --out data_sample
    python generate_dataset.py --preset full   --out data_v1 --workers 4

v1 is a SUPERSET of v0: all 28 v0 families keep the same ids/params, so their graphs
(and sha256) are unchanged for seeds 0-19 (train), 1000-1019 (val), 2000-2049 (test).

Pipeline
  A. label every candidate family on its PROBE graphs (seeds 9000-9009, disjoint from all
     splits): measured features -> cards, DSATUR/portfolio/clique-bound -> difficulty & headroom.
  B. drop families where DSATUR is provably (near-)optimal, assign pools
     (train_pool / heldout_type / heldout_size / heldout_combo), shuffle partners.
  C. emit graphs per split + manifest, run integrity checks, write report.txt.
Do NOT change SPLITS, family specs, seeds or thresholds once results exist.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import hashlib
import json
import math
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

VERSION = "v1"

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SPLITS = {"train": (0, 100), "val": (1000, 20), "test": (2000, 50), "probe": (9000, 10)}
SAMPLE_COUNTS = {"train": 5, "val": 2, "test": 3, "probe": 10}
BASELINE_TRAIN_SEEDS = 20          # per-family baselines (CMA-ES, EoH/ReEvo) use seeds 0..19 only

SIZES = {"S": (50, 100), "M": (100, 200), "L": (200, 300), "XL": (320, 500)}
HELDOUT_TYPES = {"scale_free", "geometric", "hybrid_geometric", "queen"}   # axis 1: structure type
HELDOUT_SIZE_MIN = 320                                                    # axis 2: n beyond training (<=300)
COMBO_HOLDOUT_FRACTION = 0.12   # axis 3: share of train-pool families removed by feature-combo
COMBO_SEED = 2026
DENS_EDGES = [0.05, 0.10, 0.20, 0.40]
NO_HEADROOM_GAP = 0.25          # drop family if mean(DSATUR) - mean(clique lower bound) <= this
RNG_RESTARTS_LB = 40

STRUCT_TEXT = {
    "gnp": "random (Erdos-Renyi)", "geometric": "random geometric",
    "hybrid_geometric": "geometric plus random edges", "scale_free": "scale-free",
    "planted_partition": "planted partition communities", "small_world": "small-world",
    "caveman": "clustered cliques", "interval": "interval",
    "queen": "queen-grid-like", "mycielski": "Mycielski construction",
    "equipartite": "hidden k-partite (equipartite)",
}

# --- v0 families, verbatim (keeps v0 graphs identical) -----------------------
V0_FAMILIES = [
    ("gnp_d0.05_n200-300", "gnp", "gnp", {"p": 0.05}, (200, 300)),
    ("gnp_d0.10_n100-200", "gnp", "gnp", {"p": 0.10}, (100, 200)),
    ("gnp_d0.30_n100-200", "gnp", "gnp", {"p": 0.30}, (100, 200)),
    ("gnp_d0.50_n50-100", "gnp", "gnp", {"p": 0.50}, (50, 100)),
    ("geo_r0.10_n200-300", "geometric", "geometric", {"radius": 0.10}, (200, 300)),
    ("geo_r0.15_n100-200", "geometric", "geometric", {"radius": 0.15}, (100, 200)),
    ("geo_r0.25_n100-200", "geometric", "geometric", {"radius": 0.25}, (100, 200)),
    ("hyb_r0.15_q0.02_n100-200", "hybrid_geometric", "geometric", {"radius": 0.15, "noise": 0.02}, (100, 200)),
    ("hyb_r0.20_q0.05_n100-200", "hybrid_geometric", "geometric", {"radius": 0.20, "noise": 0.05}, (100, 200)),
    ("ba_m3_n100-300", "scale_free", "ba", {"m": 3}, (100, 300)),
    ("ba_m8_n100-300", "scale_free", "ba", {"m": 8}, (100, 300)),
    ("plc_m5_p0.5_n100-200", "scale_free", "plc", {"m": 5, "p": 0.5}, (100, 200)),
    ("sbm_k4_in0.5_out0.05_n100-200", "planted_partition", "sbm", {"k": 4, "p_in": 0.5, "p_out": 0.05}, (100, 200)),
    ("sbm_k8_in0.6_out0.10_n150-250", "planted_partition", "sbm", {"k": 8, "p_in": 0.6, "p_out": 0.10}, (150, 250)),
    ("sbm_k5_in0.3_out0.15_n100-200", "planted_partition", "sbm", {"k": 5, "p_in": 0.3, "p_out": 0.15}, (100, 200)),
    ("ws_k10_p0.1_n100-200", "small_world", "ws", {"k": 10, "p": 0.1}, (100, 200)),
    ("ws_k20_p0.3_n100-200", "small_world", "ws", {"k": 20, "p": 0.3}, (100, 200)),
    ("cave_k10_p0.1_n100-200", "caveman", "caveman", {"k": 10, "p": 0.1}, (100, 200)),
    ("cave_k20_p0.2_n100-200", "caveman", "caveman", {"k": 20, "p": 0.2}, (100, 200)),
    ("interval_L0.10_n100-200", "interval", "interval", {"L": 0.10}, (100, 200)),
    ("interval_L0.20_n100-200", "interval", "interval", {"L": 0.20}, (100, 200)),
    ("queen_n49-121", "queen", "queen", {}, (49, 121)),
    ("queen_n100-200", "queen", "queen", {}, (100, 200)),
    ("myciel_t1_p0.2_n100-200", "mycielski", "mycielski", {"t": 1, "p": 0.2}, (100, 200)),
    ("myciel_t2_p0.1_n100-200", "mycielski", "mycielski", {"t": 2, "p": 0.1}, (100, 200)),
    ("kpart_k5_p0.5_n100-200", "equipartite", "kpartite", {"k": 5, "p": 0.5}, (100, 200)),
    ("kpart_k10_p0.3_n100-200", "equipartite", "kpartite", {"k": 10, "p": 0.3}, (100, 200)),
    ("kpart_k20_p0.5_n150-250", "equipartite", "kpartite", {"k": 20, "p": 0.5}, (150, 250)),
]


def _grid():
    """New v1 families from parameter grids. Each entry: (..., size_key)."""
    F = []

    def add(fid, stype, gen, params, sz):
        F.append((fid, stype, gen, params, SIZES[sz]))

    def nm(sz):
        lo, hi = SIZES[sz]
        return f"n{lo}-{hi}"

    for p, s in [(0.05, "M"), (0.05, "L"), (0.10, "S"), (0.10, "M"), (0.10, "L"), (0.15, "M"), (0.20, "M"),
                 (0.20, "L"), (0.30, "S"), (0.30, "M"), (0.30, "L"), (0.40, "M"), (0.50, "S"), (0.50, "M"),
                 (0.70, "S"), (0.05, "XL"), (0.10, "XL"), (0.30, "XL")]:
        add(f"gnp_d{p:.2f}_{nm(s)}", "gnp", "gnp", {"p": p}, s)
    for k, pi, po, s in [(3, .5, .05, "M"), (4, .5, .05, "M"), (4, .7, .1, "S"), (6, .4, .05, "M"), (6, .6, .1, "L"),
                         (8, .6, .1, "L"), (8, .3, .1, "M"), (5, .3, .15, "M"), (10, .5, .05, "L"), (10, .7, .2, "L"),
                         (12, .5, .1, "L"), (6, .8, .05, "S"), (4, .5, .05, "XL"), (8, .6, .1, "XL"), (5, .3, .15, "XL")]:
        add(f"sbm_k{k}_in{pi:.1f}_out{po:.2f}_{nm(s)}", "planted_partition", "sbm",
            {"k": k, "p_in": pi, "p_out": po}, s)
    for k, p, s in [(6, .05, "M"), (10, .1, "M"), (10, .3, "S"), (16, .1, "L"), (20, .3, "M"), (30, .2, "L"),
                    (8, .5, "S"), (10, .1, "XL"), (20, .3, "XL")]:
        add(f"ws_k{k}_p{p:g}_{nm(s)}", "small_world", "ws", {"k": k, "p": p}, s)
    for k, p, s in [(5, .05, "M"), (8, .1, "M"), (10, .1, "L"), (10, .3, "M"), (15, .1, "M"), (20, .2, "L"),
                    (30, .1, "L"), (10, .1, "XL"), (20, .2, "XL")]:
        add(f"cave_k{k}_p{p:g}_{nm(s)}", "caveman", "caveman", {"k": k, "p": p}, s)
    for L, s in [(0.03, "M"), (0.05, "L"), (0.10, "M"), (0.15, "S"), (0.20, "S"), (0.05, "XL")]:
        add(f"interval_L{L:.2f}_{nm(s)}", "interval", "interval", {"L": L}, s)
    for t, p, s in [(1, .1, "M"), (1, .2, "M"), (1, .3, "S"), (2, .05, "M"), (2, .1, "M"), (2, .2, "L"), (3, .05, "M"),
                    (3, .1, "L"), (2, .1, "XL"), (3, .05, "XL")]:
        add(f"myciel_t{t}_p{p:g}_{nm(s)}", "mycielski", "mycielski", {"t": t, "p": p}, s)
    for k, p, s in [(4, .5, "M"), (5, .5, "M"), (8, .3, "M"), (10, .3, "M"), (10, .6, "L"), (15, .4, "L"), (20, .5, "L"),
                    (30, .5, "L"), (12, .15, "M"), (20, .2, "L"), (10, .3, "XL"), (20, .2, "XL")]:
        add(f"kpart_k{k}_p{p:g}_{nm(s)}", "equipartite", "kpartite", {"k": k, "p": p}, s)
    # held-out structure types
    for r, s in [(0.08, "L"), (0.10, "L"), (0.12, "M"), (0.15, "M"), (0.20, "M"), (0.25, "S"), (0.35, "S")]:
        add(f"geo_r{r:.2f}_{nm(s)}", "geometric", "geometric", {"radius": r}, s)
    for r, q, s in [(0.10, 0.01, "M"), (0.15, 0.05, "M"), (0.25, 0.10, "S"), (0.12, 0.03, "L")]:
        add(f"hyb_r{r:.2f}_q{q:.2f}_{nm(s)}", "hybrid_geometric", "geometric", {"radius": r, "noise": q}, s)
    for m, s in [(2, "M"), (3, "L"), (5, "M"), (8, "L"), (12, "L")]:
        add(f"ba_m{m}_{nm(s)}", "scale_free", "ba", {"m": m}, s)
    for m, p, s in [(3, .3, "M"), (8, .7, "L"), (5, .9, "M")]:
        add(f"plc_m{m}_p{p:g}_{nm(s)}", "scale_free", "plc", {"m": m, "p": p}, s)
    for keep, s in [(1.0, "S"), (1.0, "M"), (1.0, "L"), (0.7, "M"), (0.7, "L"), (0.4, "M"), (0.4, "L")]:
        fid = f"queen_{nm(s)}" if keep == 1.0 else f"queen_keep{keep:g}_{nm(s)}"
        add(fid, "queen", "queen", {} if keep == 1.0 else {"keep": keep}, s)
    return F


def all_families():
    seen, out = set(), []
    for f in V0_FAMILIES + _grid():
        if f[0] not in seen:
            seen.add(f[0])
            out.append(f)
    return out


# ----------------------------------------------------------------------------
# Generators: gen(n, rng, **params) -> (n_actual, edges[m,2] int32, i<j)
# ----------------------------------------------------------------------------
def _from_upper(A):
    return np.argwhere(np.triu(A, 1)).astype(np.int32)


def _from_nx(G):
    import networkx as nx
    G = nx.convert_node_labels_to_integers(G)
    e = np.array([(min(u, v), max(u, v)) for u, v in G.edges() if u != v], dtype=np.int32)
    e = np.unique(e, axis=0) if len(e) else np.zeros((0, 2), dtype=np.int32)
    return G.number_of_nodes(), e


def gen_gnp(n, rng, p):
    return n, _from_upper(rng.random((n, n)) < p)


def gen_geometric(n, rng, radius, noise=0.0):
    pts = rng.random((n, 2))
    A = np.linalg.norm(pts[:, None] - pts[None], axis=-1) < radius
    if noise > 0:
        A |= rng.random((n, n)) < noise
    return n, _from_upper(A)


def gen_ba(n, rng, m):
    import networkx as nx
    return _from_nx(nx.barabasi_albert_graph(n, m, seed=int(rng.integers(2**31))))


def gen_plc(n, rng, m, p):
    import networkx as nx
    return _from_nx(nx.powerlaw_cluster_graph(n, m, p, seed=int(rng.integers(2**31))))


def gen_ws(n, rng, k, p):
    import networkx as nx
    return _from_nx(nx.watts_strogatz_graph(n, k, p, seed=int(rng.integers(2**31))))


def gen_caveman(n, rng, k, p):
    import networkx as nx
    return _from_nx(nx.relaxed_caveman_graph(max(2, n // k), k, p, seed=int(rng.integers(2**31))))


def _labels(n, k, rng):
    return rng.permutation(np.arange(n) % k)


def gen_sbm(n, rng, k, p_in, p_out):
    lab = _labels(n, k, rng)
    P = np.where(lab[:, None] == lab[None], p_in, p_out)
    return n, _from_upper(rng.random((n, n)) < P)


def gen_kpartite(n, rng, k, p):
    """Hidden k-colorable graph (edges only between classes), so chi <= k."""
    lab = _labels(n, k, rng)
    P = np.where(lab[:, None] == lab[None], 0.0, p)
    return n, _from_upper(rng.random((n, n)) < P)


def gen_interval(n, rng, L):
    s = rng.random(n)
    e = s + rng.uniform(0, 2 * L, n)
    return n, _from_upper((s[:, None] < e[None]) & (s[None] < e[:, None]))


def gen_queen(n, rng, keep=1.0):
    """Queen graph on a k x k board (k = ceil(sqrt(n/keep))) keeping n random squares."""
    k = math.ceil(math.sqrt(n / keep))
    sel = np.sort(rng.choice(k * k, n, replace=False))
    r, c = sel // k, sel % k
    dr, dc = np.abs(r[:, None] - r[None]), np.abs(c[:, None] - c[None])
    return n, _from_upper((dr == 0) | (dc == 0) | (dr == dc))


def gen_mycielski(n, rng, t, p):
    import networkx as nx
    n0 = max(3, round((n + 1) / 2**t) - 1)
    G = nx.Graph()
    G.add_nodes_from(range(n0))
    G.add_edges_from(map(tuple, _from_upper(rng.random((n0, n0)) < p)))
    return _from_nx(nx.mycielskian(G, iterations=t))


GENS = {"gnp": gen_gnp, "geometric": gen_geometric, "ba": gen_ba, "plc": gen_plc, "ws": gen_ws,
        "caveman": gen_caveman, "sbm": gen_sbm, "kpartite": gen_kpartite, "interval": gen_interval,
        "queen": gen_queen, "mycielski": gen_mycielski}


def make_graph(fam, seed):
    """Deterministic: a graph is a pure function of (family_id, seed)."""
    fid, _, gen, params, (lo, hi) = fam
    rng = np.random.default_rng([zlib.crc32(fid.encode()), int(seed)])
    n = int(rng.integers(lo, hi + 1))
    return GENS[gen](n, rng, **params)


# ----------------------------------------------------------------------------
# Provisional reference solvers (numpy only). The project's own verified evaluator
# replaces these; they exist to label difficulty/headroom and for sanity checks.
# ----------------------------------------------------------------------------
def adj_from_edges(n, edges):
    A = np.zeros((n, n), dtype=bool)
    if len(edges):
        A[edges[:, 0], edges[:, 1]] = True
        A[edges[:, 1], edges[:, 0]] = True
    return A


def is_proper(A, colors):
    return bool((colors >= 0).all() and not (A & (colors[:, None] == colors[None])).any())


def ncolors(colors):
    return int(colors.max()) + 1 if len(colors) else 0


def dsatur(A, rng=None):
    """Classic DSATUR: max saturation, then max uncolored degree, then lowest index
    (or uniformly random among ties if rng is given)."""
    n = A.shape[0]
    deg = A.sum(1).astype(np.int64)
    K = int(deg.max()) + 2 if n else 1
    nb = np.zeros((n, K), dtype=bool)
    sat = np.zeros(n, dtype=np.int64)
    udeg = deg.copy()
    unc = np.ones(n, dtype=bool)
    col = -np.ones(n, dtype=np.int64)
    for _ in range(n):
        key = np.where(unc, sat * (n + 1) + udeg, -1)
        cand = np.flatnonzero(key == key.max())
        v = int(cand[0]) if rng is None else int(rng.choice(cand))
        c = int(np.argmin(nb[v]))
        col[v] = c
        unc[v] = False
        nbrs = np.flatnonzero(A[v])
        new = ~nb[nbrs, c]
        sat[nbrs[new]] += 1
        nb[nbrs, c] = True
        udeg[nbrs] -= 1
    return col


def greedy_in_order(A, order):
    n = A.shape[0]
    K = int(A.sum(1).max()) + 2 if n else 1
    col = -np.ones(n, dtype=np.int64)
    for v in order:
        used = np.zeros(K, dtype=bool)
        cn = col[A[v]]
        used[cn[cn >= 0]] = True
        col[v] = int(np.argmin(used))
    return col


def smallest_last_order(A):
    n = A.shape[0]
    d = A.sum(1).astype(np.int64)
    removed = np.zeros(n, dtype=bool)
    order = []
    for _ in range(n):
        v = int(np.argmin(np.where(removed, 1 << 30, d)))
        order.append(v)
        removed[v] = True
        d[A[v]] -= 1
    return order[::-1]


def rlf(A):
    n = A.shape[0]
    col = -np.ones(n, dtype=np.int64)
    unc = np.ones(n, dtype=bool)
    c = 0
    while unc.any():
        du = (A & unc[None]).sum(1)
        v = int(np.argmax(np.where(unc, du, -1)))
        col[v] = c
        unc[v] = False
        cand = unc & ~A[v]
        excl = unc & A[v]
        while cand.any():
            ne = (A & excl[None]).sum(1)
            nc = (A & cand[None]).sum(1)
            score = np.where(cand, ne * (A.shape[0] + 1) - nc, -(1 << 30))
            u = int(np.argmax(score))
            col[u] = c
            unc[u] = False
            cand[u] = False
            newly = cand & A[u]
            cand &= ~A[u]
            excl |= newly
            excl[u] = False
        c += 1
    return col


def clique_lower_bound(A, rng, restarts=RNG_RESTARTS_LB):
    n = A.shape[0]
    best = 0
    for r in range(restarts):
        cand = np.ones(n, dtype=bool)
        size = 0
        while cand.any():
            idx = np.flatnonzero(cand)
            d = A[np.ix_(idx, idx)].sum(1).astype(float)
            d = d + (rng.random(len(idx)) * 2.0 if r else rng.random(len(idx)) * 1e-3)
            v = idx[int(np.argmax(d))]
            size += 1
            cand &= A[v]
        best = max(best, size)
    return best


def features(n, edges):
    A = np.zeros((n, n), dtype=np.float32)
    if len(edges):
        A[edges[:, 0], edges[:, 1]] = 1
        A[edges[:, 1], edges[:, 0]] = 1
    deg = A.sum(1)
    mean = deg.mean()
    tri = ((A @ A) * A).sum(1) / 2
    denom = deg * (deg - 1) / 2
    clust = np.where(denom > 0, tri / np.maximum(denom, 1), 0.0).mean()
    return {"n": int(n), "m": int(len(edges)), "density": float(len(edges) / (n * (n - 1) / 2)),
            "deg_mean": float(mean), "deg_max": int(deg.max()),
            "deg_cv": float(deg.std() / mean) if mean > 0 else 0.0, "clustering": float(clust)}


def sha(n, edges):
    return hashlib.sha256(np.int32(n).tobytes() + edges.astype(np.int32).tobytes()).hexdigest()


# ----------------------------------------------------------------------------
# Cards (all deterministic, template-based; no LLM)
# ----------------------------------------------------------------------------
DV = {"low": ["low degree variance", "nearly regular degrees", "narrow spread of vertex degrees"],
      "medium": ["medium degree variance", "moderately uneven degrees", "moderate spread of vertex degrees"],
      "high": ["high degree variance", "very uneven, heavy-tailed degrees", "a few hubs with much higher degree"]}
CL = {"low": ["low clustering", "clustering close to a random graph", "few closed triples"],
      "moderate": ["moderate clustering", "some local clustering", "mild excess of triangles"],
      "high": ["high clustering", "strongly clustered neighborhoods", "many more triangles than a random graph"]}


def buckets(stats):
    cv = stats["deg_cv"]
    ratio = stats["clustering"] / max(stats["density"], 1e-9)   # random graph: ratio ~ 1
    dv = "low" if cv < 0.2 else "medium" if cv < 0.5 else "high"
    cl = "low" if ratio < 1.5 else "moderate" if ratio < 4 else "high"
    db = int(np.searchsorted(DENS_EDGES, stats["density"]))
    return dv, cl, db


def make_cards(stype, nr, stats):
    dv, cl, db = buckets(stats)
    d, lo, hi = stats["density"], nr[0], nr[1]
    S = STRUCT_TEXT[stype]
    cards = {
        "card": f"{S}, nodes {lo}-{hi}, density {d:.2f}, {DV[dv][0]}, {CL[cl][0]}",
        "card_no_structure": f"nodes {lo}-{hi}, density {d:.2f}, {DV[dv][0]}, {CL[cl][0]}",
        "card_paraphrase_1": f"{S}; between {lo} and {hi} vertices; edge density about {d:.2f}; {DV[dv][1]}; {CL[cl][1]}",
        "card_paraphrase_2": f"Graphs of type {S}, with {lo} to {hi} nodes, {d * 100:.0f}% of possible edges present, {DV[dv][2]} and {CL[cl][2]}.",
        "card_paraphrase_3": f"Size: {lo}-{hi} nodes. Density: {d:.2f}. Structure: {S}. Degrees: {DV[dv][0].replace(' degree variance', '')} variance. Clustering: {CL[cl][0].replace(' clustering', '')}.",
        "card_partial_size_density": f"nodes {lo}-{hi}, density {d:.2f}",
        "card_partial_structure_clustering": f"{S}, {CL[cl][0]}",
        "card_partial_density_degvar": f"density {d:.2f}, {DV[dv][0]}",
        "card_numeric": f"{S}, n in [{lo}, {hi}], density {d:.3f}, degree CV {stats['deg_cv']:.2f}, clustering/density {stats['clustering'] / max(d, 1e-9):.1f}",
    }
    return cards


# ----------------------------------------------------------------------------
# Stage A: label a family on its probe graphs
# ----------------------------------------------------------------------------
def label_family(fam):
    fid, stype, gen, params, nr = fam
    off, cnt = SPLITS["probe"]
    rng = np.random.default_rng([zlib.crc32(fid.encode()), 777])
    feats, ds, port, lb, rl = [], [], [], [], []
    for i in range(cnt):
        n, e = make_graph(fam, off + i)
        A = adj_from_edges(n, e)
        feats.append(features(n, e))
        c0 = dsatur(A)
        assert is_proper(A, c0)
        cands = [ncolors(c0)]
        for _ in range(10):
            cr = dsatur(A, rng)
            assert is_proper(A, cr)
            cands.append(ncolors(cr))
        deg = A.sum(1)
        for order in (np.argsort(-deg, kind="stable"), smallest_last_order(A)):
            cc = greedy_in_order(A, order)
            assert is_proper(A, cc)
            cands.append(ncolors(cc))
        cr_ = rlf(A)
        assert is_proper(A, cr_)
        cands.append(ncolors(cr_))
        ds.append(ncolors(c0))
        rl.append(ncolors(cr_))
        port.append(min(cands))
        lb.append(clique_lower_bound(A, rng))
    stats = {k: float(np.mean([f[k] for f in feats])) for k in ("density", "deg_cv", "clustering", "n")}
    ub = params.get("k") if gen == "kpartite" else None
    rec = {
        "family_id": fid, "structure_type": stype, "generator": gen, "params": params,
        "n_range": list(nr), "probe_stats": stats, "cards": make_cards(stype, nr, stats),
        "labels": {
            "dsatur_mean": float(np.mean(ds)), "rlf_mean": float(np.mean(rl)),
            "portfolio_best_mean": float(np.mean(port)), "clique_lb_mean": float(np.mean(lb)),
            "planted_k_upper": ub,
            "gap_dsatur_to_lb": float(np.mean(ds) - np.mean(lb)),
            "headroom_proxy_vs_dsatur": float(np.mean(ds) - np.mean(port)),
            "difficulty_rel": float((np.mean(ds) - np.mean(lb)) / max(np.mean(lb), 1)),
        },
        "tier": 1 if nr[1] <= 150 else 2 if nr[1] <= 300 else 3,
    }
    return rec


# ----------------------------------------------------------------------------
# Stage B: drop no-headroom families, assign pools, shuffle partners
# ----------------------------------------------------------------------------
def assign_pools(recs):
    keep, dropped = [], []
    for r in recs:
        lab = r["labels"]
        if lab["gap_dsatur_to_lb"] <= NO_HEADROOM_GAP:
            r["drop_reason"] = (f"DSATUR mean {lab['dsatur_mean']:.2f} within {NO_HEADROOM_GAP} of "
                                f"clique lower bound {lab['clique_lb_mean']:.2f}: provably (near-)optimal")
            dropped.append(r)
        else:
            keep.append(r)
    for r in keep:
        dv, cl, db = buckets(r["probe_stats"])
        r["combo"] = [db, dv, cl]
        r["pool"] = ("heldout_type" if r["structure_type"] in HELDOUT_TYPES
                     else "heldout_size" if r["n_range"][0] >= HELDOUT_SIZE_MIN else "train_pool")
    # axis 3: hold out whole feature combos (density bin, degvar, clustering) among train-pool families
    tp = [r for r in keep if r["pool"] == "train_pool"]
    groups = {}
    for r in tp:
        groups.setdefault(tuple(r["combo"]), []).append(r)
    cands = sorted(k for k, v in groups.items() if len(v) >= 2)
    np.random.default_rng(COMBO_SEED).shuffle(cands)
    target = int(round(COMBO_HOLDOUT_FRACTION * len(tp)))
    held, per_type = 0, {}
    for r in tp:
        per_type[r["structure_type"]] = per_type.get(r["structure_type"], 0) + 1
    for k in cands:
        if held >= target:
            break
        mem = groups[k]
        if any(per_type[m["structure_type"]] - sum(1 for x in mem if x["structure_type"] == m["structure_type"]) < 2
               for m in mem):
            continue                                  # keep >= 2 train families per structure type
        for m in mem:
            m["pool"] = "heldout_combo"
            per_type[m["structure_type"]] -= 1
        held += len(mem)
    # shuffle partners (same pool): far = most different features, near = closest other family
    for pool in {r["pool"] for r in keep}:
        P = [r for r in keep if r["pool"] == pool]
        if len(P) < 2:
            continue
        X = np.array([[math.log(r["probe_stats"]["n"]), math.log10(max(r["probe_stats"]["density"], 1e-4)),
                       r["probe_stats"]["deg_cv"],
                       math.log(max(r["probe_stats"]["clustering"] / max(r["probe_stats"]["density"], 1e-9), 1e-3))]
                      for r in P])
        X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
        D = np.linalg.norm(X[:, None] - X[None], axis=-1)
        for i, r in enumerate(P):
            r["shuffle_partner_far"] = P[int(np.argmax(D[i]))]["family_id"]
            Dn = D[i].copy()
            Dn[i] = np.inf
            r["shuffle_partner_near"] = P[int(np.argmin(Dn))]["family_id"]
    return keep, dropped


# ----------------------------------------------------------------------------
# Stage C: emit graphs + manifest rows for one family
# ----------------------------------------------------------------------------
def emit_family(args):
    fam, counts, out, with_graphs, with_dsatur, verify = args
    fid = fam[0]
    rows, store = [], {}
    for split in ("train", "val", "test"):
        off, _ = SPLITS[split]
        store[split] = {}
        for i in range(counts[split]):
            seed = off + i
            n, e = make_graph(fam, seed)
            f = features(n, e)
            row = {"family_id": fid, "split": split, "seed": seed,
                   "file": f"graphs/{fid}/{split}.npz", "key": f"s{seed}", "sha256": sha(n, e), **f,
                   "baseline_train": bool(split == "train" and i < BASELINE_TRAIN_SEEDS)}
            if with_dsatur and (split != "train" or i < BASELINE_TRAIN_SEEDS):
                A = adj_from_edges(n, e)
                c = dsatur(A)
                assert is_proper(A, c)
                row["dsatur_colors_provisional"] = ncolors(c)
            if verify == "all" or (verify == "some" and seed % 10 == 0):
                n2, e2 = make_graph(fam, seed)
                assert sha(n2, e2) == row["sha256"], f"non-deterministic: {fid} {seed}"
                assert len(e) == 0 or ((e[:, 0] < e[:, 1]).all() and len(np.unique(e, axis=0)) == len(e))
            store[split][f"s{seed}"] = e
            rows.append(row)
    if with_graphs:
        d = Path(out) / "graphs" / fid
        d.mkdir(parents=True, exist_ok=True)
        for split, dct in store.items():
            np.savez_compressed(d / f"{split}.npz", **dct)
    return rows


# ----------------------------------------------------------------------------
# Report + checks
# ----------------------------------------------------------------------------
def build_report(keep, dropped, manifest, counts, preset, emit_ids):
    L, fails = [], []
    P = lambda s="": L.append(s)
    P(f"DATASET {VERSION} ({preset}): candidates={len(keep) + len(dropped)} kept={len(keep)} "
      f"dropped={len(dropped)} graphs={len(manifest)}")
    pools = {}
    for r in keep:
        pools.setdefault(r["pool"], []).append(r)
    P("\nPools: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(pools.items())))
    types = {}
    for r in keep:
        types.setdefault(r["structure_type"], {}).setdefault(r["pool"], 0)
        types[r["structure_type"]][r["pool"]] += 1
    P("\nFamilies per structure type and pool:")
    for t, d in sorted(types.items()):
        P(f"  {t:18s} " + ", ".join(f"{k}={v}" for k, v in sorted(d.items())))
    P("\nDropped (no headroom, DSATUR provably near-optimal):")
    for r in dropped:
        P(f"  {r['family_id']:34s} {r['drop_reason']}")
    hp = np.array([r["labels"]["headroom_proxy_vs_dsatur"] for r in keep])
    P(f"\nHeadroom proxy (mean DSATUR - best of cheap portfolio, colors) over kept families: "
      f"min={hp.min():.2f} median={np.median(hp):.2f} max={hp.max():.2f}; "
      f"families with proxy < 0.2: {(hp < 0.2).sum()}")
    nsd = {}
    for r in keep:
        nsd.setdefault(r["cards"]["card_no_structure"], []).append(r["family_id"])
    dup = [v for v in nsd.values() if len(v) > 1]
    fullc = {}
    for r in keep:
        fullc.setdefault(r["cards"]["card"], []).append(r["family_id"])
    dupf = [v for v in fullc.values() if len(v) > 1]
    P(f"\nCard uniqueness: full card {len(fullc)}/{len(keep)} unique; no-structure card {len(nsd)}/{len(keep)} unique")
    for v in dupf:
        P(f"  identical full card: {v}")
    for v in dup[:10]:
        if len({fid.split('_')[0] for fid in v}) > 1 or len(v) > 1:
            P(f"  identical no-structure card: {v}")
    # checks
    tp_types = {r["structure_type"] for r in pools.get("train_pool", [])}
    if tp_types & HELDOUT_TYPES:
        fails.append("heldout structure type leaked into train_pool")
    if any(r["n_range"][1] > 300 for r in pools.get("train_pool", [])):
        fails.append("train_pool contains n > 300")
    if any(r["n_range"][0] < HELDOUT_SIZE_MIN for r in pools.get("heldout_size", [])):
        fails.append("heldout_size contains small n")
    tc = {tuple(r["combo"]) for r in pools.get("train_pool", [])}
    if any(tuple(r["combo"]) in tc for r in pools.get("heldout_combo", [])):
        fails.append("heldout_combo shares a feature combo with train_pool")
    seeds = {k: set(range(o, o + c)) for k, (o, c) in SPLITS.items()}
    ks = list(seeds)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            if seeds[ks[i]] & seeds[ks[j]]:
                fails.append(f"seed overlap {ks[i]}/{ks[j]}")
    per = {}
    for m in manifest:
        per.setdefault((m["family_id"], m["split"]), 0)
        per[(m["family_id"], m["split"])] += 1
    if any(per.get((r["family_id"], s), 0) != counts[s] for r in keep if r["family_id"] in emit_ids for s in ("train", "val", "test")):
        fails.append("split counts differ from configuration")
    P("\nChecks: " + ("ALL PASSED" if not fails else "FAILED: " + "; ".join(fails)))
    P("\nPer-family table (probe stats): family | pool | n | dens | cv | clus/dens | DSATUR | LB | portfolio | headroom")
    for r in sorted(keep, key=lambda x: (x["pool"], x["structure_type"], x["family_id"])):
        s, lab = r["probe_stats"], r["labels"]
        P(f"  {r['family_id']:34s} {r['pool']:13s} {s['n']:5.0f} {s['density']:5.2f} {s['deg_cv']:5.2f} "
          f"{s['clustering'] / max(s['density'], 1e-9):6.1f} {lab['dsatur_mean']:6.1f} {lab['clique_lb_mean']:5.1f} "
          f"{lab['portfolio_best_mean']:6.1f} {lab['headroom_proxy_vs_dsatur']:5.2f}")
    return "\n".join(L), fails


def run(preset, out, workers, with_graphs, with_dsatur, verify):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fams = all_families()
    print(f"labeling {len(fams)} candidate families on probe graphs ...", flush=True)
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            recs = list(ex.map(label_family, fams, chunksize=2))
    else:
        recs = [label_family(f) for f in fams]
    keep, dropped = assign_pools(recs)
    fam_by_id = {f[0]: f for f in fams}
    counts = {k: v[1] for k, v in SPLITS.items()} if preset == "full" else SAMPLE_COUNTS
    emit = keep
    if preset == "sample":        # one family per (pool, structure type) to keep the preview small
        seen, emit = set(), []
        for r in keep:
            k = (r["pool"], r["structure_type"])
            if k not in seen:
                seen.add(k)
                emit.append(r)
    print(f"emitting graphs for {len(emit)} families ...", flush=True)
    jobs = [(fam_by_id[r["family_id"]], counts, str(out), with_graphs, with_dsatur, verify) for r in emit]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(emit_family, jobs, chunksize=1))
    else:
        res = [emit_family(j) for j in jobs]
    manifest = [row for rows in res for row in rows]
    pool_of = {r["family_id"]: r["pool"] for r in keep}
    for m in manifest:
        m["pool"] = pool_of[m["family_id"]]
    for r in emit:
        d = [m["dsatur_colors_provisional"] for m in manifest
             if m["family_id"] == r["family_id"] and m["split"] == "test" and "dsatur_colors_provisional" in m]
        if d:
            r["labels"]["dsatur_test_mean_provisional"] = float(np.mean(d))
    with open(out / "manifest.jsonl", "w") as fh:
        for m in manifest:
            fh.write(json.dumps(m) + "\n")
    json.dump([r for r in (emit if preset == "sample" else keep)], open(out / "families.json", "w"), indent=1)
    json.dump(dropped, open(out / "families_dropped.json", "w"), indent=1)
    json.dump({"version": VERSION, "preset": preset, "splits": SPLITS, "counts": counts,
               "baseline_train_seeds": BASELINE_TRAIN_SEEDS, "heldout_types": sorted(HELDOUT_TYPES),
               "heldout_size_min": HELDOUT_SIZE_MIN, "combo_holdout_fraction": COMBO_HOLDOUT_FRACTION,
               "combo_seed": COMBO_SEED, "no_headroom_gap": NO_HEADROOM_GAP}, open(out / "config.json", "w"), indent=1)
    report, fails = build_report(keep, dropped, manifest, counts, preset, {r["family_id"] for r in emit})
    (out / "report.txt").write_text(report)
    print(report if preset == "sample" else "\n".join(report.splitlines()[:60]))
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["sample", "full"], default="sample")
    ap.add_argument("--out", default="data_sample")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-graphs", action="store_true", help="write manifest/labels only (graphs are regenerable)")
    ap.add_argument("--no-dsatur", action="store_true")
    ap.add_argument("--verify", choices=["none", "some", "all"], default="some")
    a = ap.parse_args()
    run(a.preset, a.out, a.workers, not a.no_graphs, not a.no_dsatur, a.verify)
