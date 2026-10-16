"""Tests for graph coloring evaluators, DSATUR, RLF, and priority purity."""

import numpy as np
import networkx as nx
import pytest

from evaluator.coloring import dsatur, greedy_color, dsatur_priority, rlf
from evaluator.verify import verify, count_colors
from gate.parametric import parametric_color, make_parametric_priority, DSATUR_WEIGHTS


def _random_graph(n: int, p: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    triu = rng.rand(n, n) < p
    triu = np.triu(triu, k=1)
    adj = triu | triu.T
    return adj


def test_dsatur_numba_vs_greedy_color_200_graphs():
    """Verify numba dsatur matches greedy_color(dsatur_priority) on 200+ random graphs."""
    rng = np.random.RandomState(42)
    for i in range(210):
        n = rng.randint(20, 100)
        p = rng.uniform(0.05, 0.5)
        seed = 1000 + i
        adj = _random_graph(n, p, seed)

        c_numba = dsatur(adj)
        c_py = greedy_color(adj, dsatur_priority)

        assert verify(adj, c_numba), f"Numba DSATUR failed verification on graph {i}"
        assert verify(adj, c_py), f"Python DSATUR failed verification on graph {i}"
        assert np.array_equal(c_numba, c_py), f"Mismatch between numba and python DSATUR on graph {i}"
        assert count_colors(c_numba) == count_colors(c_py)


def test_dsatur_vs_networkx():
    """Compare DSATUR against networkx's saturation_largest_first."""
    rng = np.random.RandomState(123)
    dsatur_counts = []
    nx_counts = []

    for i in range(30):
        n = rng.randint(40, 80)
        p = rng.uniform(0.1, 0.4)
        adj = _random_graph(n, p, seed=2000 + i)

        c_ds = dsatur(adj)
        assert verify(adj, c_ds)
        k_ds = count_colors(c_ds)
        dsatur_counts.append(k_ds)

        # NetworkX greedy coloring
        G = nx.from_numpy_array(adj)
        nx_col_dict = nx.greedy_color(G, strategy="saturation_largest_first")
        nx_col_arr = np.array([nx_col_dict[u] for u in range(n)], dtype=np.int64)
        assert verify(adj, nx_col_arr)
        nx_counts.append(count_colors(nx_col_arr))

    mean_diff = abs(np.mean(dsatur_counts) - np.mean(nx_counts))
    assert mean_diff < 0.5, f"DSATUR differs too much from NetworkX: {mean_diff}"


def test_rlf_returns_proper_coloring():
    """Test RLF produces verified proper colorings across varied graph densities and sizes."""
    rng = np.random.RandomState(999)
    for i in range(30):
        n = rng.randint(30, 80)
        p = rng.uniform(0.05, 0.6)
        adj = _random_graph(n, p, seed=3000 + i)

        c_rlf = rlf(adj)
        assert verify(adj, c_rlf), f"RLF produced improper coloring on graph {i}"
        assert count_colors(c_rlf) > 0


def test_parametric_dsatur_reproduces_dsatur():
    """Verify that parametric priority with DSATUR weights reproduces DSATUR."""
    rng = np.random.RandomState(555)
    param_fn = make_parametric_priority(DSATUR_WEIGHTS)
    for i in range(30):
        n = rng.randint(30, 90)
        p = rng.uniform(0.1, 0.5)
        adj = _random_graph(n, p, seed=4000 + i)

        c_ds = dsatur(adj)
        c_param_jit = parametric_color(adj, DSATUR_WEIGHTS)
        c_param_py = greedy_color(adj, param_fn)

        assert verify(adj, c_param_jit)
        assert verify(adj, c_param_py)
        assert np.array_equal(c_ds, c_param_jit)
        assert np.array_equal(c_ds, c_param_py)


def test_priority_purity():
    """Calling priority twice with identical inputs yields same scores and does not mutate inputs."""
    adj = _random_graph(50, 0.2, seed=777)
    colors = np.full(50, -1, dtype=np.int64)
    colors[:10] = np.arange(10) % 3
    deg = adj.sum(axis=1).astype(np.int64)
    sat = np.zeros(50, dtype=np.int64)
    sat[:10] = 2
    uncolored_deg = deg.copy()
    uncolored_deg[:10] -= 2

    # Clones for comparison
    adj_before = adj.copy()
    colors_before = colors.copy()
    sat_before = sat.copy()
    deg_before = deg.copy()
    uncolored_deg_before = uncolored_deg.copy()

    scores1 = dsatur_priority(adj, colors, sat, deg, uncolored_deg)
    scores2 = dsatur_priority(adj, colors, sat, deg, uncolored_deg)

    assert np.array_equal(scores1, scores2)
    assert np.array_equal(adj, adj_before)
    assert np.array_equal(colors, colors_before)
    assert np.array_equal(sat, sat_before)
    assert np.array_equal(deg, deg_before)
    assert np.array_equal(uncolored_deg, uncolored_deg_before)


def test_determinism():
    """Running greedy coloring multiple times on the same graph produces identical colors."""
    adj = _random_graph(80, 0.25, seed=888)
    c1 = dsatur(adj)
    c2 = dsatur(adj)
    c3 = greedy_color(adj, dsatur_priority)
    assert np.array_equal(c1, c2)
    assert np.array_equal(c1, c3)


def test_rlf_priority_matches_standalone_rlf():
    """Verify greedy_color(adj, rlf_priority) produces identical colors to rlf(adj)."""
    rng = np.random.RandomState(42)
    from evaluator.coloring import rlf_priority
    for i in range(25):
        n = rng.randint(20, 60)
        p = rng.uniform(0.1, 0.5)
        adj = _random_graph(n, p, seed=6000 + i)

        c_standalone = rlf(adj)
        c_priority = greedy_color(adj, rlf_priority)

        assert verify(adj, c_priority)
        assert np.array_equal(c_standalone, c_priority), f"RLF priority mismatch on graph {i}"

