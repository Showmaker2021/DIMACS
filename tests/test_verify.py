"""Tests for independent verifier."""

import numpy as np
import pytest
from evaluator.verify import verify, count_colors


def test_verify_valid_coloring():
    # Triangle graph (3-clique), colors 0, 1, 2
    adj = np.array([
        [False, True, True],
        [True, False, True],
        [True, True, False],
    ], dtype=bool)
    colors = np.array([0, 1, 2], dtype=np.int64)
    assert verify(adj, colors) is True
    assert count_colors(colors) == 3


def test_verify_empty_graph():
    adj = np.zeros((0, 0), dtype=bool)
    colors = np.array([], dtype=np.int64)
    assert verify(adj, colors) is True
    assert count_colors(colors) == 0


def test_verify_rejects_wrong_length():
    adj = np.zeros((3, 3), dtype=bool)
    colors = np.array([0, 1], dtype=np.int64)
    assert verify(adj, colors) is False


def test_verify_rejects_uncolored_vertex():
    adj = np.zeros((3, 3), dtype=bool)
    colors = np.array([0, -1, 1], dtype=np.int64)
    assert verify(adj, colors) is False


def test_verify_rejects_conflicting_edge():
    # 2 vertices connected by an edge, same color
    adj = np.array([
        [False, True],
        [True, False],
    ], dtype=bool)
    colors = np.array([0, 0], dtype=np.int64)
    assert verify(adj, colors) is False


def test_verify_rejects_self_loops():
    adj = np.array([
        [True, False],
        [False, False],
    ], dtype=bool)
    colors = np.array([0, 1], dtype=np.int64)
    assert verify(adj, colors) is False


def test_verify_rejects_asymmetric_adj():
    adj = np.array([
        [False, True],
        [False, False],
    ], dtype=bool)
    colors = np.array([0, 1], dtype=np.int64)
    assert verify(adj, colors) is False


def test_verify_rejects_float_colors():
    adj = np.zeros((2, 2), dtype=bool)
    colors = np.array([0.0, 1.0])
    assert verify(adj, colors) is False
