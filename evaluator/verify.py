"""Independent verifier for graph colorings.

Rules:
- Independent code path (must not import from coloring.py or other evaluator internals).
- Every coloring produced anywhere in the project must pass verify before its color
  count is recorded.
"""

from typing import Union
import numpy as np


def verify(adj: np.ndarray, colors: np.ndarray) -> bool:
    """Verify that `colors` represents a valid, proper graph coloring for `adj`.

    Parameters:
    adj: (n, n) bool, symmetric adjacency matrix with no self-loops.
    colors: (n,) int, color assignment for each vertex (must all be >= 0).

    Returns:
    True if the coloring is proper and valid, False otherwise.
    """
    # 1. Type and dimension checks on colors
    if not isinstance(colors, np.ndarray):
        return False
    if colors.ndim != 1:
        return False
    if not np.issubdtype(colors.dtype, np.integer):
        return False

    n = colors.shape[0]

    # 2. Type and dimension checks on adj
    if not isinstance(adj, np.ndarray):
        return False
    if adj.ndim != 2 or adj.shape[0] != n or adj.shape[1] != n:
        return False
    if adj.dtype != bool:
        return False

    # Check for empty graph (0 vertices is trivially valid with empty colors)
    if n == 0:
        return True

    # Check no self-loops
    if np.any(np.diag(adj)):
        return False

    # Check symmetry of adjacency matrix
    if not np.array_equal(adj, adj.T):
        return False

    # 3. Check all vertices are colored with valid non-negative colors
    if np.any(colors < 0):
        return False

    # 4. Check proper coloring: no edge connects two vertices of the same color
    # For every edge (u, v) where adj[u, v] is True, colors[u] must not equal colors[v].
    u_idx, v_idx = np.nonzero(adj)
    if np.any(colors[u_idx] == colors[v_idx]):
        return False

    return True


def count_colors(colors: np.ndarray) -> int:
    """Count the number of distinct colors used in the coloring.

    Parameters:
    colors: (n,) int array of vertex colors.

    Returns:
    int: number of unique colors, or 0 if empty.
    """
    if len(colors) == 0:
        return 0
    return int(len(np.unique(colors)))
