"""Unit tests for the sandbox security, AST filter, and failure modes."""

import pytest
import numpy as np
from gate.evolved.sandbox import validate_ast, evaluate_candidate_safe


@pytest.fixture
def sample_graph():
    # 20-node random graph
    rng = np.random.RandomState(42)
    mat = (rng.rand(20, 20) < 0.3).astype(np.bool_)
    mat = np.triu(mat, 1)
    mat = mat | mat.T
    return [(42, mat)]


def test_valid_dsatur_code(sample_graph):
    code = """
def priority(adj, colors, sat, deg, uncolored_deg):
    n = adj.shape[0]
    return sat * (n + 1) + uncolored_deg
"""
    res = evaluate_candidate_safe(code, sample_graph, per_graph_timeout=1.0)
    assert res["success"] is True
    assert res["mean_colors"] > 0
    assert res["error"] is None


def test_rejection_of_infinite_loop(sample_graph):
    code = """
def priority(adj, colors, sat, deg, uncolored_deg):
    while True:
        pass
    return sat
"""
    # Hard timeout 1.5s
    res = evaluate_candidate_safe(code, sample_graph, per_graph_timeout=0.2, overall_timeout=1.5)
    assert res["success"] is False
    assert "timeout" in res["error"].lower()


def test_rejection_of_input_mutation(sample_graph):
    code = """
def priority(adj, colors, sat, deg, uncolored_deg):
    sat[0] = 99
    return sat
"""
    res = evaluate_candidate_safe(code, sample_graph, per_graph_timeout=1.0)
    assert res["success"] is False
    assert "mutation" in res["error"].lower() or "read-only" in res["error"].lower()


def test_rejection_of_np_random():
    code = """
def priority(adj, colors, sat, deg, uncolored_deg):
    return sat + np.random.rand(len(sat))
"""
    is_valid, err = validate_ast(code)
    assert is_valid is False
    assert "random" in err.lower()


def test_rejection_of_open():
    code = """
def priority(adj, colors, sat, deg, uncolored_deg):
    f = open('secret.txt', 'w')
    return sat
"""
    is_valid, err = validate_ast(code)
    assert is_valid is False
    assert "open" in err.lower()


def test_rejection_of_mutable_default():
    code = """
def priority(adj, colors, sat, deg, uncolored_deg, cache={}):
    return sat
"""
    is_valid, err = validate_ast(code)
    assert is_valid is False
    assert "default" in err.lower() or "arguments" in err.lower()


def test_rejection_of_global():
    code = """
my_cache = 0
def priority(adj, colors, sat, deg, uncolored_deg):
    global my_cache
    my_cache += 1
    return sat
"""
    is_valid, err = validate_ast(code)
    assert is_valid is False
    assert "global" in err.lower()


def test_rejection_of_nan_inf(sample_graph):
    code = """
def priority(adj, colors, sat, deg, uncolored_deg):
    scores = sat.astype(float)
    scores[0] = float('nan')
    return scores
"""
    res = evaluate_candidate_safe(code, sample_graph, per_graph_timeout=1.0)
    assert res["success"] is False
    assert "non-finite" in res["error"].lower() or "nan" in res["error"].lower()


def test_rejection_of_wrong_shape(sample_graph):
    code = """
def priority(adj, colors, sat, deg, uncolored_deg):
    return sat[:-1]
"""
    res = evaluate_candidate_safe(code, sample_graph, per_graph_timeout=1.0)
    assert res["success"] is False
    assert "shape" in res["error"].lower()


def test_rejection_of_too_slow_per_graph(sample_graph):
    code_busy = """
def priority(adj, colors, sat, deg, uncolored_deg):
    s = 0
    for i in range(1000000):
        s += i
    return sat
"""
    res = evaluate_candidate_safe(code_busy, sample_graph, per_graph_timeout=0.005, overall_timeout=2.0)
    assert res["success"] is False
    assert "timeout" in res["error"].lower()


# --- NEW HARDENED SANDBOX TESTS ---

def test_rejection_of_module_dict_cache():
    code = """
module_cache = {}
def priority(adj, colors, sat, deg, uncolored_deg):
    return sat
"""
    is_valid, err = validate_ast(code)
    assert is_valid is False
    assert "mutable" in err.lower()


def test_rejection_of_nested_function_defaults():
    code = """
def priority(adj, colors, sat, deg, uncolored_deg):
    def helper(x, c=[]):
        return x
    return sat
"""
    is_valid, err = validate_ast(code)
    assert is_valid is False
    assert "default" in err.lower()


def test_rejection_of_lambda_defaults():
    code = """
def priority(adj, colors, sat, deg, uncolored_deg):
    f = lambda x, y=1: x + y
    return sat
"""
    is_valid, err = validate_ast(code)
    assert is_valid is False
    assert "default" in err.lower()


def test_rejection_of_np_savetxt_loadtxt():
    code_save = """
def priority(adj, colors, sat, deg, uncolored_deg):
    np.savetxt('data.txt', sat)
    return sat
"""
    is_valid, err = validate_ast(code_save)
    assert is_valid is False
    assert "whitelist" in err.lower() or "savetxt" in err.lower()

    code_load = """
def priority(adj, colors, sat, deg, uncolored_deg):
    x = np.loadtxt('data.txt')
    return sat
"""
    is_valid, err = validate_ast(code_load)
    assert is_valid is False
    assert "whitelist" in err.lower() or "loadtxt" in err.lower()


def test_rejection_of_dunder_traversal():
    code = """
def priority(adj, colors, sat, deg, uncolored_deg):
    b = ().__class__.__bases__
    return sat
"""
    is_valid, err = validate_ast(code)
    assert is_valid is False
    assert "_" in err.lower()


def test_rejection_of_embedded_solver_timeout(sample_graph):
    # An embedded solver that tries to do heavy graph coloring / search on step 0
    # Because namespace is re-created before EVERY call to priority, this heavy work
    # is repeated on EVERY step, rapidly exceeding per-graph timeout!
    code_heavy_solver = """
def priority(adj, colors, sat, deg, uncolored_deg):
    # Simulate heavy solver (e.g. 50ms per step)
    n = len(sat)
    total = 0
    for i in range(150000):
        total += i
    return sat.astype(float)
"""
    res = evaluate_candidate_safe(code_heavy_solver, sample_graph, per_graph_timeout=0.03, overall_timeout=2.0)
    assert res["success"] is False
    assert "timeout" in res["error"].lower()


# --- TESTS FOR ATTRIBUTE WHITELIST ENFORCEMENT ---

def test_rejection_of_setflags():
    code = """
def priority(adj, colors, sat, deg, uncolored_deg):
    colors.setflags(write=True)
    return sat
"""
    is_valid, err = validate_ast(code)
    assert is_valid is False
    assert "setflags" in err.lower() or "forbidden" in err.lower()


def test_rejection_of_tofile():
    code = """
def priority(adj, colors, sat, deg, uncolored_deg):
    sat.tofile("x")
    return sat
"""
    is_valid, err = validate_ast(code)
    assert is_valid is False
    assert "tofile" in err.lower() or "forbidden" in err.lower()


def test_rejection_of_dump():
    code = """
def priority(adj, colors, sat, deg, uncolored_deg):
    sat.dump("x")
    return sat
"""
    is_valid, err = validate_ast(code)
    assert is_valid is False
    assert "dump" in err.lower() or "forbidden" in err.lower()


def test_rejection_of_data():
    code = """
def priority(adj, colors, sat, deg, uncolored_deg):
    d = sat.data
    return sat
"""
    is_valid, err = validate_ast(code)
    assert is_valid is False
    assert "data" in err.lower() or "forbidden" in err.lower()


def test_rejection_of_view():
    code = """
def priority(adj, colors, sat, deg, uncolored_deg):
    v = sat.view()
    return sat
"""
    is_valid, err = validate_ast(code)
    assert is_valid is False
    assert "view" in err.lower() or "forbidden" in err.lower()


def test_rejection_of_huge_memory_allocation(sample_graph):
    # Candidate attempting to allocate 40GB
    code_huge = """
def priority(adj, colors, sat, deg, uncolored_deg):
    # Allocate 40GB
    huge = np.zeros((100000, 100000), dtype=float)
    return sat
"""
    res = evaluate_candidate_safe(code_huge, sample_graph, per_graph_timeout=2.0, overall_timeout=5.0)
    assert res["success"] is False
    assert "memory" in res["error"].lower()


def test_rejection_of_rss_over_1_5gb(sample_graph):
    # Candidate allocating 2.0GB > 1.5GB to trigger psutil process kill
    code_rss = """
def priority(adj, colors, sat, deg, uncolored_deg):
    huge = np.ones((500, 1000, 500), dtype=np.float64)
    return sat.astype(float)
"""
    res = evaluate_candidate_safe(code_rss, sample_graph, per_graph_timeout=2.0, overall_timeout=5.0)
    assert res["success"] is False
    assert "memory" in res["error"].lower()


def test_rejection_of_gradual_allocation_over_1_5gb(sample_graph):
    # Candidate allocating and gradually writing to memory until exceeding 1.5GB
    code_gradual = """
def priority(adj, colors, sat, deg, uncolored_deg):
    arr = np.zeros((500, 1000, 500), dtype=float)
    for i in range(500):
        arr[i] = 1.0
    return sat.astype(float)
"""
    res = evaluate_candidate_safe(code_gradual, sample_graph, per_graph_timeout=2.0, overall_timeout=5.0)
    assert res["success"] is False
    assert "memory" in res["error"].lower()


def test_env_key_scrubbing_and_thread_limits(sample_graph):
    import os
    os.environ["FAKE_API_KEY"] = "secret_key_12345"
    os.environ["ANOTHER_API_KEY"] = "abcdef"

    try:
        res = evaluate_candidate_safe("__DIAGNOSTIC_CHECK__", sample_graph)
        assert res["success"] is True
        # Assert keys containing API_KEY were scrubbed
        assert "FAKE_API_KEY" not in res["api_keys"]
        assert "ANOTHER_API_KEY" not in res["api_keys"]
        assert len(res["api_keys"]) == 0

        # Assert threadpool is constrained to 1
        for pool in res["threadpool_info"]:
            assert pool.get("num_threads", 1) == 1
    finally:
        os.environ.pop("FAKE_API_KEY", None)
        os.environ.pop("ANOTHER_API_KEY", None)


def test_whitelisted_numpy_math_functions(sample_graph):
    """Verify newly whitelisted pure mathematical NumPy functions pass AST and execute safely."""
    code = """
def priority(adj, colors, sat, deg, uncolored_deg):
    n = adj.shape[0]
    # diag & diagonal
    d1 = np.diag(adj)
    d2 = adj.diagonal()
    # matmul
    m1 = np.matmul(adj.astype(float), deg.astype(float))
    # triu & tril
    u1 = np.triu(adj)
    l1 = np.tril(adj)
    # outer
    o1 = np.outer(sat[:5], deg[:5])
    # bincount & unique
    pos_colors = np.where(colors >= 0, colors, 0)
    bc = np.bincount(pos_colors)
    unq = np.unique(colors)
    # sign, minimum, maximum
    s1 = np.sign(deg)
    m_min = np.minimum(deg, sat)
    m_max = np.maximum(deg, sat)
    # count_nonzero & cumsum
    nnz = np.count_nonzero(adj)
    cs = np.cumsum(deg)
    return sat.astype(float) * float(n) + uncolored_deg.astype(float)
"""
    is_valid, err = validate_ast(code)
    assert is_valid is True, f"AST rejected whitelisted functions: {err}"

    res = evaluate_candidate_safe(code, sample_graph, per_graph_timeout=1.0)
    assert res["success"] is True, f"Sandbox execution failed: {res.get('error')}"
    assert len(res["color_counts"]) == len(sample_graph)


def test_safe_import_blocks_external_modules():
    from gate.evolved.sandbox import _safe_import
    # Allowed numpy and numpy submodules
    mod_np = _safe_import("numpy")
    assert mod_np is not None
    mod_core = _safe_import("numpy._core._methods")
    assert mod_core is not None

    # Forbidden fake numpy names and external modules
    import pytest
    with pytest.raises(ImportError, match="strictly forbidden"):
        _safe_import("numpy_fake")
    with pytest.raises(ImportError, match="strictly forbidden"):
        _safe_import("os")
    with pytest.raises(ImportError, match="strictly forbidden"):
        _safe_import("sys")
    with pytest.raises(ImportError, match="strictly forbidden"):
        _safe_import("subprocess")


def test_rejection_of_non_whitelisted_numpy_attrs():
    # Calling non-whitelisted np functions must be rejected by AST
    code_fill_diag = """
def priority(adj, colors, sat, deg, uncolored_deg):
    np.fill_diagonal(adj, 0)
    return sat
"""
    is_valid, err = validate_ast(code_fill_diag)
    assert is_valid is False
    assert "fill_diagonal" in err

    code_inv = """
def priority(adj, colors, sat, deg, uncolored_deg):
    x = np.linalg.inv(adj)
    return sat
"""
    is_valid, err = validate_ast(code_inv)
    assert is_valid is False
    assert "linalg" in err


def test_rejection_of_direct_import_statements():
    # Direct import statements must be rejected by AST
    code_import_math = """
import math
def priority(adj, colors, sat, deg, uncolored_deg):
    return sat
"""
    is_valid, err = validate_ast(code_import_math)
    assert is_valid is False
    assert "import" in err.lower()

    code_from_os = """
from os import path
def priority(adj, colors, sat, deg, uncolored_deg):
    return sat
"""
    is_valid, err = validate_ast(code_from_os)
    assert is_valid is False
    assert "import" in err.lower()

    code_import_numpy = """
import numpy as my_np
def priority(adj, colors, sat, deg, uncolored_deg):
    return sat
"""
    is_valid, err = validate_ast(code_import_numpy)
    assert is_valid is False
    assert "import" in err.lower()




