"""Strict Hardened Sandbox for Priority Heuristic Execution.

Enforces:
1. AST Static Analysis:
   - Must define function `priority(adj, colors, sat, deg, uncolored_deg)` with exactly 5 positional args.
   - NO default argument values in ANY function (priority, helper, nested, or lambda).
   - NO names or attributes starting with '_' (blocks __class__, __bases__, _hidden, etc.).
   - NO `global` or `nonlocal` statements.
   - NO `import` or `from ... import` statements.
   - NO calls to `open`, `eval`, `exec`, `__import__`, `compile`, `getattr`, `setattr`, `delattr`.
   - Whitelist of allowed NumPy attributes only (blocks np.random, np.load, np.save, np.savetxt, np.loadtxt, etc.).
   - NO mutable module-level variables (only immutable constants and def allowed).
2. Whitelisted __builtins__:
   - Basic mathematical and built-in functions only: range, len, int, float, bool, min, max, abs, sum, round, enumerate, zip, isinstance.
3. Stateless Execution:
   - Code is compiled once.
   - A completely fresh namespace is instantiated and `exec`'d before EVERY single call to `priority`.
     Any internal state, module cache, or embedded solver attempting to persist data between steps is erased.
4. Subprocess Execution with Hard Time Termination:
   - Runs in isolated child process with strict wall-clock timeout.
   - Kills process immediately if timeout is exceeded.
"""

import os
import psutil
import hashlib
from typing import Dict, List, Tuple, Any, Optional, Callable
import ast
import time
import multiprocessing as mp
from queue import Empty
import numpy as np

from evaluator.verify import verify, count_colors


FORBIDDEN_CALLS = {"open", "eval", "exec", "__import__", "compile", "getattr", "setattr", "delattr"}

# Whitelist for method/attribute access on ANY object (arrays, etc.)
ALLOWED_OBJECT_ATTRS = {
    "sum", "max", "min", "argmax", "argmin",
    "any", "all", "astype", "copy", "reshape", "ravel",
    "nonzero", "cumsum", "dot", "T", "shape", "size", "dtype",
    "diagonal", "clip", "round",
}

# Explicitly forbidden dangerous attributes (checked for clear error messages)
FORBIDDEN_ATTRS = {
    "setflags", "flags", "tofile", "dump", "dumps", "ctypes",
    "data", "base", "view", "resize", "byteswap",
}

ALLOWED_NP_ATTRS = {
    # array creation & shapes
    "zeros", "ones", "empty", "full", "zeros_like", "ones_like", "empty_like", "full_like",
    "arange", "linspace", "array", "asarray", "copy", "shape",
    # indexing & searching
    "where", "flatnonzero", "nonzero", "argmax", "argmin", "argsort", "sort",
    "isin", "in1d", "clip", "count_nonzero",
    # reductions & math
    "sum", "mean", "min", "max", "maximum", "minimum", "abs", "sqrt", "exp", "log", "cumsum", "dot", "matmul",
    # matrix & structural operations
    "diag", "diagonal", "triu", "tril", "outer", "bincount", "unique", "sign",
    # logical
    "all", "any", "logical_and", "logical_or", "logical_not", "allclose", "equal",
    # types & constants
    "int64", "int32", "float64", "float32", "bool_", "ndarray",
    "inf", "nan", "pi", "e",
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Internal helper to satisfy C-extension imports (e.g. numpy._core._methods) while forbidding all user imports."""
    if name == "numpy" or name.startswith("numpy."):
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of module '{name}' is strictly forbidden in sandbox")


SAFE_BUILTINS = {
    "range": range,
    "len": len,
    "int": int,
    "float": float,
    "bool": bool,
    "min": min,
    "max": max,
    "abs": abs,
    "sum": sum,
    "round": round,
    "enumerate": enumerate,
    "zip": zip,
    "isinstance": isinstance,
    "True": True,
    "False": False,
    "None": None,
    "__import__": _safe_import,
}


# ==============================================================================
# 1. AST FILTER
# ==============================================================================

class SecurityASTVisitor(ast.NodeVisitor):
    def __init__(self):
        self.errors = []
        self.found_priority = False

    def visit_Name(self, node: ast.Name):
        if node.id.startswith("_"):
            self.errors.append(f"Identifier '{node.id}' starting with '_' is forbidden")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name.startswith("_"):
            self.errors.append(f"Function name '{node.name}' starting with '_' is forbidden")

        if node.name == "priority":
            self.found_priority = True
            all_args = [a.arg for a in node.args.args]
            expected_args = ["adj", "colors", "sat", "deg", "uncolored_deg"]
            if all_args != expected_args:
                self.errors.append(f"Function 'priority' must have exactly 5 arguments {expected_args}, got {all_args}")
            if node.args.vararg or node.args.kwarg:
                self.errors.append("*args and **kwargs are forbidden in 'priority'")

        # NO default arguments in ANY function
        if node.args.defaults:
            self.errors.append(f"Default argument values are forbidden in function '{node.name}'")
        if node.args.kw_defaults:
            self.errors.append(f"Keyword default argument values are forbidden in function '{node.name}'")

        for a in node.args.args:
            if a.arg.startswith("_"):
                self.errors.append(f"Argument '{a.arg}' starting with '_' is forbidden")

        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda):
        # NO default arguments in lambda
        if node.args.defaults or node.args.kw_defaults:
            self.errors.append("Default argument values are forbidden in lambda expressions")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global):
        self.errors.append("Use of 'global' is forbidden")
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal):
        self.errors.append("Use of 'nonlocal' is forbidden")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        self.errors.append("Explicit 'import' statements are forbidden (numpy is provided as np and numpy)")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        self.errors.append("Explicit 'from ... import ...' statements are forbidden")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            self.errors.append(f"Call to '{node.func.id}' is forbidden")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr.startswith("_"):
            self.errors.append(f"Access to attribute '{node.attr}' starting with '_' is forbidden")

        if node.attr in FORBIDDEN_ATTRS:
            self.errors.append(f"Access to forbidden attribute '{node.attr}' is blocked")

        # Whitelist check for ALL attributes
        if isinstance(node.value, ast.Name) and node.value.id in ("np", "numpy"):
            if node.attr not in ALLOWED_NP_ATTRS:
                self.errors.append(f"NumPy attribute '{node.attr}' is not in the allowed whitelist")
        else:
            if node.attr not in ALLOWED_OBJECT_ATTRS:
                self.errors.append(f"Attribute '{node.attr}' is not in the allowed whitelist for objects")

        self.generic_visit(node)


def is_immutable_literal(node: ast.AST) -> bool:
    """Check if AST node represents an immutable literal (number, string, bool, tuple of literals)."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Tuple):
        return all(is_immutable_literal(elt) for elt in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        return True
    return False


def validate_ast(code_str: str) -> Tuple[bool, Optional[str]]:
    """Validate candidate code against strict AST rules.

    Returns:
    (is_valid, error_message)
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"SyntaxError: {str(e)}"

    # Check top-level statements for mutable module-level variables
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.Pass)):
            continue
        elif isinstance(stmt, ast.Assign):
            if not is_immutable_literal(stmt.value):
                return False, "Mutable module-level variables are forbidden (only immutable constants and def allowed)"
        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None and not is_immutable_literal(stmt.value):
                return False, "Mutable module-level variables are forbidden"
        else:
            return False, f"Statement of type {type(stmt).__name__} is forbidden at module level"

    visitor = SecurityASTVisitor()
    visitor.visit(tree)

    if not visitor.found_priority:
        return False, "Code must define a function named 'priority'"

    if visitor.errors:
        return False, f"AST Security Violation: {'; '.join(visitor.errors)}"

    return True, None


# ==============================================================================
# 2. STATELESSNESS & PURITY CHECKER
# ==============================================================================

def check_statelessness_and_purity(code_obj: Any, adj: np.ndarray) -> Tuple[bool, Optional[str]]:
    """Verify function is pure and stateless on shuffled partial coloring states."""
    n = adj.shape[0]

    colors_a = np.full(n, -1, dtype=np.int64)
    sat_a = np.zeros(n, dtype=np.int64)
    deg_a = adj.sum(axis=1).astype(np.int64)
    udeg_a = deg_a.copy()

    colors_b = colors_a.copy()
    colors_b[0] = 0
    sat_b = sat_a.copy()
    nb_0 = np.flatnonzero(adj[0])
    sat_b[nb_0] += 1
    deg_b = deg_a.copy()
    udeg_b = udeg_a.copy()
    udeg_b[nb_0] -= 1

    for arr in [adj, colors_a, sat_a, deg_a, udeg_a, colors_b, sat_b, deg_b, udeg_b]:
        arr.flags.writeable = False

    def get_fresh_priority():
        ns = {"__builtins__": SAFE_BUILTINS, "np": np, "numpy": np}
        exec(code_obj, ns)
        return ns["priority"]

    try:
        # Call 1 on State A
        fn1 = get_fresh_priority()
        scores_a1 = fn1(adj, colors_a, sat_a, deg_a, udeg_a)
        if not isinstance(scores_a1, np.ndarray) or scores_a1.shape != (n,):
            return False, f"Output must be 1D numpy array of shape ({n},), got {type(scores_a1)} of shape {getattr(scores_a1, 'shape', None)}"
        if not np.all(np.isfinite(scores_a1)):
            return False, "Output contains non-finite values (NaN or Inf)"

        # Call on State B
        fn2 = get_fresh_priority()
        scores_b = fn2(adj, colors_b, sat_b, deg_b, udeg_b)
        if not isinstance(scores_b, np.ndarray) or scores_b.shape != (n,):
            return False, f"Output on state B must be 1D numpy array of shape ({n},)"
        if not np.all(np.isfinite(scores_b)):
            return False, "Output on state B contains non-finite values (NaN or Inf)"

        # Call 2 on State A (shuffled order check)
        fn3 = get_fresh_priority()
        scores_a2 = fn3(adj, colors_a, sat_a, deg_a, udeg_a)

        if not np.array_equal(scores_a1, scores_a2):
            return False, "Function failed statelessness test: calling on same input produced different scores"

        return True, None
    except MemoryError as e:
        return False, f"Memory limit exceeded (MemoryError: {str(e)})"
    except ValueError as e:
        if "read-only" in str(e).lower() or "assignment destination is read-only" in str(e).lower():
            return False, "Input mutation detected: function attempted to modify read-only inputs"
        return False, f"ValueError: {str(e)}"
    except Exception as e:
        return False, f"Runtime error during purity test: {type(e).__name__}: {str(e)}"


# ==============================================================================
# 3. SINGLE GRAPH EVALUATION WITH TIME LIMIT & FRESH NAMESPACE PER CALL
# ==============================================================================

def color_graph_with_priority(
    adj: np.ndarray,
    compiled_code: Any,
    per_graph_timeout: float,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Greedy color a graph, re-executing in a completely fresh namespace before EVERY call to priority."""
    n = adj.shape[0]
    deg = adj.sum(axis=1).astype(np.int64)
    max_k = int(deg.max()) + 2
    colors = np.full(n, -1, dtype=np.int64)
    sat = np.zeros(n, dtype=np.int64)
    uncolored_deg = deg.copy()
    nb_has_color = np.zeros((n, max_k), dtype=bool)
    uncolored = np.ones(n, dtype=bool)

    adj_ro = adj.view()
    adj_ro.flags.writeable = False

    t_start = time.perf_counter()

    for step in range(n):
        if time.perf_counter() - t_start > per_graph_timeout:
            return None, f"Per-graph timeout exceeded ({per_graph_timeout:.3f}s)"

        colors_ro = colors.view()
        colors_ro.flags.writeable = False
        sat_ro = sat.view()
        sat_ro.flags.writeable = False
        deg_ro = deg.view()
        deg_ro.flags.writeable = False
        uncolored_deg_ro = uncolored_deg.view()
        uncolored_deg_ro.flags.writeable = False

        # Fresh namespace instantiated before EVERY single step
        call_ns = {
            "__builtins__": SAFE_BUILTINS,
            "np": np,
            "numpy": np,
        }
        try:
            exec(compiled_code, call_ns)
            priority_fn = call_ns["priority"]
            scores = priority_fn(adj_ro, colors_ro, sat_ro, deg_ro, uncolored_deg_ro)
        except MemoryError as e:
            return None, f"Memory limit exceeded (MemoryError: {str(e)})"
        except ValueError as e:
            if "read-only" in str(e).lower() or "assignment destination is read-only" in str(e).lower():
                return None, "Input mutation detected: function attempted to modify read-only inputs"
            return None, f"ValueError: {str(e)}"
        except Exception as e:
            return None, f"Runtime error in priority function: {type(e).__name__}: {str(e)}"

        if not isinstance(scores, np.ndarray) or scores.shape != (n,):
            return None, f"Scores must be 1D numpy array of shape ({n},)"

        masked_scores = np.where(uncolored, scores, -np.inf)
        max_score = np.max(masked_scores)
        if not np.isfinite(max_score):
            return None, "All uncolored vertex scores are non-finite (NaN or -Inf)"

        candidates = np.flatnonzero(masked_scores == max_score)
        v = int(candidates[0])

        c = int(np.argmin(nb_has_color[v]))
        colors[v] = c
        uncolored[v] = False

        neighbors = np.flatnonzero(adj[v])
        new_color_neighbors = neighbors[~nb_has_color[neighbors, c]]
        sat[new_color_neighbors] += 1
        nb_has_color[neighbors, c] = True
        uncolored_deg[neighbors] -= 1

    if not verify(adj, colors):
        return None, "Independent coloring verification failed (invalid coloring)"

    return colors, None


# ==============================================================================
# 4. SUBPROCESS WORKER (TERMINATES ON HARD TIMEOUT)
# ==============================================================================

# Enforce single threading in environment BEFORE child processes spawn
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _sandbox_subprocess_target(
    code_str: str,
    graphs: List[Tuple[int, np.ndarray]],
    per_graph_timeout: float,
    out_queue: mp.Queue,
):
    # 1. Scrub all *_API_KEY from environment in the worker process
    for k in list(os.environ.keys()):
        if "API_KEY" in k.upper():
            del os.environ[k]

    # 2. Limit thread concurrency for BLAS/OMP using threadpoolctl and env vars
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Test hook for security and threadpool validation
    if code_str == "__DIAGNOSTIC_CHECK__":
        try:
            import threadpoolctl
            tp_info = threadpoolctl.threadpool_info()
        except Exception:
            tp_info = []
        api_keys_left = [k for k in os.environ.keys() if "API_KEY" in k.upper()]
        out_queue.put({
            "success": True,
            "api_keys": api_keys_left,
            "threadpool_info": tp_info,
            "error": None,
        })
        return

    try:
        compiled_code = compile(code_str, "<candidate>", "exec")
    except Exception as e:
        out_queue.put({"success": False, "error": f"Compilation error: {type(e).__name__}: {str(e)}"})
        return

    # Check statelessness & purity on first graph
    purity_ok, purity_err = check_statelessness_and_purity(compiled_code, graphs[0][1])
    if not purity_ok:
        out_queue.put({"success": False, "error": f"Statelessness/Purity check failed: {purity_err}"})
        return

    color_counts = []
    tie_break_terms = []
    first_3_hashes = []

    for idx, (seed, adj) in enumerate(graphs):
        colors, err = color_graph_with_priority(adj, compiled_code, per_graph_timeout)
        if err is not None:
            out_queue.put({"success": False, "error": f"Failed on seed {seed}: {err}"})
            return

        k = count_colors(colors)
        color_counts.append(k)

        if idx < 3:
            first_3_hashes.append(hashlib.sha256(colors.tobytes()).hexdigest())

        classes, counts = np.unique(colors, return_counts=True)
        tb = float(np.sum(counts ** 2)) / float(len(colors) ** 2)
        tie_break_terms.append(tb)

    behavior_hash = "-".join(first_3_hashes)

    out_queue.put({
        "success": True,
        "color_counts": color_counts,
        "mean_colors": float(np.mean(color_counts)),
        "mean_tie_break": float(np.mean(tie_break_terms)),
        "behavior_hash": behavior_hash,
        "error": None,
    })


def evaluate_candidate_safe(
    code_str: str,
    graphs: List[Tuple[int, np.ndarray]],
    per_graph_timeout: float = 1.0,
    overall_timeout: float = 20.0,
) -> Dict[str, Any]:
    """Execute candidate with AST filter, fresh namespace, and child process with hard termination."""
    if code_str != "__DIAGNOSTIC_CHECK__":
        ast_ok, ast_err = validate_ast(code_str)
        if not ast_ok:
            return {"success": False, "error": ast_err, "mean_colors": None}

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(
        target=_sandbox_subprocess_target,
        args=(code_str, graphs, per_graph_timeout, q),
    )

    proc.start()

    t_start = time.perf_counter()
    mem_limit_bytes = int(1.5 * 1024 * 1024 * 1024)  # 1.5 GB
    mem_killed = False

    while proc.is_alive():
        elapsed = time.perf_counter() - t_start
        if elapsed > overall_timeout:
            break

        try:
            p = psutil.Process(proc.pid)
            mem_info = p.memory_info()
            rss = mem_info.rss
            private = getattr(mem_info, "private", 0)
            vms = getattr(mem_info, "vms", 0)
            if (rss > mem_limit_bytes) or (private > mem_limit_bytes) or (vms > mem_limit_bytes):
                mem_killed = True
                proc.kill()
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break

        time.sleep(0.02)

    if mem_killed:
        proc.join(timeout=0.5)
        return {"success": False, "error": "Memory limit exceeded (>1.5GB: process killed)", "mean_colors": None}

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.kill()
        return {"success": False, "error": f"Hard execution timeout exceeded ({overall_timeout:.1f}s)", "mean_colors": None}

    try:
        res = q.get_nowait()
        return res
    except Empty:
        return {"success": False, "error": f"Process exited unexpectedly (exitcode={proc.exitcode})", "mean_colors": None}
