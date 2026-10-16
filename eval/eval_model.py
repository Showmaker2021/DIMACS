"""eval/eval_model.py: Offline evaluation harness for priority heuristics.

Implements generation backends (mock, hf, vllm, api), sandboxed execution,
disk caching under runs/eval/_cache/, test-split lock protection, and
comprehensive coverage/behavioral metrics.

CLI Usage:
    python -m eval.eval_model --backend mock --split val --k 1 --n-samples 2 --run-id eval_test_01
"""

from abc import ABC, abstractmethod
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
_current_dir = str(Path(__file__).resolve().parent)
while _current_dir in sys.path:
    sys.path.remove(_current_dir)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.coverage import beats_rf, pass_at_k, coverage_at_k, best_of_k, compute_rf
from evaluator.coloring import dsatur, rlf
from evaluator.verify import count_colors
from gate.evolved.sandbox import evaluate_candidate_safe
from data.generate_dataset import make_graph, adj_from_edges

SEEDS_PATH = REPO_ROOT / "SEEDS.md"
FAMILIES_PATH = REPO_ROOT / "data" / "data_v1" / "families.json"
SPLITS_V1_2_PATH = REPO_ROOT / "data" / "data_v1_2" / "splits.json"


# ==============================================================================
# 1. PROMPT BUILDER
# ==============================================================================

def build_eval_prompt(card: str, prompt_version: str = "v1") -> str:
    """Build evaluation prompt using the vectorized priority API signature.

    Conforms to Section 14.2 of PLAN_PHASE6_ONWARD.md.
    """
    return f"""[SYSTEM] You write Python priority functions for greedy graph coloring.
[USER]
Task: write `priority(adj, colors, sat, deg, uncolored_deg)` returning a NumPy
array of scores (one per vertex). At each step the trusted colorer colors the
UNCOLORED vertex with the highest score using the smallest feasible color.

Parameters:
  adj: (n, n) bool ndarray, symmetric, no self-loops (read-only)
  colors: (n,) int64 ndarray, -1 = uncolored, >=0 color id (read-only)
  sat: (n,) int64 ndarray, number of distinct colors among colored neighbors (read-only)
  deg: (n,) int64 ndarray, total degree in full graph (read-only)
  uncolored_deg: (n,) int64 ndarray, number of uncolored neighbors (read-only)

Rules:
- Pure function: no globals, no state between calls, no I/O, no randomness.
- No imports: numpy is available as `np` and `numpy`.
- Only whitelisted NumPy operations (zeros, ones, where, flatnonzero, dot, sum, max, min, argmax, argmin, etc.).
- No default arguments in functions or lambdas.
- Positional parameters only.

Graph family description:
{card}

Return only the function code inside a ```python ... ``` block.
[ASSISTANT]
"""


def extract_code(text: str) -> str:
    """Extract Python code block from LLM response text."""
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


# ==============================================================================
# 2. GENERATION BACKENDS
# ==============================================================================

class GenerationBackend(ABC):
    """Abstract interface for heuristic code generation backends."""

    @abstractmethod
    def generate(self, prompt: str, n: int, temperature: float, seed: int) -> List[str]:
        """Generate n code completions for the given prompt."""
        pass


class MockBackend(GenerationBackend):
    """Deterministic synthetic generator emitting predictable archetypes.

    Archetypes:
      0: DSATUR-like (reproduces exact DSATUR priority)
      1: RLF-like (reproduces valid RLF-style priority)
      2: Broken code (AST violation: forbidden import)
      3: Slow code (per-graph timeout loop)
    """

    MOCK_DSATUR = (
        "def priority(adj, colors, sat, deg, uncolored_deg):\n"
        "    n = adj.shape[0]\n"
        "    return sat.astype(np.float64) * (n + 1) + uncolored_deg.astype(np.float64)\n"
    )

    MOCK_RLF = (
        "def priority(adj, colors, sat, deg, uncolored_deg):\n"
        "    n = adj.shape[0]\n"
        "    k = int(np.max(colors)) if np.any(colors >= 0) else -1\n"
        "    if k == -1:\n"
        "        return uncolored_deg.astype(np.float64)\n"
        "    in_k = (colors == k)\n"
        "    adj_k = np.dot(adj, in_k.astype(np.float64)) > 0\n"
        "    cand = (colors == -1) & (~adj_k)\n"
        "    excl = (colors == -1) & adj_k\n"
        "    n_e = np.dot(adj, excl.astype(np.float64))\n"
        "    n_c = np.dot(adj, cand.astype(np.float64))\n"
        "    scores = np.where(cand, n_e * (n + 1) - n_c, -1e9)\n"
        "    if not np.any(cand):\n"
        "        scores = np.where(colors == -1, uncolored_deg.astype(np.float64), -1e9)\n"
        "    return scores\n"
    )

    MOCK_BROKEN_AST = (
        "def priority(adj, colors, sat, deg, uncolored_deg):\n"
        "    import os\n"
        "    return deg\n"
    )

    MOCK_SLOW_TIMEOUT = (
        "def priority(adj, colors, sat, deg, uncolored_deg):\n"
        "    total = 0.0\n"
        "    for i in range(25000000):\n"
        "        total = total + i\n"
        "    return sat * 100.0 + deg\n"
    )

    def __init__(self, archetypes: Optional[List[str]] = None):
        self.archetypes = archetypes or [
            self.MOCK_DSATUR,
            self.MOCK_RLF,
            self.MOCK_BROKEN_AST,
            self.MOCK_SLOW_TIMEOUT,
        ]

    def generate(self, prompt: str, n: int, temperature: float, seed: int) -> List[str]:
        completions = []
        for i in range(n):
            idx = i % len(self.archetypes)
            raw = self.archetypes[idx]
            wrapped = f"Here is the priority function:\n```python\n{raw}\n```"
            completions.append(wrapped)
        return completions


class HFBackend(GenerationBackend):
    """HuggingFace transformers backend with strictly lazy loading."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self._loaded = False
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._loaded:
            return
        try:
            import torch
            import transformers
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required to use HFBackend. "
                "Ensure they are installed and CUDA is configured if using GPU."
            ) from e
        # Stub for model initialization
        self._loaded = True

    def generate(self, prompt: str, n: int, temperature: float, seed: int) -> List[str]:
        self._load()
        raise NotImplementedError("HFBackend execution stub: actual generation requires local weights.")


class VLLMBackend(GenerationBackend):
    """vLLM backend with strictly lazy loading."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self._loaded = False
        self._llm = None

    def _load(self):
        if self._loaded:
            return
        try:
            import vllm
        except ImportError as e:
            raise ImportError(
                "vllm is required to use VLLMBackend. "
                "Ensure vllm is installed on a supported platform."
            ) from e
        self._loaded = True

    def generate(self, prompt: str, n: int, temperature: float, seed: int) -> List[str]:
        self._load()
        raise NotImplementedError("VLLMBackend execution stub: actual generation requires vllm engine.")


class APIBackend(GenerationBackend):
    """Commercial / Cloud LLM API backend with env-var-only key loading."""

    def __init__(self, model_name: str, api_key_env: str = "LLM_API_KEY", **kwargs):
        self.model_name = model_name
        self.api_key_env = api_key_env
        self.kwargs = kwargs
        key = os.environ.get(api_key_env)
        if not key or not key.strip():
            raise ValueError(
                f"API key environment variable '{api_key_env}' is not set or empty. "
                "Set this environment variable to use APIBackend."
            )

    def generate(self, prompt: str, n: int, temperature: float, seed: int) -> List[str]:
        raise NotImplementedError("APIBackend stub: live API calls are disabled during tests.")


def get_backend(
    name: str,
    model_name: str = "mock-model",
    custom_archetypes: Optional[List[str]] = None,
) -> GenerationBackend:
    """Factory for generation backends."""
    name = name.lower()
    if name == "mock":
        return MockBackend(archetypes=custom_archetypes)
    elif name == "hf":
        return HFBackend(model_name=model_name)
    elif name == "vllm":
        return VLLMBackend(model_name=model_name)
    elif name == "api":
        return APIBackend(model_name=model_name)
    else:
        raise ValueError(f"Unknown backend '{name}'. Must be one of: mock, hf, vllm, api")


# ==============================================================================
# 3. SPLIT RESOLUTION & GRAPH DATA LOADER
# ==============================================================================

def parse_seeds_registry(seeds_path: Path = SEEDS_PATH) -> Dict[str, Dict[str, Any]]:
    """Parse SEEDS.md markdown table into dictionary."""
    if not seeds_path.is_file():
        raise FileNotFoundError(f"SEEDS.md not found at {seeds_path}")
    content = seeds_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    registry = {}
    in_table = False
    for line in lines:
        if line.startswith("|") and line.endswith("|"):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if not parts:
                continue
            if parts[0] == "name":
                in_table = True
                continue
            if in_table and all(c in "- :" for c in "".join(parts)):
                continue
            if in_table:
                name, start_s, end_s, status, purpose = parts
                registry[name] = {
                    "name": name,
                    "start": int(start_s),
                    "end": int(end_s),
                    "status": status,
                    "purpose": purpose,
                }
    return registry


def resolve_split_families_and_seeds(
    split_name: str,
    n_graphs: Optional[int] = None,
    families_filter: Optional[List[str]] = None,
    custom_split_file: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[int], str]:
    """Resolve family metadata and evaluation seeds for the given split.

    Returns:
      (families_list, seeds_list, seed_range_name)
    """
    seeds_reg = parse_seeds_registry()

    # Load all family metadata
    if not FAMILIES_PATH.is_file():
        raise FileNotFoundError(f"families.json not found at {FAMILIES_PATH}")
    with open(FAMILIES_PATH, "r", encoding="utf-8") as f:
        all_families_list = json.load(f)
    fam_map = {f["family_id"]: f for f in all_families_list}

    target_fam_ids: List[str] = []
    seed_range_name: str = split_name

    # Check Dataset v1.2 splits
    if SPLITS_V1_2_PATH.is_file():
        with open(SPLITS_V1_2_PATH, "r", encoding="utf-8") as f:
            splits_v1_2 = json.load(f).get("splits", {})
        if split_name in splits_v1_2:
            group = splits_v1_2[split_name]
            seed_range_name = group["seed_range"]
            target_fam_ids = [f["family_id"] for f in group["families"]]

    # Fallback to direct name in SEEDS.md
    if not target_fam_ids:
        if split_name in seeds_reg:
            seed_range_name = split_name
            target_fam_ids = [f["family_id"] for f in all_families_list]
        elif custom_split_file is not None and custom_split_file.is_file():
            with open(custom_split_file, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            seed_range_name = c_data.get("seed_range", "val")
            target_fam_ids = [f["family_id"] for f in c_data.get("families", [])]
        else:
            raise ValueError(
                f"Split '{split_name}' not found in Dataset v1.2 splits or SEEDS registry."
            )

    if families_filter:
        filter_set = set(families_filter)
        target_fam_ids = [fid for fid in target_fam_ids if fid in filter_set]

    if not target_fam_ids:
        raise ValueError(f"No families resolved for split '{split_name}' (filter: {families_filter})")

    resolved_families = []
    for fid in target_fam_ids:
        if fid not in fam_map:
            raise ValueError(f"Family ID '{fid}' from split not found in families.json")
        resolved_families.append(fam_map[fid])

    if seed_range_name not in seeds_reg:
        raise ValueError(f"Seed range name '{seed_range_name}' not in SEEDS registry")

    s_start = seeds_reg[seed_range_name]["start"]
    s_end = seeds_reg[seed_range_name]["end"]
    total_seeds = list(range(s_start, s_end + 1))
    if n_graphs is not None and n_graphs > 0:
        eval_seeds = total_seeds[:n_graphs]
    else:
        eval_seeds = total_seeds

    return resolved_families, eval_seeds, seed_range_name


def generate_family_graphs(
    fam_rec: Dict[str, Any],
    seeds: List[int],
) -> List[Tuple[int, np.ndarray]]:
    """Deterministically generate adjacency matrices for a family across seeds."""
    fid = fam_rec["family_id"]
    stype = fam_rec["structure_type"]
    gen = fam_rec["generator"]
    params = fam_rec.get("params", {})
    n_range = tuple(fam_rec["n_range"])
    fam_tuple = (fid, stype, gen, params, n_range)

    graphs = []
    for s in seeds:
        n, edges = make_graph(fam_tuple, s)
        adj = adj_from_edges(n, edges)
        graphs.append((s, adj))
    return graphs


# ==============================================================================
# 4. DISK CACHE & ERROR CLASSIFICATION
# ==============================================================================

def classify_sandbox_error(err_msg: Optional[str]) -> str:
    """Classify sandbox error string into canonical failure categories."""
    if not err_msg:
        return "ok"
    msg_l = err_msg.lower()
    if "ast security violation" in msg_l or "syntaxerror" in msg_l or "must define" in msg_l:
        return "ast_fail"
    if "statelessness" in msg_l or "purity" in msg_l or "mutation" in msg_l or "read-only" in msg_l:
        return "purity_fail"
    if "timeout" in msg_l:
        return "timeout"
    if "memory limit" in msg_l or "memoryerror" in msg_l:
        return "memory"
    if "verification failed" in msg_l or "invalid coloring" in msg_l:
        return "invalid_coloring"
    return "exception"


class EvaluationCache:
    """Disk cache under runs/eval/_cache/ keyed by (code_hash, graph_set_id, config_version)."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def compute_key(self, code_str: str, graph_set_id: str, config_version: str) -> str:
        code_h = hashlib.sha256(code_str.strip().encode("utf-8")).hexdigest()
        raw = f"{code_h}_{graph_set_id}_{config_version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self.cache_dir / f"{key}.json"
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.hits += 1
                return data
            except Exception:
                pass
        self.misses += 1
        return None

    def put(self, key: str, data: Dict[str, Any]):
        path = self.cache_dir / f"{key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return float(self.hits / total) if total > 0 else 0.0


# ==============================================================================
# 5. TEST-LOCK MANAGER
# ==============================================================================

class TestLockManager:
    """Test-lock for any split starting with 'test' to prevent repeated evaluation."""
    __test__ = False

    def __init__(self, lock_dir: Path):
        self.lock_dir = lock_dir
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    def is_locked(self, split: str, model_name: str, config_hash: str) -> bool:
        if not split.lower().startswith("test"):
            return False
        sanitized_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
        lock_file = self.lock_dir / f"{split}_{sanitized_model}_{config_hash}.done"
        return lock_file.is_file()

    def create_lock(
        self,
        split: str,
        model_name: str,
        config_hash: str,
        metadata: Dict[str, Any],
    ):
        if not split.lower().startswith("test"):
            return
        sanitized_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
        lock_file = self.lock_dir / f"{split}_{sanitized_model}_{config_hash}.done"
        with open(lock_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


# ==============================================================================
# 6. CANDIDATE EVALUATOR
# ==============================================================================

def evaluate_single_candidate(
    code_str: str,
    graphs: List[Tuple[int, np.ndarray]],
    probe_graphs: List[Tuple[int, np.ndarray]],
    graph_set_id: str,
    config_version: str,
    cache: EvaluationCache,
    per_graph_timeout: float = 1.0,
    overall_timeout: float = 20.0,
) -> Dict[str, Any]:
    """Evaluate candidate on graphs with disk caching and behavioral fingerprinting."""
    cache_key = cache.compute_key(code_str, graph_set_id, config_version)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    t0 = time.perf_counter()
    res = evaluate_candidate_safe(
        code_str=code_str,
        graphs=graphs,
        per_graph_timeout=per_graph_timeout,
        overall_timeout=overall_timeout,
    )
    elapsed = time.perf_counter() - t0

    is_success = bool(res.get("success", False))
    err = res.get("error")
    status = classify_sandbox_error(err)
    color_counts = res.get("color_counts", [])
    mean_colors = float(res["mean_colors"]) if res.get("mean_colors") is not None else None

    # Probe fingerprinting on seeds 9000-9009
    probe_fingerprint: Optional[str] = None
    if is_success and probe_graphs:
        p_res = evaluate_candidate_safe(
            code_str=code_str,
            graphs=probe_graphs,
            per_graph_timeout=per_graph_timeout,
            overall_timeout=overall_timeout,
        )
        if p_res.get("success"):
            probe_fingerprint = str(p_res.get("behavior_hash"))
        else:
            probe_fingerprint = "probe_fail"

    record = {
        "status": status,
        "success": is_success,
        "error": err,
        "color_counts": color_counts,
        "mean_colors": mean_colors,
        "mean_tie_break": float(res.get("mean_tie_break", 0.0)),
        "behavior_hash": res.get("behavior_hash"),
        "probe_fingerprint": probe_fingerprint,
        "runtime_s": float(elapsed),
        "graphs_count": len(graphs),
    }

    cache.put(cache_key, record)
    return record


# ==============================================================================
# 7. MAIN EVALUATION ENGINE
# ==============================================================================

def get_git_commit() -> str:
    """Retrieve current git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_library_versions() -> Dict[str, str]:
    """Retrieve installed library versions."""
    versions = {}
    for mod_name in ["numpy", "numba", "scipy", "networkx", "pandas"]:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pass
    return versions


def run_evaluation(
    backend_name: str,
    split_name: str,
    k: int = 1,
    n_samples: int = 1,
    temperature: float = 0.7,
    n_graphs: Optional[int] = None,
    run_id: str = "eval_run",
    seed: int = 42,
    force_unlock: bool = False,
    model_name: str = "mock-model",
    eps: float = 0.05,
    output_dir: Path = Path("runs/eval"),
    cache_dir: Path = Path("runs/eval/_cache"),
    lock_dir: Path = Path("runs/eval/_locks"),
    workers: int = 4,
    prompt_version: str = "v1",
    families_filter: Optional[List[str]] = None,
    per_graph_timeout: float = 1.0,
    overall_timeout: float = 20.0,
    custom_backend_archetypes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run full offline evaluation harness across families in a split."""
    run_dir = output_dir / run_id
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory '{run_dir}' already exists. Refusing to overwrite existing run_id '{run_id}'."
        )

    if k > n_samples:
        raise ValueError(f"k ({k}) cannot exceed n_samples ({n_samples})")
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got k={k}")

    # Resolve families and evaluation seeds
    families, eval_seeds, seed_range_name = resolve_split_families_and_seeds(
        split_name=split_name,
        n_graphs=n_graphs,
        families_filter=families_filter,
    )

    # Config fingerprint for test-lock
    config_dict = {
        "backend": backend_name,
        "split": split_name,
        "seed_range": seed_range_name,
        "k": k,
        "n_samples": n_samples,
        "temperature": temperature,
        "n_graphs": len(eval_seeds),
        "seed": seed,
        "model_name": model_name,
        "eps": eps,
        "prompt_version": prompt_version,
    }
    config_hash = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    # Check test-lock
    lock_mgr = TestLockManager(lock_dir)
    force_unlock_used = False
    if lock_mgr.is_locked(split_name, model_name, config_hash):
        if not force_unlock:
            raise RuntimeError(
                f"Test split '{split_name}' has already been evaluated for model '{model_name}' "
                f"(config hash: {config_hash}). Second evaluation is blocked to prevent data snooping. "
                "Pass --force-unlock to override."
            )
        else:
            force_unlock_used = True
            print(f"[WARNING] Overriding test-lock for split '{split_name}' with --force-unlock.")

    # Initialize backend and cache
    backend = get_backend(backend_name, model_name=model_name, custom_archetypes=custom_backend_archetypes)
    cache = EvaluationCache(cache_dir)
    config_version = f"v1_to{per_graph_timeout}_{overall_timeout}"

    # Pre-generate probe seeds 9000-9009
    probe_seeds = list(range(9000, 9010))

    per_card_records: List[Dict[str, Any]] = []

    for fam_idx, fam in enumerate(families):
        fid = fam["family_id"]
        card_text = fam.get("cards", {}).get("card", fid)

        # Generate split graphs and probe graphs
        split_graphs = generate_family_graphs(fam, eval_seeds)
        probe_graphs = generate_family_graphs(fam, probe_seeds)

        # Baseline evaluation on split graphs: DSATUR & RLF
        dsatur_counts = []
        dsatur_times = []
        for _, adj in split_graphs:
            t_s = time.perf_counter()
            c = count_colors(dsatur(adj))
            dsatur_times.append(time.perf_counter() - t_s)
            dsatur_counts.append(c)

        rlf_counts = [count_colors(rlf(adj)) for _, adj in split_graphs]

        dsatur_mean = float(np.mean(dsatur_counts))
        rlf_mean = float(np.mean(rlf_counts))
        rf_val = compute_rf(dsatur_mean, rlf_mean)
        t_dsatur_mean = float(np.mean(dsatur_times)) if dsatur_times else 1e-4

        # Baseline probe fingerprints (matching sandbox behavior_hash format)
        ds_first_3 = [
            hashlib.sha256(dsatur(adj).tobytes()).hexdigest()
            for _, adj in probe_graphs[:3]
        ]
        ds_probe_hash = "-".join(ds_first_3)

        rlf_first_3 = [
            hashlib.sha256(rlf(adj).tobytes()).hexdigest()
            for _, adj in probe_graphs[:3]
        ]
        rlf_probe_hash = "-".join(rlf_first_3)

        # Generate candidate code completions
        prompt = build_eval_prompt(card_text, prompt_version=prompt_version)
        card_seed = seed + fam_idx * 1000
        raw_completions = backend.generate(prompt, n=n_samples, temperature=temperature, seed=card_seed)
        candidate_codes = [extract_code(txt) for txt in raw_completions]

        # Evaluate candidates (Windows-safe ThreadPoolExecutor)
        graph_set_id = f"{fid}_{seed_range_name}_{eval_seeds[0]}_{eval_seeds[-1]}_{len(eval_seeds)}"

        eval_fn = lambda c_code: evaluate_single_candidate(
            code_str=c_code,
            graphs=split_graphs,
            probe_graphs=probe_graphs,
            graph_set_id=graph_set_id,
            config_version=config_version,
            cache=cache,
            per_graph_timeout=per_graph_timeout,
            overall_timeout=overall_timeout,
        )

        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                eval_results = list(executor.map(eval_fn, candidate_codes))
        else:
            eval_results = [eval_fn(c_code) for c_code in candidate_codes]

        # Process results for this card
        cand_records = []
        failure_breakdown = {
            "ast_fail": 0,
            "purity_fail": 0,
            "timeout": 0,
            "memory": 0,
            "invalid_coloring": 0,
            "exception": 0,
        }

        valid_colors: List[float] = []
        valid_probe_fps: List[str] = []
        runtime_ratios: List[float] = []
        code_lengths: List[int] = []
        win_flags: List[bool] = []
        baseline_like_flags: List[bool] = []
        dsatur_like_flags: List[bool] = []
        rlf_like_flags: List[bool] = []

        for sample_i, (c_code, res_item) in enumerate(zip(candidate_codes, eval_results)):
            st = res_item["status"]
            c_len = len(c_code)
            code_lengths.append(c_len)

            if st != "ok":
                failure_breakdown[st] = failure_breakdown.get(st, 0) + 1
                cand_records.append({
                    "sample_idx": sample_i,
                    "status": st,
                    "mean_colors": None,
                    "savings_rf": None,
                    "savings_rf_pct": None,
                    "savings_dsatur": None,
                    "savings_dsatur_pct": None,
                    "beats_rf": False,
                    "probe_fingerprint": None,
                    "runtime_ratio": None,
                    "code_length": c_len,
                    "error": res_item.get("error"),
                })
                win_flags.append(False)
                dsatur_like_flags.append(False)
                rlf_like_flags.append(False)
                baseline_like_flags.append(False)
            else:
                m_c = float(res_item["mean_colors"])
                valid_colors.append(m_c)
                sav_rf = rf_val - m_c
                sav_rf_pct = (sav_rf / rf_val * 100.0) if rf_val > 0.0 else 0.0
                sav_ds = dsatur_mean - m_c
                sav_ds_pct = (sav_ds / dsatur_mean * 100.0) if dsatur_mean > 0.0 else 0.0
                is_win = beats_rf(m_c, rf_val, eps=eps)
                win_flags.append(is_win)

                # Runtime ratio
                per_graph_rt = res_item["runtime_s"] / max(len(split_graphs), 1)
                rt_ratio = per_graph_rt / max(t_dsatur_mean, 1e-6)
                runtime_ratios.append(rt_ratio)

                # Probe fingerprint
                p_fp = res_item.get("probe_fingerprint")
                if p_fp:
                    valid_probe_fps.append(p_fp)
                # Check baseline similarity via behavioral hash
                b_hash = res_item.get("behavior_hash")
                is_ds = (p_fp == ds_probe_hash) or (b_hash and b_hash.startswith(ds_probe_hash[:8]))
                is_rlf = (p_fp == rlf_probe_hash) or (b_hash and b_hash.startswith(rlf_probe_hash[:8]))
                dsatur_like_flags.append(is_ds)
                rlf_like_flags.append(is_rlf)
                baseline_like_flags.append(is_ds or is_rlf)

                cand_records.append({
                    "sample_idx": sample_i,
                    "status": "ok",
                    "mean_colors": m_c,
                    "savings_rf": sav_rf,
                    "savings_rf_pct": sav_rf_pct,
                    "savings_dsatur": sav_ds,
                    "savings_dsatur_pct": sav_ds_pct,
                    "beats_rf": is_win,
                    "probe_fingerprint": p_fp,
                    "runtime_ratio": rt_ratio,
                    "code_length": c_len,
                    "error": None,
                })

        valid_count = len(valid_colors)
        valid_rate = float(valid_count / n_samples)
        card_mean_colors = float(np.mean(valid_colors)) if valid_colors else None
        c_wins = sum(1 for w in win_flags if w)

        # Best-of-k for this card
        cand_colors_first_k = [cand_records[j]["mean_colors"] for j in range(min(k, len(cand_records)))]
        valid_first_k = [c for c in cand_colors_first_k if c is not None]
        if valid_first_k:
            best_k_c = float(min(valid_first_k))
            best_k_sav_rf = rf_val - best_k_c
            best_k_sav_rf_pct = (best_k_sav_rf / rf_val * 100.0) if rf_val > 0.0 else 0.0
            best_k_sav_ds = dsatur_mean - best_k_c
            best_k_sav_ds_pct = (best_k_sav_ds / dsatur_mean * 100.0) if dsatur_mean > 0.0 else 0.0
        else:
            best_k_c = None
            best_k_sav_rf = 0.0
            best_k_sav_rf_pct = 0.0
            best_k_sav_ds = 0.0
            best_k_sav_ds_pct = 0.0

        p_at_k = pass_at_k(n_samples, c_wins, k) if k <= n_samples else 0.0
        cov_at_k = 1.0 if any(win_flags[:k]) else 0.0
        distinct_fps = len(set(valid_probe_fps))

        per_card_records.append({
            "family_id": fid,
            "card": card_text,
            "n_samples": n_samples,
            "k": k,
            "valid_candidates": valid_count,
            "valid_rate": valid_rate,
            "failure_breakdown": failure_breakdown,
            "dsatur_mean": dsatur_mean,
            "rlf_mean": rlf_mean,
            "r_f": rf_val,
            "mean_colors": card_mean_colors,
            "mean_savings_rf": (rf_val - card_mean_colors) if card_mean_colors is not None else 0.0,
            "mean_savings_rf_pct": ((rf_val - card_mean_colors) / rf_val * 100.0) if (card_mean_colors is not None and rf_val > 0.0) else 0.0,
            "best_of_k_colors": best_k_c,
            "best_of_k_savings_rf": best_k_sav_rf,
            "best_of_k_savings_rf_pct": best_k_sav_rf_pct,
            "best_of_k_savings_dsatur": best_k_sav_ds,
            "best_of_k_savings_dsatur_pct": best_k_sav_ds_pct,
            "coverage_at_k": cov_at_k,
            "pass_at_k": p_at_k,
            "mean_runtime_ratio_vs_dsatur": float(np.mean(runtime_ratios)) if runtime_ratios else None,
            "mean_code_length": float(np.mean(code_lengths)) if code_lengths else 0.0,
            "frac_baseline_like": float(np.mean(baseline_like_flags)) if baseline_like_flags else 0.0,
            "frac_dsatur_like": float(np.mean(dsatur_like_flags)) if dsatur_like_flags else 0.0,
            "frac_rlf_like": float(np.mean(rlf_like_flags)) if rlf_like_flags else 0.0,
            "distinct_fingerprints": distinct_fps,
            "candidates": cand_records,
        })

    # Summary aggregations across cards
    total_cards = len(per_card_records)
    total_failures = {
        "ast_fail": sum(r["failure_breakdown"]["ast_fail"] for r in per_card_records),
        "purity_fail": sum(r["failure_breakdown"]["purity_fail"] for r in per_card_records),
        "timeout": sum(r["failure_breakdown"]["timeout"] for r in per_card_records),
        "memory": sum(r["failure_breakdown"]["memory"] for r in per_card_records),
        "invalid_coloring": sum(r["failure_breakdown"]["invalid_coloring"] for r in per_card_records),
        "exception": sum(r["failure_breakdown"]["exception"] for r in per_card_records),
    }

    mean_valid_rate = float(np.mean([r["valid_rate"] for r in per_card_records])) if total_cards > 0 else 0.0
    valid_card_colors = [r["mean_colors"] for r in per_card_records if r["mean_colors"] is not None]
    overall_mean_colors = float(np.mean(valid_card_colors)) if valid_card_colors else None

    # Coverage@k using pure eval.coverage
    card_win_counts = [sum(1 for c in r["candidates"] if c["beats_rf"]) for r in per_card_records]
    expected_coverage_k = coverage_at_k(card_win_counts, n_samples=n_samples, k=k) if k <= n_samples else 0.0

    # Best-of-k across cards using pure eval.coverage
    card_candidate_colors = [
        [c["mean_colors"] for c in r["candidates"]] for r in per_card_records
    ]
    rf_colors = [r["r_f"] for r in per_card_records]
    bok_metrics = best_of_k(card_candidate_colors, rf_colors, k=k)

    # Runtime ratio average
    r_ratios = [r["mean_runtime_ratio_vs_dsatur"] for r in per_card_records if r["mean_runtime_ratio_vs_dsatur"] is not None]
    mean_rt_ratio = float(np.mean(r_ratios)) if r_ratios else None

    summary_data = {
        "run_id": run_id,
        "split": split_name,
        "seed_range": seed_range_name,
        "backend": backend_name,
        "model_name": model_name,
        "k": k,
        "n_samples": n_samples,
        "temperature": temperature,
        "seed": seed,
        "total_cards": total_cards,
        "valid_rate": mean_valid_rate,
        "failure_counts": total_failures,
        "mean_colors": overall_mean_colors,
        "mean_savings_rf_colors": float(np.mean([r["mean_savings_rf"] for r in per_card_records])) if total_cards > 0 else 0.0,
        "mean_savings_rf_pct": float(np.mean([r["mean_savings_rf_pct"] for r in per_card_records])) if total_cards > 0 else 0.0,
        "coverage_at_k": expected_coverage_k,
        "mean_pass_at_k": float(np.mean([r["pass_at_k"] for r in per_card_records])) if total_cards > 0 else 0.0,
        "best_of_k_colors": bok_metrics["mean_best_colors"],
        "best_of_k_savings_rf_colors": bok_metrics["mean_savings_rf"],
        "best_of_k_savings_rf_pct": bok_metrics["mean_savings_rf_pct"],
        "card_success_rate": bok_metrics["card_success_rate"],
        "mean_runtime_ratio_vs_dsatur": mean_rt_ratio,
        "mean_code_length": float(np.mean([r["mean_code_length"] for r in per_card_records])) if total_cards > 0 else 0.0,
        "frac_baseline_like": float(np.mean([r["frac_baseline_like"] for r in per_card_records])) if total_cards > 0 else 0.0,
        "frac_dsatur_like": float(np.mean([r["frac_dsatur_like"] for r in per_card_records])) if total_cards > 0 else 0.0,
        "frac_rlf_like": float(np.mean([r["frac_rlf_like"] for r in per_card_records])) if total_cards > 0 else 0.0,
        "mean_distinct_fingerprints": float(np.mean([r["distinct_fingerprints"] for r in per_card_records])) if total_cards > 0 else 0.0,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "cache_hit_rate": cache.hit_rate,
        "force_unlock_used": force_unlock_used,
    }

    config_data = {
        "run_id": run_id,
        "backend": backend_name,
        "model_name": model_name,
        "split": split_name,
        "seed_range": seed_range_name,
        "k": k,
        "n_samples": n_samples,
        "temperature": temperature,
        "n_graphs": len(eval_seeds),
        "seed": seed,
        "eps": eps,
        "prompt_version": prompt_version,
        "force_unlock_used": force_unlock_used,
        "config_hash": config_hash,
        "git_commit": get_git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "library_versions": get_library_versions(),
    }

    # Write output artifacts
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "per_card.jsonl", "w", encoding="utf-8") as f:
        for r in per_card_records:
            f.write(json.dumps(r) + "\n")

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    # Create lock file on completion for test splits
    if split_name.lower().startswith("test"):
        lock_mgr.create_lock(
            split=split_name,
            model_name=model_name,
            config_hash=config_hash,
            metadata={
                "run_id": run_id,
                "split": split_name,
                "model_name": model_name,
                "config_hash": config_hash,
                "timestamp": config_data["timestamp"],
                "git_commit": config_data["git_commit"],
            },
        )

    return {
        "run_dir": str(run_dir),
        "summary": summary_data,
        "config": config_data,
    }


# ==============================================================================
# 8. CLI ENTRYPOINT
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline evaluation harness for priority heuristics")
    parser.add_argument("--backend", type=str, choices=["mock", "hf", "vllm", "api"], default="mock")
    parser.add_argument("--split", type=str, required=True, help="Dataset split name (e.g. val, test_pilot, test_interp)")
    parser.add_argument("--k", type=int, default=1, help="Metric evaluation budget k")
    parser.add_argument("--n-samples", type=int, default=1, help="Number of samples N generated per card (N >= K)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--n-graphs", type=int, default=None, help="Number of graphs per family")
    parser.add_argument("--run-id", type=str, required=True, help="Unique run identifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force-unlock", action="store_true", help="Force evaluation override for test split")
    parser.add_argument("--model-name", type=str, default="mock-model", help="Model name or checkpoint identifier")
    parser.add_argument("--eps", type=float, default=0.05, help="Canonical margin eps for beating R(F)")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/eval"), help="Base output directory")
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/eval/_cache"), help="Disk cache directory")
    parser.add_argument("--lock-dir", type=Path, default=Path("runs/eval/_locks"), help="Test locks directory")
    parser.add_argument("--workers", type=int, default=4, help="Worker threads for sandboxed evaluation")
    parser.add_argument("--prompt-version", type=str, default="v1", help="Prompt template version")
    parser.add_argument("--families", type=str, default=None, help="Comma-separated list of family IDs")
    parser.add_argument("--per-graph-timeout", type=float, default=1.0, help="Per-graph evaluation timeout in seconds")
    parser.add_argument("--overall-timeout", type=float, default=20.0, help="Overall candidate timeout in seconds")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    fam_filter = [f.strip() for f in args.families.split(",")] if args.families else None

    try:
        res = run_evaluation(
            backend_name=args.backend,
            split_name=args.split,
            k=args.k,
            n_samples=args.n_samples,
            temperature=args.temperature,
            n_graphs=args.n_graphs,
            run_id=args.run_id,
            seed=args.seed,
            force_unlock=args.force_unlock,
            model_name=args.model_name,
            eps=args.eps,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            lock_dir=args.lock_dir,
            workers=args.workers,
            prompt_version=args.prompt_version,
            families_filter=fam_filter,
            per_graph_timeout=args.per_graph_timeout,
            overall_timeout=args.overall_timeout,
        )
        print(f"Evaluation finished successfully -> {res['run_dir']}")
        print(f"Valid rate: {res['summary']['valid_rate']:.3f} | Coverage@{args.k}: {res['summary']['coverage_at_k']:.3f}")
        print(f"Cache hit rate: {res['summary']['cache_hit_rate']:.3f}")
    except Exception as e:
        print(f"Evaluation error: {type(e).__name__}: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
