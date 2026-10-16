"""Dataset loader for frozen dataset v1.0.

Provides:
- load_family(family_id, split, check_sha=False) -> list[(seed, adj)]
- families(pool=None) -> list[dict]
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
import numpy as np

DATA_DIR = Path("data/data_v1")


class DatasetManifest:
    """Singleton/cached loader for manifest.jsonl and families.json."""
    _instance = None

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.families_path = data_dir / "families.json"
        self.manifest_path = data_dir / "manifest.jsonl"
        self._families: Optional[List[Dict[str, Any]]] = None
        self._manifest_index: Optional[Dict[Tuple[str, str, int], Dict[str, Any]]] = None
        self._family_split_seeds: Optional[Dict[Tuple[str, str], List[Dict[str, Any]]]] = None

    def _load(self):
        if self._families is None:
            with open(self.families_path, "r", encoding="utf-8") as f:
                self._families = json.load(f)

        if self._manifest_index is None:
            self._manifest_index = {}
            self._family_split_seeds = {}
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    fid = row["family_id"]
                    sp = row["split"]
                    sd = row["seed"]
                    self._manifest_index[(fid, sp, sd)] = row
                    key = (fid, sp)
                    if key not in self._family_split_seeds:
                        self._family_split_seeds[key] = []
                    self._family_split_seeds[key].append(row)

    @property
    def all_families(self) -> List[Dict[str, Any]]:
        self._load()
        return self._families  # type: ignore

    def get_manifest_rows(self, family_id: str, split: str) -> List[Dict[str, Any]]:
        self._load()
        actual_split = "train" if split == "baseline_train" else split
        rows = self._family_split_seeds.get((family_id, actual_split), [])  # type: ignore
        if split == "baseline_train":
            rows = [r for r in rows if r.get("baseline_train", False)]
        return rows


_MANIFEST = DatasetManifest()


def families(pool: Optional[str] = None, data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return family records from families.json, optionally filtered by pool.

    Parameters:
    pool: e.g. 'train_pool', 'heldout_type', 'heldout_size', 'heldout_combo'.
    data_dir: optional custom data directory.

    Returns:
    List of family dictionaries.
    """
    manifest = DatasetManifest(data_dir) if data_dir is not None else _MANIFEST
    all_fams = manifest.all_families
    if pool is None:
        return all_fams
    return [f for f in all_fams if f.get("pool") == pool]


def load_family(
    family_id: str,
    split: str,
    check_sha: bool = False,
    data_dir: Optional[Path] = None,
) -> List[Tuple[int, np.ndarray]]:
    """Load graphs for a given family and split.

    Parameters:
    family_id: string family identifier, e.g. 'gnp_d0.05_n200-300'.
    split: 'train', 'baseline_train', 'val', 'test', or 'probe'.
    check_sha: if True, validates sha256 of each graph against the manifest.
    data_dir: optional path to dataset root (defaults to data/data_v1).

    Returns:
    list of (seed: int, adj: np.ndarray bool (n, n)).
    """
    root = data_dir if data_dir is not None else DATA_DIR
    manifest = DatasetManifest(root) if data_dir is not None else _MANIFEST

    GENERATOR_SPLITS = {
        "teacher_val": (3000, 20),
        "test_pilot": (4000, 200),
        "sft_val_probe": (5000, 20),
        "teacher_rescore": (6000, 50),
        "test_interp": (7000, 50),
        "test_extrap": (7100, 50),
        "test_comp": (7200, 50),
        "test_heldout_type": (7300, 50),
        "probe": (9000, 10),
    }

    if split in GENERATOR_SPLITS:
        from data.generate_dataset import make_graph, adj_from_edges
        fam_rec = next((f for f in manifest.all_families if f["family_id"] == family_id), None)
        if fam_rec is None:
            raise ValueError(f"Family '{family_id}' not found in manifest")
        start, count = GENERATOR_SPLITS[split]
        fam_tuple = (
            family_id,
            fam_rec["structure_type"],
            fam_rec["generator"],
            fam_rec.get("params", {}),
            tuple(fam_rec["n_range"]),
        )
        graphs = []
        for s in range(start, start + count):
            n, edges = make_graph(fam_tuple, s)
            adj = adj_from_edges(n, edges)
            graphs.append((s, adj))
        return graphs

    actual_split = "train" if split == "baseline_train" else split
    npz_path = root / "graphs" / family_id / f"{actual_split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Graph file not found: {npz_path}")

    rows = manifest.get_manifest_rows(family_id, split)
    if not rows:
        raise ValueError(f"No manifest entries found for family '{family_id}' and split '{split}'")

    # Load npz archive
    archive = np.load(npz_path)

    graphs: List[Tuple[int, np.ndarray]] = []
    for r in rows:
        seed = r["seed"]
        key = r["key"]
        n = r["n"]
        edges = archive[key]  # shape (m, 2), int32

        if check_sha:
            computed_sha = hashlib.sha256(
                np.int32(n).tobytes() + edges.astype(np.int32).tobytes()
            ).hexdigest()
            if computed_sha != r["sha256"]:
                raise ValueError(
                    f"SHA256 mismatch for family {family_id}, seed {seed}: "
                    f"computed {computed_sha} != expected {r['sha256']}"
                )

        adj = np.zeros((n, n), dtype=bool)
        if len(edges) > 0:
            adj[edges[:, 0], edges[:, 1]] = True
            adj[edges[:, 1], edges[:, 0]] = True

        graphs.append((seed, adj))

    return graphs
