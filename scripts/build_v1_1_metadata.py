"""Build Dataset v1.1 metadata files without altering frozen data/data_v1/.

Creates:
- data/data_v1_1/families_v1_1.json
- data/data_v1_1/config.json
"""

import json
from pathlib import Path
from typing import Dict, Any


def format_card_detailed(fam: Dict[str, Any]) -> str:
    """Format card_detailed incorporating exact generator parameters."""
    stype = fam["structure_type"]
    gen = fam["generator"]
    params = fam.get("params", {})
    nr = fam["n_range"]
    lo, hi = nr[0], nr[1]
    stats = fam.get("probe_stats", {})
    d = stats.get("density", 0.0)
    card_base = fam["cards"]["card"]

    # Parameter description strings
    if gen == "gnp":
        p = params.get("p", d)
        param_desc = f"Erdos-Renyi G(n, p) with edge probability p={p:g}"
    elif gen == "geometric":
        r = params.get("radius", 0.0)
        noise = params.get("noise", 0.0)
        if noise > 0:
            param_desc = f"hybrid 2D geometric graph with connection radius r={r:.2f} and random edge noise q={noise:.2f}"
        else:
            param_desc = f"2D random geometric graph with connection radius r={r:.2f}"
    elif gen == "ba":
        m = params.get("m", 2)
        param_desc = f"Barabasi-Albert scale-free graph with preferential attachment m={m} edges per new vertex"
    elif gen == "plc":
        m = params.get("m", 2)
        p = params.get("p", 0.0)
        param_desc = f"Holme-Kim powerlaw-cluster graph with m={m} edges added per step and triangle probability p={p:g}"
    elif gen == "sbm":
        k = params.get("k", 2)
        p_in = params.get("p_in", 0.0)
        p_out = params.get("p_out", 0.0)
        param_desc = f"stochastic block model with k={k} communities, intra-community edge prob p_in={p_in:g}, and inter-community edge prob p_out={p_out:g}"
    elif gen == "ws":
        k = params.get("k", 4)
        p = params.get("p", 0.0)
        param_desc = f"Watts-Strogatz small-world graph with initial ring degree k={k} and rewiring probability p={p:g}"
    elif gen == "caveman":
        k = params.get("k", 5)
        p = params.get("p", 0.0)
        param_desc = f"relaxed caveman graph with clique size k={k} and inter-clique rewiring probability p={p:g}"
    elif gen == "interval":
        L = params.get("L", 0.1)
        param_desc = f"1D random interval graph with maximum interval half-length L={L:g}"
    elif gen == "queen":
        keep = params.get("keep", 1.0)
        if keep < 1.0:
            param_desc = f"queen graph on board with randomly sampled square fraction keep={keep:g}"
        else:
            param_desc = "queen graph on full k x k board with all squares present (keep=1.0)"
    elif gen == "mycielski":
        t = params.get("t", 1)
        p = params.get("p", 0.1)
        param_desc = f"Mycielskian graph constructed from base G(n0, p={p:g}) over t={t} iterations"
    elif gen == "kpartite":
        k = params.get("k", 2)
        p = params.get("p", 0.5)
        param_desc = f"equipartite k-partite graph with k={k} balanced partitions and cross-partition edge prob p={p:g}"
    else:
        param_desc = f"generator={gen}, params={params}"

    return f"{card_base}, generator: {param_desc}"


def main():
    src_families_path = Path("data/data_v1/families.json")
    src_config_path = Path("data/data_v1/config.json")
    out_dir = Path("data/data_v1_1")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(src_families_path, "r", encoding="utf-8") as f:
        fams = json.load(f)

    with open(src_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 1. Update scale_free from heldout_type to train_pool
    # 2. Update small_world from train_pool to heldout_type (keep heldout_size unchanged)
    # 3. Add card_detailed to cards dict for all families
    sf_count = 0
    sw_count = 0
    for fam in fams:
        stype = fam.get("structure_type")
        pool = fam.get("pool")

        if stype == "scale_free" and pool == "heldout_type":
            fam["pool"] = "train_pool"
            sf_count += 1
        elif stype == "small_world" and pool == "train_pool":
            fam["pool"] = "heldout_type"
            sw_count += 1

        # Add card_detailed
        fam["cards"]["card_detailed"] = format_card_detailed(fam)

    print(f"Updated {sf_count} scale_free families from heldout_type to train_pool.")
    print(f"Updated {sw_count} small_world families from train_pool to heldout_type (2 heldout_size kept unchanged).")
    print(f"Added card_detailed to all {len(fams)} families.")

    # Save families_v1_1.json
    out_families_path = out_dir / "families_v1_1.json"
    with open(out_families_path, "w", encoding="utf-8") as f:
        json.dump(fams, f, indent=1)
    print(f"Saved {out_families_path}")

    # 4. Update config with teacher_val split and new heldout_types
    config["version"] = "v1.1"
    config["splits"]["teacher_val"] = [3000, 20]
    config["counts"]["teacher_val"] = 20

    # Frozen heldout types for v1.1
    config["heldout_types"] = ["geometric", "hybrid_geometric", "queen", "small_world"]

    out_config_path = out_dir / "config.json"
    with open(out_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=1)
    print(f"Saved {out_config_path}")


if __name__ == "__main__":
    main()
