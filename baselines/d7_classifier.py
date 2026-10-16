"""Diagnostic Baseline D7: Card-based Algorithm Selector {DSATUR, RLF}.

Implements Task 9.1 from PLAN_PHASE6_ONWARD.md:
- Pure NumPy / SciPy classifiers (Logistic Regression & Decision Tree, no sklearn).
- Trained strictly on train_pool from data/data_v1_2/splits.json.
- Evaluated across all splits: train_pool, val, test_interp, test_extrap, test_comp, heldout_type.
- Reports prediction accuracy and regret vs R(F) = min(DSATUR, RLF).
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import scipy.optimize as opt

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gate.dataset import load_family
from evaluator.coloring import dsatur, rlf
from evaluator.verify import count_colors


# ==============================================================================
# PURE NUMPY / SCIPY CLASSIFIERS (NO SKLEARN DEPENDENCY)
# ==============================================================================

class PureLogisticRegression:
    """L2-regularized Logistic Regression solved via L-BFGS-B."""
    def __init__(self, l2_reg: float = 1.0):
        self.l2_reg = l2_reg
        self.weights = None
        self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        y_signed = np.where(y > 0, 1.0, -1.0)

        def loss_and_grad(params):
            w = params[:-1]
            b = params[-1]
            margins = y_signed * (X @ w + b)
            loss = np.sum(np.logaddexp(0.0, -margins)) + 0.5 * self.l2_reg * np.sum(w ** 2)
            sig = 1.0 / (1.0 + np.exp(np.clip(margins, -30, 30)))
            grad_w = -X.T @ (y_signed * sig) + self.l2_reg * w
            grad_b = -np.sum(y_signed * sig)
            return loss, np.append(grad_w, grad_b)

        init_params = np.zeros(n_features + 1)
        res = opt.minimize(loss_and_grad, init_params, jac=True, method="L-BFGS-B")
        self.weights = res.x[:-1]
        self.bias = res.x[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.weights + self.bias
        return np.where(logits >= 0.0, 1, 0)


class PureDecisionTree:
    """Binary Decision Tree classifier with Gini impurity."""
    def __init__(self, max_depth: int = 2):
        self.max_depth = max_depth
        self.tree = None

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None, None, 0.0

        p1 = np.mean(y)
        base_gini = 1.0 - (p1 ** 2 + (1.0 - p1) ** 2)
        best_gain = -1.0
        best_feat = None
        best_thresh = None

        for j in range(n_features):
            vals = np.unique(X[:, j])
            if len(vals) <= 1:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            for th in thresholds:
                left_mask = X[:, j] <= th
                n_l = np.sum(left_mask)
                n_r = n_samples - n_l
                if n_l == 0 or n_r == 0:
                    continue
                p1_l = np.mean(y[left_mask])
                p1_r = np.mean(y[~left_mask])
                gini_l = 1.0 - (p1_l ** 2 + (1.0 - p1_l) ** 2)
                gini_r = 1.0 - (p1_r ** 2 + (1.0 - p1_r) ** 2)
                weighted_gini = (n_l * gini_l + n_r * gini_r) / n_samples
                gain = base_gini - weighted_gini
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = th

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        pred = 1 if np.mean(y) >= 0.5 else 0
        node = {"leaf": True, "pred": pred}
        if depth >= self.max_depth or len(np.unique(y)) <= 1:
            return node

        feat, thresh, gain = self._best_split(X, y)
        if feat is None or gain <= 1e-7:
            return node

        left_mask = X[:, feat] <= thresh
        return {
            "leaf": False,
            "feature": feat,
            "threshold": thresh,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.tree = self._build_tree(X, y, depth=0)

    def _predict_row(self, row: np.ndarray, node: dict) -> int:
        if node["leaf"]:
            return node["pred"]
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_row(row, node["left"])
        else:
            return self._predict_row(row, node["right"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(row, self.tree) for row in X])


# ==============================================================================
# FEATURE EXTRACTION & PIPELINE
# ==============================================================================

def extract_card_features(fam_record: dict) -> np.ndarray:
    """Extract numeric card features: [density, deg_cv, clustering, normalized_n]."""
    stats = fam_record.get("probe_stats", {})
    nr = fam_record.get("n_range", [100, 200])
    n_mean = float(np.mean(nr))

    density = float(stats.get("density", 0.2))
    deg_cv = float(stats.get("deg_cv", 0.3))
    clustering = float(stats.get("clustering", 0.1))
    norm_n = n_mean / 300.0

    return np.array([density, deg_cv, clustering, norm_n], dtype=np.float64)


def get_family_performance(family_id: str, split_name: str) -> Tuple[float, float]:
    """Compute mean color counts of DSATUR and RLF on specified split graphs."""
    # Map split_name to load_family split parameter
    split_map = {
        "train_pool": "baseline_train",
        "val": "val",
        "test_interp": "test_interp",
        "test_extrap": "test_extrap",
        "test_comp": "test_comp",
        "heldout_type": "test_heldout_type",
    }
    actual_split = split_map.get(split_name, split_name)
    graphs = load_family(family_id, actual_split)

    ds_counts = [count_colors(dsatur(adj)) for _, adj in graphs]
    rlf_counts = [count_colors(rlf(adj)) for _, adj in graphs]

    return float(np.mean(ds_counts)), float(np.mean(rlf_counts))


def train_d7_classifiers(
    splits_path: str = "data/data_v1_2/splits.json",
    families_path: str = "data/data_v1_1/families_v1_1.json",
) -> Tuple[PureLogisticRegression, PureDecisionTree, Dict[str, Any]]:
    """Train D7 classifiers strictly on train_pool families."""
    with open(splits_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)
    with open(families_path, "r", encoding="utf-8") as f:
        fams_list = json.load(f)

    fam_map = {f["family_id"]: f for f in fams_list}
    train_pool_records = splits_data["splits"]["train_pool"]["families"]

    X_train = []
    y_train = []  # 1 = choose RLF, 0 = choose DSATUR

    for item in train_pool_records:
        fid = item["family_id"]
        fam_rec = fam_map[fid]
        feats = extract_card_features(fam_rec)
        X_train.append(feats)

        # Labels from baseline_train performance
        c_ds, c_rlf = get_family_performance(fid, "train_pool")
        y_train.append(1 if c_rlf < c_ds else 0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    lr_model = PureLogisticRegression(l2_reg=1.0)
    lr_model.fit(X_train, y_train)

    dt_model = PureDecisionTree(max_depth=2)
    dt_model.fit(X_train, y_train)

    metadata = {
        "n_train_families": len(train_pool_records),
        "rlf_wins_train": int(np.sum(y_train)),
        "dsatur_wins_train": int(len(y_train) - np.sum(y_train)),
    }
    return lr_model, dt_model, metadata


def evaluate_split(
    model: Any,
    split_name: str,
    splits_data: dict,
    fam_map: dict,
) -> Dict[str, Any]:
    """Evaluate classifier on a specific split."""
    fam_items = splits_data["splits"][split_name]["families"]
    total_fams = len(fam_items)

    correct_picks = 0
    total_regret_colors = 0.0
    total_rf_colors = 0.0

    always_rlf_regret = 0.0
    always_dsatur_regret = 0.0

    family_details = []

    for item in fam_items:
        fid = item["family_id"]
        fam_rec = fam_map[fid]
        feats = extract_card_features(fam_rec).reshape(1, -1)

        pred = int(model.predict(feats)[0])  # 1 = RLF, 0 = DSATUR

        c_ds, c_rlf = get_family_performance(fid, split_name)
        r_f = min(c_ds, c_rlf)

        chosen_c = c_rlf if pred == 1 else c_ds
        regret = chosen_c - r_f

        if chosen_c <= r_f + 1e-6:
            correct_picks += 1

        total_regret_colors += regret
        total_rf_colors += r_f

        always_rlf_regret += (c_rlf - r_f)
        always_dsatur_regret += (c_ds - r_f)

        family_details.append({
            "family_id": fid,
            "prediction": "RLF" if pred == 1 else "DSATUR",
            "c_dsatur": c_ds,
            "c_rlf": c_rlf,
            "r_f": r_f,
            "chosen_colors": chosen_c,
            "regret": regret,
        })

    accuracy = correct_picks / total_fams if total_fams > 0 else 0.0
    mean_regret = total_regret_colors / total_fams if total_fams > 0 else 0.0
    regret_pct = (total_regret_colors / total_rf_colors * 100.0) if total_rf_colors > 0 else 0.0

    mean_rlf_regret = always_rlf_regret / total_fams if total_fams > 0 else 0.0
    mean_ds_regret = always_dsatur_regret / total_fams if total_fams > 0 else 0.0

    return {
        "split": split_name,
        "n_families": total_fams,
        "accuracy": accuracy,
        "mean_regret_colors": mean_regret,
        "regret_pct": regret_pct,
        "mean_always_rlf_regret": mean_rlf_regret,
        "mean_always_dsatur_regret": mean_ds_regret,
        "details": family_details,
    }


def run_d7_benchmark(
    splits_path: str = "data/data_v1_2/splits.json",
    families_path: str = "data/data_v1_1/families_v1_1.json",
    out_dir: str = "runs/baselines/d7",
) -> Dict[str, Any]:
    """Execute complete D7 benchmark across all splits."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(splits_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)
    with open(families_path, "r", encoding="utf-8") as f:
        fams_list = json.load(f)
    fam_map = {f["family_id"]: f for f in fams_list}

    lr_model, dt_model, train_meta = train_d7_classifiers(splits_path, families_path)

    split_names = ["train_pool", "val", "test_interp", "test_extrap", "test_comp", "heldout_type"]
    dt_results = {}
    lr_results = {}

    for sname in split_names:
        dt_results[sname] = evaluate_split(dt_model, sname, splits_data, fam_map)
        lr_results[sname] = evaluate_split(lr_model, sname, splits_data, fam_map)

    # Save results
    summary = {
        "train_metadata": train_meta,
        "decision_tree": {k: {k2: v2 for k2, v2 in v.items() if k2 != "details"} for k, v in dt_results.items()},
        "logistic_regression": {k: {k2: v2 for k2, v2 in v.items() if k2 != "details"} for k, v in lr_results.items()},
    }

    with open(out_path / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Generate Markdown Table
    md_lines = [
        "# D7 Card-Based Algorithm Selector Baseline ({DSATUR, RLF})",
        "",
        f"Trained strictly on `train_pool` ({train_meta['n_train_families']} families).",
        "",
        "## Decision Tree Evaluation across Splits",
        "| Split | N Families | Accuracy | Mean Regret (colors) | Regret (%) | Always RLF Regret | Always DSATUR Regret |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for sname in split_names:
        res = dt_results[sname]
        md_lines.append(
            f"| `{sname}` | {res['n_families']} | {res['accuracy']*100:.1f}% | "
            f"{res['mean_regret_colors']:.3f} | {res['regret_pct']:.2f}% | "
            f"{res['mean_always_rlf_regret']:.3f} | {res['mean_always_dsatur_regret']:.3f} |"
        )

    md_lines.extend([
        "",
        "## Logistic Regression Evaluation across Splits",
        "| Split | N Families | Accuracy | Mean Regret (colors) | Regret (%) | Always RLF Regret | Always DSATUR Regret |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for sname in split_names:
        res = lr_results[sname]
        md_lines.append(
            f"| `{sname}` | {res['n_families']} | {res['accuracy']*100:.1f}% | "
            f"{res['mean_regret_colors']:.3f} | {res['regret_pct']:.2f}% | "
            f"{res['mean_always_rlf_regret']:.3f} | {res['mean_always_dsatur_regret']:.3f} |"
        )

    md_content = "\n".join(md_lines) + "\n"
    with open(out_path / "summary.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print(md_content)
    return summary


def main():
    parser = argparse.ArgumentParser(description="D7 Baseline Algorithm Selector Runner")
    parser.add_argument("--splits-path", default="data/data_v1_2/splits.json", help="Path to splits.json")
    parser.add_argument("--families-path", default="data/data_v1_1/families_v1_1.json", help="Path to families metadata")
    parser.add_argument("--out-dir", default="runs/baselines/d7", help="Output directory")
    args = parser.parse_args()

    run_d7_benchmark(
        splits_path=args.splits_path,
        families_path=args.families_path,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
