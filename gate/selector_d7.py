"""Diagnostic D7: Card-based Selector {DSATUR, RLF}.

Implements:
1. Feature extraction from family cards / probe stats:
   - structure_type (11 categories)
   - density
   - degree coefficient of variation (deg_cv)
   - clustering coefficient
   - average graph size n
2. Classifiers (Pure NumPy / SciPy, no external sklearn dependency):
   - Logistic Regression (L2-regularized with scipy.optimize.minimize)
   - Decision Tree (CART depth 1..3 with Gini impurity / best split search)
3. Cross-Validation Schemes:
   - CV by Family: Leave-One-Out Cross-Validation (LOOCV, 106 folds)
   - CV by Structure Type: Leave-One-Group-Out Cross-Validation (LOGO-CV, 11 folds)
4. Metrics:
   - Prediction Accuracy (% times selector picks the better algorithm)
   - Regret vs R(F) (in colors and %)
   - Comparison against "Always RLF" and "Always DSATUR" baselines

Produces:
- results/d7_selector_results.csv
- results/d7_summary.txt
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import scipy.optimize as opt
import pandas as pd

from gate.dataset import families


# ==============================================================================
# PURE NUMPY / SCIPY CLASSIFIERS
# ==============================================================================

class PureLogisticRegression:
    def __init__(self, l2_reg: float = 1.0):
        self.l2_reg = l2_reg
        self.weights = None
        self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        # Target: -1, +1 for logistic loss
        y_signed = np.where(y > 0, 1.0, -1.0)

        def loss_and_grad(params):
            w = params[:-1]
            b = params[-1]
            margins = y_signed * (X @ w + b)
            # log(1 + exp(-margins))
            # stable log1pexp
            loss = np.sum(np.logaddexp(0.0, -margins)) + 0.5 * self.l2_reg * np.sum(w ** 2)
            # grad
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
                right_mask = ~left_mask

                n_l = np.sum(left_mask)
                n_r = np.sum(right_mask)
                if n_l == 0 or n_r == 0:
                    continue

                p_l = np.mean(y[left_mask])
                p_r = np.mean(y[right_mask])
                gini_l = 1.0 - (p_l ** 2 + (1.0 - p_l) ** 2)
                gini_r = 1.0 - (p_r ** 2 + (1.0 - p_r) ** 2)

                gain = base_gini - (n_l / n_samples * gini_l + n_r / n_samples * gini_r)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = th

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        p1 = np.mean(y) if len(y) > 0 else 0.5
        pred = 1 if p1 >= 0.5 else 0

        if depth >= self.max_depth or len(y) <= 2 or p1 == 0.0 or p1 == 1.0:
            return {"leaf": True, "pred": pred}

        feat, thresh, gain = self._best_split(X, y)
        if feat is None or gain <= 1e-6:
            return {"leaf": True, "pred": pred}

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "leaf": False,
            "feature": feat,
            "threshold": thresh,
            "left": left_child,
            "right": right_child,
            "pred": pred,
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
# MAIN D7 EVALUATION
# ==============================================================================

def main():
    Path("results").mkdir(exist_ok=True)

    with open("data/data_v1/families.json", encoding="utf-8") as f:
        fams_raw = json.load(f)
    fam_dict = {f["family_id"]: f for f in fams_raw}

    gate_df = pd.read_csv("results/gate_per_family.csv")
    fams = gate_df.to_dict("records")
    n_fams = len(fams)

    # 1. Prepare Features and Ground Truth Targets
    # Features: density, deg_cv, clustering, n
    # Structure types list for one-hot encoding
    stypes = sorted(list(set(r["structure_type"] for r in fams)))
    st_to_idx = {st: i for i, st in enumerate(stypes)}

    X_dense = []
    X_with_st = []
    y = []  # 1 if RLF < DSATUR (RLF is strictly better), 0 if DSATUR <= RLF
    ds_colors = []
    rlf_colors = []
    r_colors = []
    st_groups = []

    for r in fams:
        fid = r["family_id"]
        raw = fam_dict[fid]
        stats = raw["probe_stats"]

        num_feats = [
            float(stats["density"]),
            float(stats["deg_cv"]),
            float(stats["clustering"]),
            float(stats["n"]) / 300.0,  # normalized n
        ]
        X_dense.append(num_feats)

        # One-hot structure type
        st_onehot = [0.0] * len(stypes)
        st_onehot[st_to_idx[r["structure_type"]]] = 1.0
        X_with_st.append(num_feats + st_onehot)

        # Labels based on TEST performance
        c_ds = float(r["dsatur_test"])
        c_rlf = float(r["rlf_test"])
        ds_colors.append(c_ds)
        rlf_colors.append(c_rlf)
        r_colors.append(min(c_ds, c_rlf))
        st_groups.append(r["structure_type"])

        # Target: 1 = choose RLF, 0 = choose DSATUR
        # If equal, DSATUR is preferred (lower complexity)
        y.append(1 if c_rlf < c_ds else 0)

    X_dense = np.array(X_dense, dtype=np.float64)
    X_with_st = np.array(X_with_st, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    ds_colors = np.array(ds_colors, dtype=np.float64)
    rlf_colors = np.array(rlf_colors, dtype=np.float64)
    r_colors = np.array(r_colors, dtype=np.float64)
    st_groups = np.array(st_groups)

    # Standardize numerical features
    mean_d = np.mean(X_dense, axis=0)
    std_d = np.std(X_dense, axis=0) + 1e-8
    X_dense_scaled = (X_dense - mean_d) / std_d

    mean_st = np.mean(X_with_st[:, :4], axis=0)
    std_st = np.std(X_with_st[:, :4], axis=0) + 1e-8
    X_with_st_scaled = X_with_st.copy()
    X_with_st_scaled[:, :4] = (X_with_st[:, :4] - mean_st) / std_st

    # Baseline Policies
    # Policy 1: Always RLF (pred = 1)
    always_rlf_regret = rlf_colors - r_colors
    # Policy 2: Always DSATUR (pred = 0)
    always_ds_regret = ds_colors - r_colors

    # --------------------------------------------------------------------------
    # EVALUATION HARNESS
    # --------------------------------------------------------------------------
    def evaluate_cv(X_mat, cv_type="family", model_name="lr"):
        preds = np.zeros(n_fams, dtype=np.int64)

        if cv_type == "family":
            # Leave-One-Out CV across 106 families
            for i in range(n_fams):
                train_idx = [j for j in range(n_fams) if j != i]
                test_idx = [i]

                if model_name == "lr":
                    clf = PureLogisticRegression(l2_reg=1.0)
                else:
                    clf = PureDecisionTree(max_depth=3)

                clf.fit(X_mat[train_idx], y[train_idx])
                preds[i] = clf.predict(X_mat[test_idx])[0]

        elif cv_type == "structure_type":
            # Leave-One-Group-Out CV by structure type
            unique_types = np.unique(st_groups)
            for st in unique_types:
                test_idx = np.where(st_groups == st)[0]
                train_idx = np.where(st_groups != st)[0]

                if model_name == "lr":
                    clf = PureLogisticRegression(l2_reg=1.0)
                else:
                    clf = PureDecisionTree(max_depth=3)

                clf.fit(X_mat[train_idx], y[train_idx])
                preds[test_idx] = clf.predict(X_mat[test_idx])

        # Chosen algorithm colors: if pred == 1 -> RLF, else DSATUR
        chosen_colors = np.where(preds == 1, rlf_colors, ds_colors)
        regret_colors = chosen_colors - r_colors
        regret_pct = np.where(r_colors > 0, regret_colors / r_colors * 100.0, 0.0)
        acc = np.mean(preds == y) * 100.0

        return {
            "preds": preds,
            "accuracy": acc,
            "mean_regret_colors": float(np.mean(regret_colors)),
            "mean_regret_pct": float(np.mean(regret_pct)),
            "median_regret_colors": float(np.median(regret_colors)),
            "median_regret_pct": float(np.median(regret_pct)),
            "chosen_colors": chosen_colors,
            "regret_colors": regret_colors,
        }

    # Run combinations
    # 1. LOOCV Family (Numeric only + with structure type)
    res_lr_fam = evaluate_cv(X_dense_scaled, cv_type="family", model_name="lr")
    res_dt_fam = evaluate_cv(X_dense, cv_type="family", model_name="dt")

    # 2. LOGO-CV Structure Type (Must use numeric only since held-out structure type is unseen)
    res_lr_st = evaluate_cv(X_dense_scaled, cv_type="structure_type", model_name="lr")
    res_dt_st = evaluate_cv(X_dense, cv_type="structure_type", model_name="dt")

    # Baseline metrics
    mean_rlf_regret = float(np.mean(always_rlf_regret))
    pct_rlf_regret = float(np.mean(always_rlf_regret / r_colors * 100.0))
    mean_ds_regret = float(np.mean(always_ds_regret))
    pct_ds_regret = float(np.mean(always_ds_regret / r_colors * 100.0))

    # Save detailed per-family predictions for best selector
    rows = []
    for i, r in enumerate(fams):
        fid = r["family_id"]
        rows.append({
            "family_id": fid,
            "structure_type": r["structure_type"],
            "pool": r["pool"],
            "dsatur_test": ds_colors[i],
            "rlf_test": rlf_colors[i],
            "optimal_algo": "rlf" if y[i] == 1 else "dsatur",
            "pred_lr_fam": "rlf" if res_lr_fam["preds"][i] == 1 else "dsatur",
            "pred_dt_fam": "rlf" if res_dt_fam["preds"][i] == 1 else "dsatur",
            "pred_lr_st": "rlf" if res_lr_st["preds"][i] == 1 else "dsatur",
            "pred_dt_st": "rlf" if res_dt_st["preds"][i] == 1 else "dsatur",
            "regret_always_rlf": always_rlf_regret[i],
            "regret_lr_fam": res_lr_fam["regret_colors"][i],
            "regret_dt_fam": res_dt_fam["regret_colors"][i],
        })
    pd.DataFrame(rows).to_csv("results/d7_selector_results.csv", index=False)

    # --------------------------------------------------------------------------
    # FORMAT SUMMARY
    # --------------------------------------------------------------------------
    summary = []
    summary.append("=" * 80)
    summary.append("DIAGNOSTIC D7: CARD-BASED ALGORITHM SELECTOR {DSATUR, RLF}")
    summary.append("Evaluated across all 106 families using Cross-Validation")
    summary.append("=" * 80)

    summary.append("\n1. BASELINE COMPARISON (Without Selector):")
    summary.append(f"   Always DSATUR : Accuracy = {np.mean(y == 0)*100:.1f}% | Mean Regret = {mean_ds_regret:.3f} colors ({pct_ds_regret:.2f}%)")
    summary.append(f"   Always RLF    : Accuracy = {np.mean(y == 1)*100:.1f}% | Mean Regret = {mean_rlf_regret:.3f} colors ({pct_rlf_regret:.2f}%)")
    summary.append(f"   (RLF is better on {np.sum(y == 1)}/106 families = {np.mean(y == 1)*100:.1f}%; DSATUR is better/equal on {np.sum(y == 0)}/106 families)")

    summary.append("\n2. CROSS-VALIDATION BY FAMILY (LOOCV across 106 families):")
    summary.append(f"   Logistic Regression : Accuracy = {res_lr_fam['accuracy']:.1f}% | Mean Regret = {res_lr_fam['mean_regret_colors']:.3f} colors ({res_lr_fam['mean_regret_pct']:.2f}%) | Median = {res_lr_fam['median_regret_pct']:.2f}%")
    summary.append(f"   Decision Tree       : Accuracy = {res_dt_fam['accuracy']:.1f}% | Mean Regret = {res_dt_fam['mean_regret_colors']:.3f} colors ({res_dt_fam['mean_regret_pct']:.2f}%) | Median = {res_dt_fam['median_regret_pct']:.2f}%")

    summary.append("\n3. CROSS-VALIDATION BY STRUCTURE TYPE (Leave-One-Group-Out CV across 11 types):")
    summary.append(f"   Logistic Regression : Accuracy = {res_lr_st['accuracy']:.1f}% | Mean Regret = {res_lr_st['mean_regret_colors']:.3f} colors ({res_lr_st['mean_regret_pct']:.2f}%) | Median = {res_lr_st['median_regret_pct']:.2f}%")
    summary.append(f"   Decision Tree       : Accuracy = {res_dt_st['accuracy']:.1f}% | Mean Regret = {res_dt_st['mean_regret_colors']:.3f} colors ({res_dt_st['mean_regret_pct']:.2f}%) | Median = {res_dt_st['median_regret_pct']:.2f}%")

    # Non-control breakdown for Decision Tree LOOCV
    nc_mask = np.array(['interval' not in r['family_id'] for r in fams])
    summary.append("\n4. NON-CONTROL FAMILIES BREAKDOWN (N=99 families):")
    summary.append(f"   Always RLF Regret          : {float(np.mean(always_rlf_regret[nc_mask])):.3f} colors ({float(np.mean(always_rlf_regret[nc_mask] / r_colors[nc_mask] * 100.0)):.2f}%)")
    summary.append(f"   Always DSATUR Regret       : {float(np.mean(always_ds_regret[nc_mask])):.3f} colors ({float(np.mean(always_ds_regret[nc_mask] / r_colors[nc_mask] * 100.0)):.2f}%)")
    summary.append(f"   Decision Tree (Family CV)  : Regret = {float(np.mean(res_dt_fam['regret_colors'][nc_mask])):.3f} colors ({float(np.mean(res_dt_fam['regret_colors'][nc_mask] / r_colors[nc_mask] * 100.0)):.2f}%)")
    summary.append(f"   Logistic Reg (Family CV)   : Regret = {float(np.mean(res_lr_fam['regret_colors'][nc_mask])):.3f} colors ({float(np.mean(res_lr_fam['regret_colors'][nc_mask] / r_colors[nc_mask] * 100.0)):.2f}%)")

    summary_text = "\n".join(summary)
    print("\n" + summary_text)

    with open("results/d7_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("\nSaved D7 results to results/d7_summary.txt and results/d7_selector_results.csv")


if __name__ == "__main__":
    main()
