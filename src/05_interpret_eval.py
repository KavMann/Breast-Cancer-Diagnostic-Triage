"""
05_interpret_eval.py
Feature interpretation (dev data only), SHAP plots, and single-pass untouched test evaluation
including cross-dataset transfer (BUS-BRA -> BrEaST zero-shot) with 1,000-resample bootstrap 95% CIs.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix, recall_score
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
import shap


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / len(y_true)) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def fit_temperature(y_true, y_prob):
    p_clip = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    logits = logit(p_clip)

    def nll(T):
        p_cal = expit(logits / T)
        return -np.mean(y_true * np.log(np.clip(p_cal, 1e-9, 1.0)) + (1.0 - y_true) * np.log(np.clip(1.0 - p_cal, 1e-9, 1.0)))

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


def apply_temperature(y_prob, T):
    p_clip = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    return expit(logit(p_clip) / T)


def build_preprocessor(X_df):
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols))
    return ColumnTransformer(transformers)


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], y_prob[idx]))
    if not vals:
        return 0.0, 0.0
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run_interpret_eval():
    print("================================================================================")
    print("STEP 5: SHAP / LOG-ODDS INTERPRETABILITY & UNTOUCHED TEST EVALUATION")
    print("================================================================================")

    config = load_config()
    seed = config.get("random_seed", 42)
    n_resamples = config.get("bootstrap_resamples", 1000)

    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    with open("data/splits/splits_manifest.json", "r") as f:
        splits = json.load(f)

    with open("data/processed/best_models_registry.json", "r") as f:
        best_models = json.load(f)

    # Dataset feature definitions
    shared_radiomics = [
        "rad_area", "rad_perimeter", "rad_circularity", "rad_solidity",
        "rad_eccentricity", "rad_extent", "rad_aspect_ratio",
        "rad_in_mean", "rad_in_std", "rad_in_skew", "rad_in_entropy",
        "rad_ring_mean", "rad_ring_std", "rad_ring_skew", "rad_ring_entropy",
        "rad_contrast_ratio"
    ]

    cohort_configs = {
        "BrEaST": {
            "csv": "data/processed/breast_processed.csv",
            "features": shared_radiomics + ["shape_binary", "margin_binary", "age"],
            "target": "malignant",
            "dev_idx": splits["BrEaST"]["dev_idx"],
            "test_idx": splits["BrEaST"]["test_idx"]
        },
        "BUS-BRA": {
            "csv": "data/processed/busbra_processed.csv",
            "features": shared_radiomics,
            "target": "malignant",
            "dev_idx": splits["BUSBRA"]["dev_idx"],
            "test_idx": splits["BUSBRA"]["test_idx"]
        },
        "UCI_MMD": {
            "csv": "data/processed/uci_mmd_processed.csv",
            "features": ["Age", "Shape", "Margin", "Density", "shape_binary", "margin_binary"],
            "target": "malignant",
            "dev_idx": splits["UCI_MMD"]["dev_idx"],
            "test_idx": splits["UCI_MMD"]["test_idx"]
        }
    }

    # --------------------------------------------------------------------------
    # Part A: Feature Interpretation (Dev Data Only)
    # --------------------------------------------------------------------------
    print("\n--- [A] Computing Standardized Log-Odds & SHAP Plots on Dev Data ---")
    log_odds_rows = []

    for ds_name, cfg in cohort_configs.items():
        df_full = pd.read_csv(cfg["csv"])
        df_dev = df_full.iloc[cfg["dev_idx"]].copy().reset_index(drop=True)
        X_dev = df_dev[cfg["features"]]
        y_dev = df_dev[cfg["target"]].values

        prep = build_preprocessor(X_dev)
        X_dev_t = prep.fit_transform(X_dev)
        feat_names = prep.get_feature_names_out()

        # Fit Standardized Logistic Regression for Log-Odds Interpretation
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        lr.fit(X_dev_t, y_dev)

        # Approximate 95% CIs for log-odds using bootstrap on Dev
        rng = np.random.default_rng(seed)
        boot_coefs = []
        for _ in range(500):
            b_idx = rng.integers(0, len(y_dev), len(y_dev))
            if len(np.unique(y_dev[b_idx])) < 2:
                continue
            b_lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
            b_lr.fit(X_dev_t[b_idx], y_dev[b_idx])
            boot_coefs.append(b_lr.coef_[0])
        boot_coefs = np.array(boot_coefs)

        for i, f_name in enumerate(feat_names):
            coef_val = float(lr.coef_[0][i])
            lo_ci = float(np.percentile(boot_coefs[:, i], 2.5))
            hi_ci = float(np.percentile(boot_coefs[:, i], 97.5))
            or_val = float(np.exp(coef_val))
            or_lo = float(np.exp(lo_ci))
            or_hi = float(np.exp(hi_ci))

            log_odds_rows.append({
                "Dataset": ds_name,
                "Feature": f_name,
                "Log_Odds": round(coef_val, 4),
                "Log_Odds_95CI": f"[{lo_ci:.3f}, {hi_ci:.3f}]",
                "Odds_Ratio": round(or_val, 4),
                "Odds_Ratio_95CI": f"[{or_lo:.3f}, {or_hi:.3f}]"
            })

        # SHAP Summary Plots for the Dev Set
        bm = best_models[ds_name]
        try:
            if "Forest" in bm or "HistGB" in bm:
                tree_clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=seed)
                tree_clf.fit(X_dev_t, y_dev)
                explainer = shap.TreeExplainer(tree_clf)
                shap_values = explainer.shap_values(X_dev_t)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
            else:
                explainer = shap.LinearExplainer(lr, X_dev_t)
                shap_values = explainer.shap_values(X_dev_t)

            plt.figure(figsize=(7.5, 5))
            shap.summary_plot(shap_values, X_dev_t, feature_names=feat_names, plot_type="bar", max_display=12, show=False)
            plt.title(f"Top Predictor Importance (SHAP on Dev) — {ds_name} [{bm}]")
            plt.tight_layout()
            shap_fig_path = f"outputs/figures/shap_{ds_name}.png"
            plt.savefig(shap_fig_path, dpi=200)
            plt.close()
            print(f"  Saved SHAP plot to {shap_fig_path}")
        except Exception as e:
            print(f"  [WARN] SHAP plot fallback for {ds_name}: {e}")

    df_log_odds = pd.DataFrame(log_odds_rows)
    df_log_odds.to_csv("outputs/tables/dev_feature_log_odds.csv", index=False)
    print("  Saved standardized log-odds and odds ratios to outputs/tables/dev_feature_log_odds.csv")

    # --------------------------------------------------------------------------
    # Part B: Untouched Test Split Evaluation (Run ONCE)
    # --------------------------------------------------------------------------
    print("\n--- [B] Single-Pass Untouched Test Split Evaluation ---")
    final_test_rows = []

    for ds_name, cfg in cohort_configs.items():
        df_full = pd.read_csv(cfg["csv"])
        df_dev = df_full.iloc[cfg["dev_idx"]].copy().reset_index(drop=True)
        df_test = df_full.iloc[cfg["test_idx"]].copy().reset_index(drop=True)

        X_dev = df_dev[cfg["features"]]
        y_dev = df_dev[cfg["target"]].values
        X_test = df_test[cfg["features"]]
        y_test = df_test[cfg["target"]].values

        bm = best_models[ds_name]

        # Fit preprocessor strictly on Dev, transform Test
        prep = build_preprocessor(X_dev)
        X_dev_t = prep.fit_transform(X_dev)
        X_test_t = prep.transform(X_test)

        # Fit Platt-calibrated classifier on full Dev
        base_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed) if bm == "LogReg" else \
                   RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=seed)
        cal_clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
        cal_clf.fit(X_dev_t, y_dev)

        p_test = cal_clf.predict_proba(X_test_t)[:, 1]
        y_pred = (p_test >= 0.50).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        roc = roc_auc_score(y_test, p_test)
        pr = average_precision_score(y_test, p_test)
        rec = recall_score(y_test, y_pred)
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ece = compute_ece(y_test, p_test)
        brier = brier_score_loss(y_test, p_test)

        roc_lo, roc_hi = bootstrap_ci(y_test, p_test, roc_auc_score, n_boot=n_resamples, seed=seed)
        pr_lo, pr_hi = bootstrap_ci(y_test, p_test, average_precision_score, n_boot=n_resamples, seed=seed)
        rec_lo, rec_hi = bootstrap_ci(y_test, p_test, lambda yt, yp: recall_score(yt, (yp >= 0.5).astype(int)), n_boot=n_resamples, seed=seed)

        final_test_rows.append({
            "Evaluation_Setting": f"{ds_name} (Untouched Test)",
            "Train_Cohort": ds_name,
            "Test_Cohort": ds_name,
            "Model": f"{bm} (Sigmoid)",
            "Test_N": len(y_test),
            "Test_Malignant": int(y_test.sum()),
            "ROC_AUC": round(roc, 4),
            "ROC_AUC_95CI": f"{roc:.3f} ({roc_lo:.3f}-{roc_hi:.3f})",
            "PR_AUC": round(pr, 4),
            "PR_AUC_95CI": f"{pr:.3f} ({pr_lo:.3f}-{pr_hi:.3f})",
            "Recall": f"{rec*100:.1f}% ({rec_lo*100:.1f}-{rec_hi*100:.1f}%)",
            "Specificity": f"{spec*100:.1f}%",
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "ECE": round(ece, 4),
            "Brier": round(brier, 4)
        })

    # --------------------------------------------------------------------------
    # Part C: Cross-Dataset Domain Shift Transfer (BUS-BRA -> BrEaST Zero-Shot)
    # --------------------------------------------------------------------------
    print("\n--- [C] Cross-Dataset Transfer: Train BUS-BRA -> Test BrEaST Zero-Shot ---")
    df_bus_full = pd.read_csv("data/processed/busbra_processed.csv")
    df_bus_dev = df_bus_full.iloc[splits["BUSBRA"]["dev_idx"]].copy().reset_index(drop=True)

    df_breast_full = pd.read_csv("data/processed/breast_processed.csv")
    df_breast_test = df_breast_full.iloc[splits["BrEaST"]["test_idx"]].copy().reset_index(drop=True)

    X_bus_dev = df_bus_dev[shared_radiomics]
    y_bus_dev = df_bus_dev["malignant"].values

    X_breast_test = df_breast_test[shared_radiomics]
    y_breast_test = df_breast_test["malignant"].values

    prep_transfer = build_preprocessor(X_bus_dev)
    X_bus_t = prep_transfer.fit_transform(X_bus_dev)
    X_breast_t = prep_transfer.transform(X_breast_test)

    # 1. Uncalibrated / Standard Platt on Transfer
    transfer_clf = CalibratedClassifierCV(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed), method="sigmoid", cv=3)
    transfer_clf.fit(X_bus_t, y_bus_dev)
    p_transfer_raw = transfer_clf.predict_proba(X_breast_t)[:, 1]
    y_pred_tr_raw = (p_transfer_raw >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_breast_test, y_pred_tr_raw).ravel()
    roc_tr = roc_auc_score(y_breast_test, p_transfer_raw)
    pr_tr = average_precision_score(y_breast_test, p_transfer_raw)
    rec_tr = recall_score(y_breast_test, y_pred_tr_raw)
    spec_tr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ece_tr = compute_ece(y_breast_test, p_transfer_raw)
    brier_tr = brier_score_loss(y_breast_test, p_transfer_raw)

    roc_lo, roc_hi = bootstrap_ci(y_breast_test, p_transfer_raw, roc_auc_score, n_boot=n_resamples, seed=seed)
    pr_lo, pr_hi = bootstrap_ci(y_breast_test, p_transfer_raw, average_precision_score, n_boot=n_resamples, seed=seed)

    final_test_rows.append({
        "Evaluation_Setting": "Cross-Dataset Zero-Shot (No TempScale)",
        "Train_Cohort": "BUS-BRA (Dev)",
        "Test_Cohort": "BrEaST (Test)",
        "Model": "LogReg Radiomics Transfer",
        "Test_N": len(y_breast_test),
        "Test_Malignant": int(y_breast_test.sum()),
        "ROC_AUC": round(roc_tr, 4),
        "ROC_AUC_95CI": f"{roc_tr:.3f} ({roc_lo:.3f}-{roc_hi:.3f})",
        "PR_AUC": round(pr_tr, 4),
        "PR_AUC_95CI": f"{pr_tr:.3f} ({pr_lo:.3f}-{pr_hi:.3f})",
        "Recall": f"{rec_tr*100:.1f}%",
        "Specificity": f"{spec_tr*100:.1f}%",
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "ECE": round(ece_tr, 4),
        "Brier": round(brier_tr, 4)
    })

    # 2. Transfer With Temperature Scaling
    # Fit temperature on source CV probabilities
    bus_oof_p = transfer_clf.predict_proba(X_bus_t)[:, 1]
    T_transfer = fit_temperature(y_bus_dev, bus_oof_p)
    p_transfer_temp = apply_temperature(p_transfer_raw, T_transfer)
    y_pred_tr_temp = (p_transfer_temp >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_breast_test, y_pred_tr_temp).ravel()
    roc_tr_t = roc_auc_score(y_breast_test, p_transfer_temp)
    pr_tr_t = average_precision_score(y_breast_test, p_transfer_temp)
    rec_tr_t = recall_score(y_breast_test, y_pred_tr_temp)
    spec_tr_t = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ece_tr_t = compute_ece(y_breast_test, p_transfer_temp)
    brier_tr_t = brier_score_loss(y_breast_test, p_transfer_temp)

    roc_lo, roc_hi = bootstrap_ci(y_breast_test, p_transfer_temp, roc_auc_score, n_boot=n_resamples, seed=seed)
    pr_lo, pr_hi = bootstrap_ci(y_breast_test, p_transfer_temp, average_precision_score, n_boot=n_resamples, seed=seed)

    final_test_rows.append({
        "Evaluation_Setting": f"Cross-Dataset Zero-Shot (TempScale T={T_transfer:.2f})",
        "Train_Cohort": "BUS-BRA (Dev)",
        "Test_Cohort": "BrEaST (Test)",
        "Model": "LogReg Radiomics Transfer + TS",
        "Test_N": len(y_breast_test),
        "Test_Malignant": int(y_breast_test.sum()),
        "ROC_AUC": round(roc_tr_t, 4),
        "ROC_AUC_95CI": f"{roc_tr_t:.3f} ({roc_lo:.3f}-{roc_hi:.3f})",
        "PR_AUC": round(pr_tr_t, 4),
        "PR_AUC_95CI": f"{pr_tr_t:.3f} ({pr_lo:.3f}-{pr_hi:.3f})",
        "Recall": f"{rec_tr_t*100:.1f}%",
        "Specificity": f"{spec_tr_t*100:.1f}%",
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "ECE": round(ece_tr_t, 4),
        "Brier": round(brier_tr_t, 4)
    })

    df_final = pd.DataFrame(final_test_rows)
    df_final.to_csv("outputs/tables/final_test_results.csv", index=False)

    print("\n================================================================================")
    print("FINAL UNTOUCHED TEST SET & CROSS-DATASET DOMAIN SHIFT RESULTS:")
    print("================================================================================")
    display_cols = ["Evaluation_Setting", "ROC_AUC_95CI", "PR_AUC_95CI", "Recall", "Specificity", "TP", "TN", "FP", "FN", "ECE", "Brier"]
    print(df_final[display_cols].to_string(index=False))

    print("\n[SUCCESS] Step 5 finished. Final test results saved to outputs/tables/final_test_results.csv.")


if __name__ == "__main__":
    run_interpret_eval()