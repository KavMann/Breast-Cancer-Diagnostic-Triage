"""
04_triage_deferral.py
Asymmetric rule-out triage, two-stage learned deferral gate, AUGRC, DCA, and clinical workload tables.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Trapezoid function with NumPy 1.x / 2.x fallback
TRAPEZOID_FN = getattr(np, "trapezoid", getattr(np, "trapz", None))
if TRAPEZOID_FN is None:
    from scipy.integrate import trapezoid as TRAPEZOID_FN


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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


def compute_risk_coverage(y_true, y_prob, defer_scores, coverages=np.linspace(0.50, 1.0, 51)):
    """
    Computes risk-coverage curve where lower defer_score = retained (autonomous clearance),
    higher defer_score = deferred to clinician.
    Returns: coverages, selective_risks, AUGRC
    """
    order = np.argsort(defer_scores)
    n = len(y_true)
    y_pred = (y_prob >= 0.50).astype(int)
    errors = (y_pred != y_true).astype(float)

    risks = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        retained_idx = order[:k]
        risks.append(float(errors[retained_idx].mean()))

    gen_risks = coverages * np.array(risks)
    augrc = float(TRAPEZOID_FN(gen_risks, coverages))
    return coverages, np.array(risks), augrc


def decision_curve_analysis(y_true, y_prob, thresholds=np.linspace(0.05, 0.50, 46)):
    """
    Computes Decision Curve Analysis (DCA) Net Benefit across threshold probabilities.
    Net Benefit = (TP/N) - (FP/N) * (pt / (1 - pt))
    """
    n = len(y_true)
    nb_model, nb_all = [], []
    prev = float(y_true.mean())

    for pt in thresholds:
        pred = (y_prob >= pt).astype(int)
        tp = float(((pred == 1) & (y_true == 1)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        odds = pt / (1.0 - pt)
        nb_model.append((tp / n) - (fp / n) * odds)
        nb_all.append(prev - (1.0 - prev) * odds)

    return thresholds, np.array(nb_model), np.array(nb_all)


def run_triage_deferral():
    print("================================================================================")
    print("STEP 4: ASYMMETRIC TRIAGE, LEARNED DEFERRAL, AUGRC & DECISION CURVE ANALYSIS")
    print("================================================================================")

    config = load_config()
    seed = config.get("random_seed", 42)
    capacities = config.get("deferral_capacities", [0.10, 0.20, 0.30])
    tau_high_fixed = config.get("tau_high", 0.70)
    rng = np.random.default_rng(seed)

    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    # Load Dev OOF predictions and model registry
    with open("data/processed/dev_oof_predictions.json", "r") as f:
        oof_dict = json.load(f)

    with open("data/processed/best_models_registry.json", "r") as f:
        best_models = json.load(f)

    with open("data/splits/splits_manifest.json", "r") as f:
        splits = json.load(f)

    cohort_files = {
        "BrEaST": ("data/processed/breast_processed.csv", splits["BrEaST"]["dev_idx"]),
        "BUS-BRA": ("data/processed/busbra_processed.csv", splits["BUSBRA"]["dev_idx"]),
        "UCI_MMD": ("data/processed/uci_mmd_processed.csv", splits["UCI_MMD"]["dev_idx"])
    }

    triage_table_rows = []
    triage_thresholds = {}

    for ds_name, (csv_path, dev_idx) in cohort_files.items():
        print(f"\n>>> Analyzing Clinical Triage & Deferral on {ds_name} <<<")
        df_full = pd.read_csv(csv_path)
        df_dev = df_full.iloc[dev_idx].copy().reset_index(drop=True)

        bm = best_models[ds_name]
        y_dev = np.array(oof_dict[ds_name]["y_dev"])
        p_dev = np.array(oof_dict[ds_name]["models"][bm]["sigmoid"])

        # ----------------------------------------------------------------------
        # Policy 1: Symmetric Uncertainty Baseline (|p - 0.5|)
        # ----------------------------------------------------------------------
        sym_defer = -np.abs(p_dev - 0.50)

        # ----------------------------------------------------------------------
        # Policy 2: Asymmetric Two-Threshold Rule-Out
        # ----------------------------------------------------------------------
        # Find tau_low achieving >= 95% and >= 98% recall on dev
        cand_taus = np.linspace(0.01, 0.50, 500)
        tau_low_95 = 0.05
        tau_low_98 = 0.02

        for t in cand_taus:
            rec = (p_dev[y_dev == 1] >= t).mean()
            if rec >= 0.95:
                tau_low_95 = float(t)
            if rec >= 0.98:
                tau_low_98 = float(t)

        tau_low = tau_low_95
        tau_high = tau_high_fixed
        print(f"  Tuned Asymmetric Rule-Out Thresholds: tau_low(95%)={tau_low_95:.3f}, tau_low(98%)={tau_low_98:.3f}, tau_high={tau_high:.2f}")

        # Priority deferral score: cases inside gray zone [tau_low, tau_high] get highest score
        asym_defer = np.where(
            (p_dev >= tau_low) & (p_dev <= tau_high),
            1.0 - np.abs(p_dev - (tau_low + tau_high) / 2.0),
            -np.minimum(np.abs(p_dev - tau_low), np.abs(p_dev - tau_high))
        )

        # ----------------------------------------------------------------------
        # Policy 3: Two-Stage Learned Deferral Gate (5x penalty on FN)
        # ----------------------------------------------------------------------
        y_pred_dev = (p_dev >= 0.50).astype(int)
        is_error = (y_pred_dev != y_dev).astype(int)
        sample_w = np.where((y_dev == 1) & (y_pred_dev == 0), 5.0, 1.0)

        feature_cols = [c for c in df_dev.columns if c.startswith("rad_") or c in ["Age", "Shape", "Margin", "Density", "shape_binary", "margin_binary", "age"]]
        X_gate = df_dev[feature_cols].copy()
        X_gate["p_hat"] = p_dev
        X_gate["p_margin"] = np.abs(p_dev - 0.50)

        if len(np.unique(is_error)) > 1:
            gate_pipe = Pipeline([
                ("prep", build_preprocessor(X_gate)),
                ("gate", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
            ])
            gate_pipe.fit(X_gate, is_error, gate__sample_weight=sample_w)
            learned_defer = gate_pipe.predict_proba(X_gate)[:, 1]
        else:
            learned_defer = sym_defer

        # ----------------------------------------------------------------------
        # Policy 4: Random Referral Baseline
        # ----------------------------------------------------------------------
        rand_defer = rng.random(len(y_dev))

        triage_thresholds[ds_name] = {
            "selected_model": bm,
            "tau_low_95": tau_low_95,
            "tau_low_98": tau_low_98,
            "tau_high": tau_high
        }

        # ----------------------------------------------------------------------
        # Plot Risk-Coverage Curve (AUGRC)
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.5, 5))
        policies = [
            ("Symmetric |p-0.5|", sym_defer, "tab:blue"),
            (f"Asymmetric Rule-Out [{tau_low:.2f}, {tau_high:.2f}]", asym_defer, "tab:orange"),
            ("Two-Stage Learned Gate", learned_defer, "tab:green"),
            ("Random Deferral Baseline", rand_defer, "tab:gray")
        ]

        for pol_name, d_scores, col in policies:
            covs, risks, augrc = compute_risk_coverage(y_dev, p_dev, d_scores)
            ax.plot(covs * 100, risks * 100, label=f"{pol_name} (AUGRC={augrc:.4f})", lw=1.8, color=col)

            # Compute workload & cost metrics at 10%, 20%, 30% capacity
            for cap in capacities:
                k_ret = int(round((1.0 - cap) * len(y_dev)))
                ret_order = np.argsort(d_scores)[:k_ret]
                y_ret = y_dev[ret_order]
                pred_ret = (p_dev[ret_order] >= 0.50).astype(int)

                fn = int(((y_ret == 1) & (pred_ret == 0)).sum())
                fp = int(((y_ret == 0) & (pred_ret == 1)).sum())
                cost_5to1 = 5 * fn + 1 * fp
                cost_10to1 = 10 * fn + 1 * fp
                auto_cleared_per_1000 = int(round((1.0 - cap) * 1000))

                triage_table_rows.append({
                    "Dataset": ds_name,
                    "Policy": pol_name.split(" [")[0],
                    "Clinician_Capacity": f"{int(cap*100)}%",
                    "Auto_Cleared_per_1000": auto_cleared_per_1000,
                    "Retained_FN": fn,
                    "Retained_FP": fp,
                    "Cost_5FN_1FP": cost_5to1,
                    "Cost_10FN_1FP": cost_10to1,
                    "AUGRC_0to50": round(augrc, 4)
                })

        ax.set_title(f"Risk-Coverage Curve (Dev 5-Fold OOF) — {ds_name}")
        ax.set_xlabel("Coverage (% Scans Auto-Triaged Without Referral)")
        ax.set_ylabel("Selective Error Rate Among Retained (%)")
        ax.legend(loc="upper left", fontsize=8.5)
        plt.tight_layout()
        augrc_path = f"outputs/figures/augrc_{ds_name}.png"
        plt.savefig(augrc_path, dpi=200)
        plt.close()
        print(f"  AUGRC plot saved to: {augrc_path}")

        # ----------------------------------------------------------------------
        # Plot Decision Curve Analysis (DCA)
        # ----------------------------------------------------------------------
        pts, nb_m, nb_a = decision_curve_analysis(y_dev, p_dev)
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        ax.plot(pts, nb_m, lw=2.2, label=f"Model: {bm} (Sigmoid Calibrated)", color="tab:blue")
        ax.plot(pts, nb_a, "--", lw=1.5, label="Biopsy All Scans", color="tab:red")
        ax.axhline(0.0, color="gray", ls=":", lw=1.2, label="Biopsy No Scans")

        ax.set_ylim(-0.05, max(0.40, float(nb_m.max()) + 0.05))
        ax.set_title(f"Decision Curve Analysis (Net Clinical Benefit) — {ds_name}")
        ax.set_xlabel("Threshold Probability ($p_t$)")
        ax.set_ylabel("Net Clinical Benefit")
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        dca_path = f"outputs/figures/dca_{ds_name}.png"
        plt.savefig(dca_path, dpi=200)
        plt.close()
        print(f"  DCA plot saved to: {dca_path}")

    # Save Triage Cost & Workload Table
    df_triage = pd.DataFrame(triage_table_rows)
    df_triage.to_csv("outputs/tables/dev_triage_cost_workload.csv", index=False)

    with open("data/processed/triage_thresholds.json", "w") as f:
        json.dump(triage_thresholds, f, indent=2)

    print("\n================================================================================")
    print("CLINICAL TRIAGE WORKLOAD & ERROR-COST COMPARISON (20% CAPACITY SAMPLE):")
    print("================================================================================")
    sample_view = df_triage[df_triage["Clinician_Capacity"] == "20%"].sort_values(["Dataset", "Cost_10FN_1FP"])
    print(sample_view.to_string(index=False))

    print("\n[SUCCESS] Step 4 finished. Workload table and AUGRC/DCA figures generated.")


if __name__ == "__main__":
    run_triage_deferral()