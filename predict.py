"""
predict.py
Inference helper: Loads trained calibrated models and runs predictions on new data,
applying both standard 0.50 thresholding and clinical asymmetric triage deferral rules.
"""

import json
import joblib
import pandas as pd
import numpy as np


def load_model(cohort="BrEaST"):
    with open("models/model_manifest.json", "r") as f:
        manifest = json.load(f)
    if cohort not in manifest:
        raise ValueError(f"Cohort {cohort} not found. Available: {list(manifest.keys())}")
    model_path = manifest[cohort]["file"]
    model = joblib.load(model_path)
    return model, manifest[cohort]["features"]


def run_clinical_triage(cohort="BrEaST", n_samples=5):
    print("=" * 80)
    print(f"CLINICAL INFERENCE & TRIAGE PIPELINE — COHORT: {cohort.upper()}")
    print("=" * 80)

    model, features = load_model(cohort)

    # Load triage thresholds
    with open("data/processed/triage_thresholds.json", "r") as f:
        triage_cfg = json.load(f)[cohort]

    tau_low = triage_cfg["tau_low_95"]
    tau_high = triage_cfg["tau_high"]
    print(f"Triage Policies Active: tau_low (>=95% recall) = {tau_low:.3f} | tau_high = {tau_high:.2f}")

    # Load sample data
    data_files = {
        "BrEaST": "data/processed/breast_processed.csv",
        "BUS-BRA": "data/processed/busbra_processed.csv",
        "UCI_MMD": "data/processed/uci_mmd_processed.csv"
    }
    df = pd.read_csv(data_files[cohort]).head(n_samples)
    X = df[features]
    probs = model.predict_proba(X)[:, 1]

    print("\n" + "-" * 80)
    print(f"{'Case':<6} | {'P(Malignant)':<12} | {'Sym 0.50 Pred':<14} | {'Clinical Triage Recommendation':<32} | {'Ground Truth'}")
    print("-" * 80)

    for i, (p, actual) in enumerate(zip(probs, df["malignant"])):
        sym_pred = "Malignant" if p >= 0.50 else "Benign"

        if p < tau_low:
            triage_action = f"AUTO-CLEARED (Low Risk)"
        elif p > tau_high:
            triage_action = f"IMMEDIATE BIOPSY (High Risk)"
        else:
            triage_action = f"DEFERRED (Radiologist Review)"

        actual_str = "Malignant" if actual == 1 else "Benign"
        print(f"Case {i+1:<3} | {p:<12.4f} | {sym_pred:<14} | {triage_action:<32} | {actual_str}")

    print("-" * 80)
    print("Clinical Interpretation:")
    print("  * AUTO-CLEARED: Safely dismissed for routine annual follow-up (Recall >= 95%).")
    print("  * DEFERRED: Middle gray-zone routed to breast radiologist to eliminate false negatives.")
    print("  * IMMEDIATE BIOPSY: High confidence malignancy fast-tracked for tissue sampling.\n")


if __name__ == "__main__":
    run_clinical_triage("BrEaST", n_samples=5)
