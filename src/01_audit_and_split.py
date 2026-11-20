"""
01_audit_and_split.py
Clinical cohort ingestion, exclusion audit, and patient-leakage-free stratified splits.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_audit_and_split():
    config = load_config()
    seed = config.get("random_seed", 42)
    test_size = config.get("test_size", 0.20)
    np.random.seed(seed)

    os.makedirs("data/splits", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("================================================================================")
    print("STEP 1: COHORT AUDIT AND LEAKAGE-FREE SPLIT GENERATION")
    print("================================================================================")

    # --------------------------------------------------------------------------
    # 1. BrEaST (TCIA Breast-Lesions-USG)
    # --------------------------------------------------------------------------
    print("\n--- [1] Auditing BrEaST (TCIA Breast-Lesions-USG) ---")
    breast_candidates = [
        f for f in os.listdir("data/raw")
        if "breast" in f.lower() and (f.endswith(".xlsx") or f.endswith(".csv")) and "wisconsin" not in f.lower()
    ]
    if not breast_candidates:
        raise FileNotFoundError("Could not find BrEaST clinical spreadsheet in data/raw/")

    breast_file = os.path.join("data/raw", breast_candidates[0])
    df_breast_raw = pd.read_excel(breast_file) if breast_file.endswith(".xlsx") else pd.read_csv(breast_file)
    n_raw_breast = len(df_breast_raw)
    print(f"Loaded raw BrEaST data: {n_raw_breast} cases from {breast_file}")

    # Exclusion 1: Drop normal cases
    cls_col = next((c for c in df_breast_raw.columns if "classification" in c.lower()), "Classification")
    is_normal = df_breast_raw[cls_col].astype(str).str.lower().str.contains("normal")
    n_normal = int(is_normal.sum())

    # Exclusion 2: Drop 12 post-biopsy / post-surgical / trauma artifact scans
    # Verified CaseIDs corresponding to surgical scars, post-biopsy hematomas, fat necrosis, seromas, implant ruptures
    post_biopsy_cases = [1, 73, 104, 115, 138, 141, 157, 186, 194, 216, 222, 253]
    is_post_biopsy = df_breast_raw["CaseID"].isin(post_biopsy_cases) & (~is_normal)
    n_post_biopsy = int(is_post_biopsy.sum())

    # Apply exclusions
    df_breast = df_breast_raw[~(is_normal | is_post_biopsy)].copy().reset_index(drop=True)
    df_breast["malignant"] = (df_breast[cls_col].astype(str).str.lower() == "malignant").astype(int)

    n_benign = int((df_breast["malignant"] == 0).sum())
    n_malignant = int((df_breast["malignant"] == 1).sum())

    print(f"Exclusion Audit:")
    print(f"  - Initial cases: {n_raw_breast}")
    print(f"  - Excluded normal cases: {n_normal}")
    print(f"  - Excluded post-biopsy/surgical artifact cases: {n_post_biopsy}")
    print(f"  - Retained benchmark cohort: {len(df_breast)} cases ({n_benign} benign, {n_malignant} malignant)")

    assert len(df_breast) == 240, f"Expected 240 cases, got {len(df_breast)}"
    assert n_benign == 142, f"Expected 142 benign cases, got {n_benign}"
    assert n_malignant == 98, f"Expected 98 malignant cases, got {n_malignant}"
    print("[CHECK PASSED] Exact BrEaST cohort size verified (240 cases: 142 benign, 98 malignant).")

    # Stratified 80/20 Dev/Test Split
    sss_breast = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dev_idx_b, test_idx_b = next(sss_breast.split(df_breast, df_breast["malignant"].values))

    # Class balance check
    b_dev_y = df_breast.iloc[dev_idx_b]["malignant"].values
    b_test_y = df_breast.iloc[test_idx_b]["malignant"].values
    print(f"BrEaST Split Balance:")
    print(f"  - Dev : {len(dev_idx_b)} cases | Malignant: {b_dev_y.sum()} ({b_dev_y.mean()*100:.1f}%)")
    print(f"  - Test: {len(test_idx_b)} cases | Malignant: {b_test_y.sum()} ({b_test_y.mean()*100:.1f}%)")

    # --------------------------------------------------------------------------
    # 2. BUS-BRA (Zenodo 8231412)
    # --------------------------------------------------------------------------
    print("\n--- [2] Auditing BUS-BRA (Zenodo 8231412) ---")
    bus_file = "data/raw/bus_data.csv"
    if not os.path.exists(bus_file):
        for root, _, files in os.walk("data/raw/busbra"):
            for f in files:
                if f.lower().endswith(".csv") and "bus" in f.lower():
                    bus_file = os.path.join(root, f)
                    break
    df_bus_raw = pd.read_csv(bus_file)
    n_raw_bus = len(df_bus_raw)
    print(f"Loaded raw BUS-BRA data: {n_raw_bus} images from {bus_file}")

    # Patient ID mapping: 'Case' column groups multiple views of the same patient
    df_bus = df_bus_raw.copy()
    df_bus["patient_id"] = df_bus["Case"].astype(str)
    path_col = next((c for c in df_bus.columns if c.lower() in ["pathology", "class", "diagnosis"]), "Pathology")
    df_bus["malignant"] = (df_bus[path_col].astype(str).str.lower() == "malignant").astype(int)

    n_patients = df_bus["patient_id"].nunique()
    print(f"BUS-BRA Patient Mapping:")
    print(f"  - Total images: {len(df_bus)}")
    print(f"  - Unique patients: {n_patients}")
    print(f"  - Malignant images: {df_bus['malignant'].sum()} ({df_bus['malignant'].mean()*100:.1f}%)")

    assert len(df_bus) == 1875, f"Expected 1875 images, got {len(df_bus)}"
    assert n_patients == 1064, f"Expected 1064 unique patients, got {n_patients}"
    print("[CHECK PASSED] Exact BUS-BRA case and patient counts verified (1,875 images / 1,064 patients).")

    # Group-Aware 80/20 Dev/Test Split on patient_id
    sgkf = StratifiedGroupKFold(n_splits=int(1.0 / test_size), shuffle=True, random_state=seed)
    dev_idx_bus, test_idx_bus = next(sgkf.split(df_bus, df_bus["malignant"].values, groups=df_bus["patient_id"].values))

    # Mandatory Patient Group Disjointness Assertion
    dev_patients = set(df_bus.iloc[dev_idx_bus]["patient_id"])
    test_patients = set(df_bus.iloc[test_idx_bus]["patient_id"])
    assert dev_patients.isdisjoint(test_patients), "PATIENT GROUP LEAKAGE DETECTED in BUS-BRA!"
    print(f"[CHECK PASSED] Patient leakage test: set(dev_patient_ids).isdisjoint(test_patient_ids) is TRUE.")
    print(f"BUS-BRA Group Split Balance:")
    print(f"  - Dev : {len(dev_idx_bus)} images ({len(dev_patients)} patients) | Malignant: {df_bus.iloc[dev_idx_bus]['malignant'].sum()}")
    print(f"  - Test: {len(test_idx_bus)} images ({len(test_patients)} patients) | Malignant: {df_bus.iloc[test_idx_bus]['malignant'].sum()}")

    # --------------------------------------------------------------------------
    # 3. UCI Mammographic Mass (UCI #161)
    # --------------------------------------------------------------------------
    print("\n--- [3] Auditing UCI Mammographic Mass ---")
    mmd_file = "data/raw/mammographic_masses.data"
    mmd_cols = ["BI_RADS", "Age", "Shape", "Margin", "Density", "Severity"]
    df_mmd_raw = pd.read_csv(mmd_file, names=mmd_cols, na_values="?")
    n_raw_mmd = len(df_mmd_raw)

    # Clean rows with missing target Severity
    df_mmd = df_mmd_raw.dropna(subset=["Severity"]).copy().reset_index(drop=True)
    df_mmd["malignant"] = df_mmd["Severity"].astype(int)
    print(f"Loaded UCI Mammographic Mass: {n_raw_mmd} raw rows, {len(df_mmd)} retained cases.")
    print(f"  - Malignant: {df_mmd['malignant'].sum()} ({df_mmd['malignant'].mean()*100:.1f}%)")

    # Stratified 80/20 Dev/Test Split
    sss_mmd = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dev_idx_m, test_idx_m = next(sss_mmd.split(df_mmd, df_mmd["malignant"].values))

    print(f"UCI MMD Split Balance:")
    print(f"  - Dev : {len(dev_idx_m)} cases | Malignant: {df_mmd.iloc[dev_idx_m]['malignant'].sum()}")
    print(f"  - Test: {len(test_idx_m)} cases | Malignant: {df_mmd.iloc[test_idx_m]['malignant'].sum()}")

    # --------------------------------------------------------------------------
    # Save Split Indices and Cleaned Tables
    # --------------------------------------------------------------------------
    split_meta = {
        "BrEaST": {
            "dev_idx": dev_idx_b.tolist(),
            "test_idx": test_idx_b.tolist(),
            "total_n": len(df_breast),
            "dev_n": len(dev_idx_b),
            "test_n": len(test_idx_b),
        },
        "BUSBRA": {
            "dev_idx": dev_idx_bus.tolist(),
            "test_idx": test_idx_bus.tolist(),
            "total_n": len(df_bus),
            "dev_n": len(dev_idx_bus),
            "test_n": len(test_idx_bus),
            "dev_patients": len(dev_patients),
            "test_patients": len(test_patients),
        },
        "UCI_MMD": {
            "dev_idx": dev_idx_m.tolist(),
            "test_idx": test_idx_m.tolist(),
            "total_n": len(df_mmd),
            "dev_n": len(dev_idx_m),
            "test_n": len(test_idx_m),
        }
    }

    # Save CSVs
    pd.DataFrame({"dev_idx": pd.Series(dev_idx_b), "test_idx": pd.Series(test_idx_b)}).to_csv("data/splits/breast_splits.csv", index=False)
    pd.DataFrame({"dev_idx": pd.Series(dev_idx_bus), "test_idx": pd.Series(test_idx_bus)}).to_csv("data/splits/busbra_splits.csv", index=False)
    pd.DataFrame({"dev_idx": pd.Series(dev_idx_m), "test_idx": pd.Series(test_idx_m)}).to_csv("data/splits/uci_mmd_splits.csv", index=False)

    # Save JSON summary
    with open("data/splits/splits_manifest.json", "w") as f:
        json.dump(split_meta, f, indent=2)

    # Save cleaned cohort frames
    df_breast.to_csv("data/processed/breast_cleaned.csv", index=False)
    df_bus.to_csv("data/processed/busbra_cleaned.csv", index=False)
    df_mmd.to_csv("data/processed/uci_mmd_cleaned.csv", index=False)

    print("\n[SUCCESS] Step 1 finished. All split indices and cleaned manifests saved to data/splits/.")


if __name__ == "__main__":
    run_audit_and_split()