"""
02_radiomics_and_features.py
Mask-to-morphology radiomics extraction, strict data leakage blocklist, and cross-cohort harmonized schema.
"""

import os
import yaml
import numpy as np
import pandas as pd
from skimage import io as skio, measure, morphology
from scipy.stats import skew, entropy as sp_entropy


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


LEAKAGE_TERMS = [
    "birads", "bi-rads", "bi_rads",
    "interpretation",
    "verification",
    "histopath", "histology",
    "diagnosis",
    "classification", "severity", "malignant", "target"
]


def assert_strict_no_leakage(feature_cols, dataset_name):
    bad = [c for c in feature_cols if any(k in c.lower() for k in LEAKAGE_TERMS)]
    assert len(bad) == 0, f"LEAKAGE DETECTED in {dataset_name}: {bad}"
    print(f"\n[LEAKAGE AUDIT PASSED] {dataset_name} ({len(feature_cols)} features):")
    print(f"X.columns.tolist() = {feature_cols}")


def extract_radiomics_from_mask(image_path, mask_path):
    """
    Extracts 7 objective geometric shape descriptors and 4 first-order
    grayscale intensity stats inside lesion mask vs. 10-pixel peritumoral ring.
    """
    if not (mask_path and os.path.exists(mask_path)):
        return None
    try:
        mask = skio.imread(mask_path)
    except Exception:
        return None

    if mask.ndim == 3:
        mask = mask[..., 0]
    binary = (mask > 0).astype(np.uint8)

    props = measure.regionprops(binary)
    if not props:
        return None

    # Primary lesion: largest connected component
    p = max(props, key=lambda r: r.area)
    area = float(p.area)
    perim = float(p.perimeter) if p.perimeter > 0 else 1e-6
    circularity = float(4.0 * np.pi * area / (perim ** 2))
    minor = float(p.axis_minor_length) if p.axis_minor_length > 0 else 1e-6
    aspect_ratio = float(p.axis_major_length / minor)

    feats = {
        "rad_area": area,
        "rad_perimeter": perim,
        "rad_circularity": circularity,
        "rad_solidity": float(p.solidity),
        "rad_eccentricity": float(p.eccentricity),
        "rad_extent": float(p.extent),
        "rad_aspect_ratio": aspect_ratio,
    }

    # Grayscale image stats inside vs. peritumoral ring
    if image_path and os.path.exists(image_path):
        try:
            img = skio.imread(image_path)
            if img.ndim == 3:
                img = img.mean(axis=2)
            img = img.astype(np.float32)

            if img.shape == binary.shape:
                inside = img[binary > 0]
                dilated = morphology.binary_dilation(binary > 0, morphology.disk(10))
                ring = img[dilated & (binary == 0)]

                if inside.size > 5:
                    hist_in, _ = np.histogram(inside, bins=32, range=(0, 255), density=True)
                    feats["rad_in_mean"] = float(inside.mean())
                    feats["rad_in_std"] = float(inside.std())
                    feats["rad_in_skew"] = float(skew(inside))
                    feats["rad_in_entropy"] = float(sp_entropy(hist_in + 1e-12))
                else:
                    feats["rad_in_mean"] = 0.0
                    feats["rad_in_std"] = 0.0
                    feats["rad_in_skew"] = 0.0
                    feats["rad_in_entropy"] = 0.0

                if ring.size > 5:
                    hist_ring, _ = np.histogram(ring, bins=32, range=(0, 255), density=True)
                    feats["rad_ring_mean"] = float(ring.mean())
                    feats["rad_ring_std"] = float(ring.std())
                    feats["rad_ring_skew"] = float(skew(ring))
                    feats["rad_ring_entropy"] = float(sp_entropy(hist_ring + 1e-12))
                    feats["rad_contrast_ratio"] = float((feats["rad_in_mean"] + 1e-6) / (feats["rad_ring_mean"] + 1e-6))
                else:
                    feats["rad_ring_mean"] = 0.0
                    feats["rad_ring_std"] = 0.0
                    feats["rad_ring_skew"] = 0.0
                    feats["rad_ring_entropy"] = 0.0
                    feats["rad_contrast_ratio"] = 1.0
        except Exception:
            pass

    return feats


def run_radiomics_and_features():
    print("================================================================================")
    print("STEP 2: RADIOMICS EXTRACTION & FEATURE HARMONIZATION")
    print("================================================================================")

    os.makedirs("data/processed", exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. Process BrEaST Radiomics & Clinical Features
    # --------------------------------------------------------------------------
    print("\n--- [1] Extracting BrEaST Radiomics & Clinical Features ---")
    df_breast = pd.read_csv("data/processed/breast_cleaned.csv")
    breast_img_dir = "data/raw/breast/BrEaST-Lesions_USG-images_and_masks"
    if not os.path.exists(breast_img_dir):
        for root, dirs, _ in os.walk("data/raw/breast"):
            if "case001.png" in os.listdir(root):
                breast_img_dir = root
                break

    rows_breast = []
    for _, row in df_breast.iterrows():
        img_p = os.path.join(breast_img_dir, str(row["Image_filename"]))
        mask_p = os.path.join(breast_img_dir, str(row["Mask_tumor_filename"]))
        fdict = extract_radiomics_from_mask(img_p, mask_p)
        if fdict is None:
            # Fallback zero defaults if mask missing
            fdict = {
                "rad_area": 0.0, "rad_perimeter": 0.0, "rad_circularity": 0.0,
                "rad_solidity": 0.0, "rad_eccentricity": 0.0, "rad_extent": 0.0, "rad_aspect_ratio": 1.0,
                "rad_in_mean": 0.0, "rad_in_std": 0.0, "rad_in_skew": 0.0, "rad_in_entropy": 0.0,
                "rad_ring_mean": 0.0, "rad_ring_std": 0.0, "rad_ring_skew": 0.0, "rad_ring_entropy": 0.0,
                "rad_contrast_ratio": 1.0
            }
        fdict["CaseID"] = row["CaseID"]
        rows_breast.append(fdict)

    df_b_rad = pd.DataFrame(rows_breast)
    df_breast_merged = df_breast.merge(df_b_rad, on="CaseID", how="left")

    # Construct harmonized binary shape and margin for BrEaST
    shape_str = df_breast_merged["Shape"].astype(str).str.lower()
    df_breast_merged["shape_binary"] = shape_str.str.contains("irreg").astype(int)
    margin_str = df_breast_merged["Margin"].astype(str).str.lower()
    df_breast_merged["margin_binary"] = (~margin_str.str.contains("circumscribed")).astype(int)
    df_breast_merged["age"] = pd.to_numeric(df_breast_merged["Age"], errors="coerce")

    # Strictly exclude leakage columns from predictors
    exclude_breast = LEAKAGE_TERMS + [
        "caseid", "image_filename", "mask_tumor_filename", "mask_other_filename",
        "pixel_size", "shape", "margin", "tissue_composition", "signs", "symptoms",
        "echogenicity", "posterior_features", "halo", "calcifications", "skin_thickening",
        "age"
    ]
    rad_cols_breast = [c for c in df_breast_merged.columns if c.startswith("rad_")]
    breast_feat_cols = rad_cols_breast + ["shape_binary", "margin_binary", "age"]
    assert_strict_no_leakage(breast_feat_cols, "BrEaST")

    df_breast_merged.to_csv("data/processed/breast_processed.csv", index=False)
    print(f"Saved BrEaST processed dataset: {df_breast_merged.shape}")

    # --------------------------------------------------------------------------
    # 2. Process BUS-BRA Radiomics
    # --------------------------------------------------------------------------
    print("\n--- [2] Extracting BUS-BRA Radiomics ---")
    df_bus = pd.read_csv("data/processed/busbra_cleaned.csv")
    bus_img_dir = "data/raw/busbra/BUSBRA/Images"
    bus_mask_dir = "data/raw/busbra/BUSBRA/Masks"

    rows_bus = []
    for _, row in df_bus.iterrows():
        raw_id = str(row["ID"])
        stem = raw_id[4:] if raw_id.startswith("bus_") else raw_id
        img_p = os.path.join(bus_img_dir, f"{raw_id}.png")
        mask_p = os.path.join(bus_mask_dir, f"mask_{stem}.png")

        fdict = extract_radiomics_from_mask(img_p, mask_p)
        if fdict is None:
            fdict = {
                "rad_area": 0.0, "rad_perimeter": 0.0, "rad_circularity": 0.0,
                "rad_solidity": 0.0, "rad_eccentricity": 0.0, "rad_extent": 0.0, "rad_aspect_ratio": 1.0,
                "rad_in_mean": 0.0, "rad_in_std": 0.0, "rad_in_skew": 0.0, "rad_in_entropy": 0.0,
                "rad_ring_mean": 0.0, "rad_ring_std": 0.0, "rad_ring_skew": 0.0, "rad_ring_entropy": 0.0,
                "rad_contrast_ratio": 1.0
            }
        fdict["ID"] = row["ID"]
        rows_bus.append(fdict)

    df_bus_rad = pd.DataFrame(rows_bus)
    df_bus_merged = df_bus.merge(df_bus_rad, on="ID", how="left")

    # Harmonized features for BUS-BRA:
    # Derive objective shape_binary from solidity and circularity
    df_bus_merged["shape_binary"] = ((df_bus_merged["rad_solidity"] < 0.85) | (df_bus_merged["rad_circularity"] < 0.55)).astype(int)
    # Derive objective margin_binary from border contrast ratio
    df_bus_merged["margin_binary"] = (df_bus_merged["rad_contrast_ratio"] < 0.90).astype(int)
    df_bus_merged["age"] = np.nan  # BUS-BRA does not include patient age; preserved in harmonized schema

    bus_feat_cols = [c for c in df_bus_merged.columns if c.startswith("rad_")]
    assert_strict_no_leakage(bus_feat_cols, "BUS-BRA")

    df_bus_merged.to_csv("data/processed/busbra_processed.csv", index=False)
    print(f"Saved BUS-BRA processed dataset: {df_bus_merged.shape}")

    # --------------------------------------------------------------------------
    # 3. Process UCI Mammographic Mass
    # --------------------------------------------------------------------------
    print("\n--- [3] Processing UCI Mammographic Mass Features ---")
    df_mmd = pd.read_csv("data/processed/uci_mmd_cleaned.csv")

    # Harmonized features:
    # Shape: 1=round, 2=oval, 3=lobular, 4=irregular -> shape_binary: 1 if irregular/lobular else 0
    df_mmd["shape_binary"] = df_mmd["Shape"].isin([3, 4]).astype(int)
    # Margin: 1=circumscribed, 2=microlobulated, 3=obscured, 4=ill-defined, 5=spiculated -> margin_binary: 0 if 1 else 1
    df_mmd["margin_binary"] = (df_mmd["Margin"] != 1).astype(int)
    df_mmd["age"] = pd.to_numeric(df_mmd["Age"], errors="coerce")

    # Standard feature set strictly excluding BI_RADS and Severity
    mmd_feat_cols = ["Age", "Shape", "Margin", "Density", "shape_binary", "margin_binary"]
    assert_strict_no_leakage(mmd_feat_cols, "UCI_MMD")

    df_mmd.to_csv("data/processed/uci_mmd_processed.csv", index=False)
    print(f"Saved UCI Mammographic Mass processed dataset: {df_mmd.shape}")

    # --------------------------------------------------------------------------
    # 4. Harmonized Cross-Cohort Domain Shift Table
    # --------------------------------------------------------------------------
    print("\n--- [4] Building Cross-Cohort Harmonized Feature Table ---")
    common_cols = ["cohort", "age", "shape_binary", "margin_binary", "malignant"]

    h_breast = df_breast_merged[["age", "shape_binary", "margin_binary", "malignant"]].copy()
    h_breast["cohort"] = "BrEaST"

    h_bus = df_bus_merged[["age", "shape_binary", "margin_binary", "malignant"]].copy()
    h_bus["cohort"] = "BUS-BRA"

    h_mmd = df_mmd[["age", "shape_binary", "margin_binary", "malignant"]].copy()
    h_mmd["cohort"] = "UCI_MMD"

    df_harmonized = pd.concat([h_breast, h_bus, h_mmd], ignore_index=True)[common_cols]
    df_harmonized.to_csv("data/processed/harmonized_cross_cohort.csv", index=False)
    print(f"Saved harmonized cross-cohort table: {df_harmonized.shape}")
    print(df_harmonized.groupby("cohort")[["malignant", "shape_binary", "margin_binary"]].mean().round(3))

    print("\n[SUCCESS] Step 2 finished. Processed feature tables saved to data/processed/ without scaler/imputer fitting.")


if __name__ == "__main__":
    run_radiomics_and_features()