"""
export_models.py
Serializes trained calibrated pipelines, preprocessors, and triage deferral gates into models/.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


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


def export_all_models():
    os.makedirs("models", exist_ok=True)
    seed = 42

    with open("data/splits/splits_manifest.json", "r") as f:
        splits = json.load(f)

    with open("data/processed/best_models_registry.json", "r") as f:
        best_models = json.load(f)

    shared_radiomics = [
        "rad_area", "rad_perimeter", "rad_circularity", "rad_solidity",
        "rad_eccentricity", "rad_extent", "rad_aspect_ratio",
        "rad_in_mean", "rad_in_std", "rad_in_skew", "rad_in_entropy",
        "rad_ring_mean", "rad_ring_std", "rad_ring_skew", "rad_ring_entropy",
        "rad_contrast_ratio"
    ]

    manifest = {}

    # 1. BrEaST Model
    df_b = pd.read_csv("data/processed/breast_processed.csv").iloc[splits["BrEaST"]["dev_idx"]]
    feats_b = shared_radiomics + ["shape_binary", "margin_binary", "age"]
    X_b, y_b = df_b[feats_b], df_b["malignant"].values

    pipe_b = Pipeline([
        ("prep", build_preprocessor(X_b)),
        ("clf", CalibratedClassifierCV(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed), method="sigmoid", cv=3))
    ])
    pipe_b.fit(X_b, y_b)
    b_path = "models/breast_calibrated_pipeline.joblib"
    joblib.dump(pipe_b, b_path)
    manifest["BrEaST"] = {
        "file": b_path,
        "features": feats_b,
        "model_type": "LogisticRegression",
        "calibration": "Platt Sigmoid (cv=3)",
        "train_samples": len(df_b),
        "target": "malignant (1=malignant, 0=benign)"
    }
    print(f"[OK] Exported {b_path}")

    # 2. BUS-BRA Model
    df_bus = pd.read_csv("data/processed/busbra_processed.csv").iloc[splits["BUSBRA"]["dev_idx"]]
    feats_bus = shared_radiomics
    X_bus, y_bus = df_bus[feats_bus], df_bus["malignant"].values

    pipe_bus = Pipeline([
        ("prep", build_preprocessor(X_bus)),
        ("clf", CalibratedClassifierCV(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed), method="sigmoid", cv=3))
    ])
    pipe_bus.fit(X_bus, y_bus)
    bus_path = "models/busbra_calibrated_pipeline.joblib"
    joblib.dump(pipe_bus, bus_path)
    manifest["BUS-BRA"] = {
        "file": bus_path,
        "features": feats_bus,
        "model_type": "LogisticRegression",
        "calibration": "Platt Sigmoid (cv=3)",
        "train_samples": len(df_bus),
        "target": "malignant (1=malignant, 0=benign)"
    }
    print(f"[OK] Exported {bus_path}")

    # 3. UCI Mammographic Mass Model
    df_mmd = pd.read_csv("data/processed/uci_mmd_processed.csv").iloc[splits["UCI_MMD"]["dev_idx"]]
    feats_mmd = ["Age", "Shape", "Margin", "Density", "shape_binary", "margin_binary"]
    X_mmd, y_mmd = df_mmd[feats_mmd], df_mmd["malignant"].values

    pipe_mmd = Pipeline([
        ("prep", build_preprocessor(X_mmd)),
        ("clf", CalibratedClassifierCV(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed), method="sigmoid", cv=3))
    ])
    pipe_mmd.fit(X_mmd, y_mmd)
    mmd_path = "models/uci_mmd_calibrated_pipeline.joblib"
    joblib.dump(pipe_mmd, mmd_path)
    manifest["UCI_MMD"] = {
        "file": mmd_path,
        "features": feats_mmd,
        "model_type": "LogisticRegression",
        "calibration": "Platt Sigmoid (cv=3)",
        "train_samples": len(df_mmd),
        "target": "malignant (1=malignant, 0=benign)"
    }
    print(f"[OK] Exported {mmd_path}")

    # 4. Learned Triage Deferral Gate
    # Trained on BUS-BRA dev set error predictions
    p_bus = pipe_bus.predict_proba(X_bus)[:, 1]
    y_pred_bus = (p_bus >= 0.50).astype(int)
    is_err_bus = (y_pred_bus != y_bus).astype(int)
    sample_w = np.where((y_bus == 1) & (y_pred_bus == 0), 5.0, 1.0)

    X_gate = X_bus.copy()
    X_gate["p_hat"] = p_bus
    X_gate["p_margin"] = np.abs(p_bus - 0.50)

    gate_pipe = Pipeline([
        ("prep", build_preprocessor(X_gate)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
    ])
    gate_pipe.fit(X_gate, is_err_bus, clf__sample_weight=sample_w)
    gate_path = "models/triage_deferral_gate_pipeline.joblib"
    joblib.dump(gate_pipe, gate_path)
    manifest["Triage_Gate"] = {
        "file": gate_path,
        "features": list(X_gate.columns),
        "model_type": "LogisticRegression (5x FN penalty)",
        "purpose": "Learned deferral gate predicting probability of base model diagnostic error"
    }
    print(f"[OK] Exported {gate_path}")

    with open("models/model_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("[OK] Exported models/model_manifest.json")


if __name__ == "__main__":
    export_all_models()