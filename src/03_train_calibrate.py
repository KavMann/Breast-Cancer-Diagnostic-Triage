"""
03_train_calibrate.py
Fold-contained 5-fold cross-validation training and probabilistic calibration audit (Dev split only).
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit


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


def get_base_models(seed=42):
    return {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "SVC_RBF": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=seed),
        "HistGBDT": HistGradientBoostingClassifier(random_state=seed),
    }


def run_train_calibrate():
    print("================================================================================")
    print("STEP 3: FOLD-CONTAINED TRAINING & CALIBRATION AUDIT (DEV SPLIT ONLY)")
    print("================================================================================")

    config = load_config()
    seed = config.get("random_seed", 42)
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    # Load splits manifest
    with open("data/splits/splits_manifest.json", "r") as f:
        splits = json.load(f)

    # Datasets specification
    cohort_specs = {
        "BrEaST": {
            "df_path": "data/processed/breast_processed.csv",
            "features": [
                "rad_area", "rad_perimeter", "rad_circularity", "rad_solidity",
                "rad_eccentricity", "rad_extent", "rad_aspect_ratio",
                "rad_in_mean", "rad_in_std", "rad_in_skew", "rad_in_entropy",
                "rad_ring_mean", "rad_ring_std", "rad_ring_skew", "rad_ring_entropy",
                "rad_contrast_ratio", "shape_binary", "margin_binary", "age"
            ],
            "target": "malignant",
            "group_col": None,
            "dev_idx": splits["BrEaST"]["dev_idx"]
        },
        "BUS-BRA": {
            "df_path": "data/processed/busbra_processed.csv",
            "features": [
                "rad_area", "rad_perimeter", "rad_circularity", "rad_solidity",
                "rad_eccentricity", "rad_extent", "rad_aspect_ratio",
                "rad_in_mean", "rad_in_std", "rad_in_skew", "rad_in_entropy",
                "rad_ring_mean", "rad_ring_std", "rad_ring_skew", "rad_ring_entropy",
                "rad_contrast_ratio"
            ],
            "target": "malignant",
            "group_col": "patient_id",
            "dev_idx": splits["BUSBRA"]["dev_idx"]
        },
        "UCI_MMD": {
            "df_path": "data/processed/uci_mmd_processed.csv",
            "features": ["Age", "Shape", "Margin", "Density", "shape_binary", "margin_binary"],
            "target": "malignant",
            "group_col": None,
            "dev_idx": splits["UCI_MMD"]["dev_idx"]
        }
    }

    cv_results = []
    oof_predictions = {}
    best_models_registry = {}

    for ds_name, spec in cohort_specs.items():
        print(f"\n>>> Running 5-Fold CV on {ds_name} (Dev N = {len(spec['dev_idx'])}) <<<")
        df_full = pd.read_csv(spec["df_path"])
        df_dev = df_full.iloc[spec["dev_idx"]].copy().reset_index(drop=True)

        X_dev = df_dev[spec["features"]]
        y_dev = df_dev[spec["target"]].values
        groups_dev = df_dev[spec["group_col"]].values if spec["group_col"] else None

        # Build cross-validation splitter
        if groups_dev is not None:
            splitter = StratifiedGroupKFold(n_splits=5)
            cv_folds = list(splitter.split(X_dev, y_dev, groups=groups_dev))
            print(f"Using StratifiedGroupKFold (groups={spec['group_col']}) across {len(cv_folds)} folds.")
        else:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            cv_folds = list(splitter.split(X_dev, y_dev))
            print(f"Using StratifiedKFold across {len(cv_folds)} folds.")

        oof_predictions[ds_name] = {"y_dev": y_dev.tolist(), "models": {}}
        best_auc_ds = -1.0
        best_model_ds = None

        models = get_base_models(seed=seed)
        for m_name, base_clf in models.items():
            print(f"  Training {m_name}...")
            oof_raw = np.zeros(len(y_dev))
            oof_sig = np.zeros(len(y_dev))
            oof_iso = np.zeros(len(y_dev))

            for tr_idx, va_idx in cv_folds:
                X_tr, y_tr = X_dev.iloc[tr_idx], y_dev[tr_idx]
                X_va = X_dev.iloc[va_idx]

                # Fold-contained preprocessing pipeline
                # 1. Uncalibrated
                pipe_raw = Pipeline([
                    ("prep", build_preprocessor(X_tr)),
                    ("clf", clone(base_clf))
                ])
                pipe_raw.fit(X_tr, y_tr)
                oof_raw[va_idx] = pipe_raw.predict_proba(X_va)[:, 1]

                # 2. Sigmoid (Platt) Calibration
                pipe_sig = Pipeline([
                    ("prep", build_preprocessor(X_tr)),
                    ("clf", CalibratedClassifierCV(clone(base_clf), method="sigmoid", cv=3))
                ])
                pipe_sig.fit(X_tr, y_tr)
                oof_sig[va_idx] = pipe_sig.predict_proba(X_va)[:, 1]

                # 3. Isotonic Calibration
                pipe_iso = Pipeline([
                    ("prep", build_preprocessor(X_tr)),
                    ("clf", CalibratedClassifierCV(clone(base_clf), method="isotonic", cv=3))
                ])
                pipe_iso.fit(X_tr, y_tr)
                oof_iso[va_idx] = pipe_iso.predict_proba(X_va)[:, 1]

            # 4. Temperature Scaling on out-of-fold probabilities
            T_opt = fit_temperature(y_dev, oof_raw)
            oof_temp = apply_temperature(oof_raw, T_opt)

            # Evaluate each calibration variant
            variants = [
                ("Raw", oof_raw),
                ("Sigmoid", oof_sig),
                ("Isotonic", oof_iso),
                (f"TempScale(T={T_opt:.2f})", oof_temp)
            ]

            for cal_name, probs in variants:
                roc = roc_auc_score(y_dev, probs)
                pr = average_precision_score(y_dev, probs)
                ece = compute_ece(y_dev, probs)
                brier = brier_score_loss(y_dev, probs)

                cv_results.append({
                    "Dataset": ds_name,
                    "Model": m_name,
                    "Calibration": cal_name,
                    "Dev_ROC_AUC": round(roc, 4),
                    "Dev_PR_AUC": round(pr, 4),
                    "Dev_ECE": round(ece, 4),
                    "Dev_Brier": round(brier, 4)
                })

            oof_predictions[ds_name]["models"][m_name] = {
                "raw": oof_raw.tolist(),
                "sigmoid": oof_sig.tolist(),
                "isotonic": oof_iso.tolist(),
                "temp": oof_temp.tolist(),
                "T_opt": T_opt
            }

            # Select best model by Sigmoid-calibrated PR-AUC and ROC-AUC
            sig_pr = average_precision_score(y_dev, oof_sig)
            if sig_pr > best_auc_ds:
                best_auc_ds = sig_pr
                best_model_ds = m_name

        best_models_registry[ds_name] = best_model_ds
        print(f"  --> Top Model for {ds_name}: {best_model_ds} (Dev PR-AUC = {best_auc_ds:.4f})")

        # ----------------------------------------------------------------------
        # Reliability Diagram for Top Model
        # ----------------------------------------------------------------------
        bm = best_model_ds
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")

        cal_curves = [
            ("Uncalibrated (Raw)", np.array(oof_predictions[ds_name]["models"][bm]["raw"])),
            ("Platt Sigmoid", np.array(oof_predictions[ds_name]["models"][bm]["sigmoid"])),
            ("Isotonic", np.array(oof_predictions[ds_name]["models"][bm]["isotonic"])),
            ("Temperature Scaled", np.array(oof_predictions[ds_name]["models"][bm]["temp"]))
        ]

        for label, prob_arr in cal_curves:
            frac_pos, mean_pred = calibration_curve(y_dev, prob_arr, n_bins=8, strategy="quantile")
            ece_val = compute_ece(y_dev, prob_arr)
            ax.plot(mean_pred, frac_pos, marker="o", lw=1.5, label=f"{label} (ECE={ece_val:.3f})")

        ax.set_title(f"Reliability Diagram (Dev 5-Fold OOF) — {ds_name} [{bm}]")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Malignancy Fraction")
        ax.legend(loc="upper left", fontsize=8.5)
        plt.tight_layout()
        fig_path = f"outputs/figures/calibration_{ds_name}.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"  Reliability diagram saved to: {fig_path}")

    # Save Dev CV results table
    df_cv = pd.DataFrame(cv_results)
    df_cv.to_csv("outputs/tables/dev_cv_calibration_comparison.csv", index=False)
    print("\n================================================================================")
    print("DEV 5-FOLD CV CALIBRATION COMPARISON (TOP MODELS):")
    print("================================================================================")
    top_summary = df_cv[df_cv["Calibration"] == "Sigmoid"].sort_values(["Dataset", "Dev_PR_AUC"], ascending=[True, False])
    print(top_summary.to_string(index=False))

    # Save OOF predictions and model registry
    with open("data/processed/dev_oof_predictions.json", "w") as f:
        json.dump(oof_predictions, f)

    with open("data/processed/best_models_registry.json", "w") as f:
        json.dump(best_models_registry, f, indent=2)

    print("\n[SUCCESS] Step 3 finished. Results saved to outputs/tables/ and data/processed/.")


if __name__ == "__main__":
    run_train_calibrate()