# Breast Cancer Diagnostic Triage — Model Code

Academic machine-learning research prototype. **Not for clinical use or medical decision-making.**

This repository contains the model-related code, configurations, and serialized artifacts from the project archive. It deliberately excludes patient datasets, processed CSVs, reports, and generated plots.

## Contents

- `src/01_audit_and_split.py` — dataset audit and split creation
- `src/02_radiomics_and_features.py` — feature engineering
- `src/03_train_calibrate.py` — model training and calibration
- `src/04_triage_deferral.py` — selective prediction and deferral analysis
- `src/05_interpret_eval.py` — interpretation and evaluation routines
- `export_models.py` — model export script
- `predict.py` — illustrative inference script
- `models/*.joblib` — serialized fitted pipelines
- `models/model_manifest.json` — feature specifications and model metadata
- `configs/config.yaml` — experiment settings
- `requirements.txt` — Python package dependencies
- `run_all.sh`, `run_all.ps1` — pipeline scripts

## Reproducing

The code references files in `data/raw/`, `data/processed/`, and `data/splits/`, which are **not included** in this code-only repository. Obtain appropriately licensed data and regenerate the required intermediate files before running training or inference. The included models are not independently revalidated here.

Saved `.joblib` objects use Python pickle serialization; load only if you trust their origin and use a compatible Python/scikit-learn environment.

## Limitations

The corrected research benchmark has not been independently verified against all saved artifacts. The inference script contains illustrative triage labels rather than clinically validated recommendations. No model in this repository is suitable for autonomous screening, diagnosis, patient clearance, or biopsy decisions.
