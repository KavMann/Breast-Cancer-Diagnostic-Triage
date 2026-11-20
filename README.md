# Breast Cancer Diagnostic Triage — Model Code

Academic machine-learning research prototype. **Not for clinical use or medical decision-making.**

This repository contains model-related Python code, configurations, and serialized artifacts. Datasets, reports and generated plots are **not redistributed**; the original public sources and formal citations are documented below.

## Model code and files

| Path | Purpose |
| --- | --- |
| `src/01_audit_and_split.py` | Cohort audit and split generation |
| `src/02_radiomics_and_features.py` | Ultrasound mask feature extraction |
| `src/03_train_calibrate.py` | Training and probability calibration |
| `src/04_triage_deferral.py` | Selective prediction / deferral |
| `src/05_interpret_eval.py` | Interpretability and evaluation |
| `export_models.py`, `predict.py` | Model exports and illustrative inference |
| `models/*.joblib`, `models/model_manifest.json` | Serialized model pipelines and metadata |
| `configs/config.yaml`, `requirements.txt` | Reproducibility configuration |
| `run_all.sh`, `run_all.ps1` | Pipeline entry points |

## Datasets, download locations, and citations

The model code uses **three distinct public benchmark datasets**, not the older Wisconsin Diagnostic Breast Cancer dataset. Obtain the data from the original hosts and follow each dataset's license/usage policy.

### 1. BrEaST / Breast-Lesions-USG (breast ultrasound)

- **Original source:** [The Cancer Imaging Archive — Breast-Lesions-USG](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/)
- **Dataset DOI:** [10.7937/9WKK-Q141](https://doi.org/10.7937/9WKK-Q141)
- **Associated paper:** Pawłowska, A., et al. (2024). *Curated benchmark dataset for ultrasound based breast lesion analysis.* *Scientific Data*. [https://doi.org/10.1038/s41597-024-02984-z](https://doi.org/10.1038/s41597-024-02984-z)
- **Repository's expected inputs:** Clinical spreadsheet in `data/raw/` with `breast` in its filename and ultrasound images/masks extracted to their expected folders. The original project source also identifies a [clinical spreadsheet](https://www.cancerimagingarchive.net/wp-content/uploads/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx) and [images and masks archive](https://www.cancerimagingarchive.net/wp-content/uploads/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip); use the collection page as the authoritative access point if direct URLs change.
- **Project subset:** The source code filters normal cases and 12 listed post-biopsy / post-surgical artifact case IDs, expecting 240 retained cases (142 benign, 98 malignant). This is the **project's filtered cohort**, not necessarily the full dataset size.

**Citation:** Pawłowska et al. *A curated benchmark dataset for ultrasound based breast lesion analysis (Breast-Lesions-USG), Version 1.* The Cancer Imaging Archive (2024). https://doi.org/10.7937/9WKK-Q141

### 2. BUS-BRA (breast ultrasound)

- **Original source:** [BUS-BRA dataset, Zenodo record 8231412](https://zenodo.org/records/8231412)
- **Dataset DOI:** [10.5281/zenodo.8231412](https://doi.org/10.5281/zenodo.8231412)
- **Related reference implementation:** [wgomezf/BUS-BRA](https://github.com/wgomezf/BUS-BRA)
- **Repository's expected inputs:** Extract the Zenodo dataset and place the relevant metadata table at `data/raw/bus_data.csv`, or retain its CSV within `data/raw/busbra/` as handled by the ingestion script. Image/mask paths must be adjusted to the actual extracted layout.
- **Project cohort:** 1,875 ultrasound images from 1,064 patients, grouped by the `Case` patient identifier. Counts reflect the project audit expectations.

**Required research paper citation:** Gómez-Flores, W., Gregorio-Calas, M. J., & Coelho de Albuquerque Pereira, W. (2024). BUS-BRA: A breast ultrasound dataset for assessing computer-aided diagnosis systems. *Medical Physics, 51*(4), 3110–3123. https://doi.org/10.1002/mp.16812

**Dataset citation:** BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems. Zenodo. https://doi.org/10.5281/zenodo.8231412

The authors explicitly request citation of the *Medical Physics* paper for research based on BUS-BRA.

### 3. UCI Mammographic Mass (mammography tabular features)

- **Original source:** [UCI Machine Learning Repository — Mammographic Mass (#161)](https://archive.ics.uci.edu/dataset/161/mammographic+mass)
- **Dataset DOI:** [10.24432/C53K6Z](https://doi.org/10.24432/C53K6Z)
- **Expected input:** Download `mammographic_masses.data` and place it at `data/raw/mammographic_masses.data`.
- **Original size:** 961 records. The ingestion script interprets `?` as missing, drops missing target `Severity`, and excludes `BI_RADS` as a predictor.
- **License:** CC BY 4.0, subject to attribution and the source terms.

**Citation:** Elter, M. (2007). *Mammographic Mass* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53K6Z

## Reproducing the project

1. Install Python dependencies from `requirements.txt`.
2. Download all three datasets from the cited sources and arrange `data/raw/` as the scripts expect. **Images, masks, and the UCI original data file were not all present in the archived project ZIP**, so a clean run requires sourcing and placing them appropriately.
3. Inspect `src/01_audit_and_split.py` and `src/02_radiomics_and_features.py` for directory structure, dataset-specific column names, image filenames and mask mappings before running.
4. Execute `bash run_all.sh` (Linux/macOS) or `./run_all.ps1` (PowerShell) after the inputs are configured. These commands have not been independently tested in a clean environment in this repository.

The trained `.joblib` models are Python pickle-based files: load only from trusted sources in a compatible Python / scikit-learn environment.

## Scope and limitations

Model inference and evaluation scripts are educational research artifacts. The corrected evaluation was not independently revalidated against every saved model in this repository. Thresholds and suggested triage actions are **not** clinically validated. Do not use the software to screen, diagnose, clear patients or decide on biopsy.
