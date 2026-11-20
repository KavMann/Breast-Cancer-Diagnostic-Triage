#!/usr/bin/env bash
set -e

echo "================================================================================"
echo "BREAST CANCER DIAGNOSTIC TRIAGE: END-TO-END PIPELINE"
echo "Team 4 | Section A | AMLBT (Kavneer Singh Mann, Divyank Singh, Shreya Singhal)"
echo "================================================================================"

echo ""
echo ">>> [1/5] Running Cohort Audit and Leakage-Free Stratified Splitting..."
python src/01_audit_and_split.py

echo ""
echo ">>> [2/5] Extracting Radiomics and Harmonized Feature Sets..."
python src/02_radiomics_and_features.py

echo ""
echo ">>> [3/5] Running Fold-Contained Training & Probabilistic Calibration Audit..."
python src/03_train_calibrate.py

echo ""
echo ">>> [4/5] Executing Clinical Triage Deferral, AUGRC, and Decision Curves..."
python src/04_triage_deferral.py

echo ""
echo ">>> [5/5] Computing SHAP / Log-Odds & Evaluating Untouched Test Sets..."
python src/05_interpret_eval.py

echo ""
echo "================================================================================"
echo "PIPELINE VERIFICATION: CHECKING ARTIFACTS"
echo "================================================================================"

TABLES=(
  "outputs/tables/dev_cv_calibration_comparison.csv"
  "outputs/tables/dev_triage_cost_workload.csv"
  "outputs/tables/dev_feature_log_odds.csv"
  "outputs/tables/final_test_results.csv"
)

FIGURES=(
  "outputs/figures/calibration_BrEaST.png"
  "outputs/figures/calibration_BUS-BRA.png"
  "outputs/figures/calibration_UCI_MMD.png"
  "outputs/figures/augrc_BrEaST.png"
  "outputs/figures/augrc_BUS-BRA.png"
  "outputs/figures/augrc_UCI_MMD.png"
  "outputs/figures/dca_BrEaST.png"
  "outputs/figures/dca_BUS-BRA.png"
  "outputs/figures/dca_UCI_MMD.png"
  "outputs/figures/shap_BrEaST.png"
  "outputs/figures/shap_BUS-BRA.png"
  "outputs/figures/shap_UCI_MMD.png"
)

for t in "${TABLES[@]}"; do
  if [ -f "$t" ]; then
    echo "  [FOUND TABLE]  $t"
  else
    echo "  [MISSING TABLE] $t"
    exit 1
  fi
done

for f in "${FIGURES[@]}"; do
  if [ -f "$f" ]; then
    echo "  [FOUND FIGURE] $f"
  else
    echo "  [MISSING FIGURE] $f"
    exit 1
  fi
done

echo ""
echo "================================================================================"
echo "SUCCESS: ALL 4 TABLES AND 12 FIGURES VERIFIED IN outputs/"
echo "================================================================================"