# PowerShell reproduction script
$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "BREAST CANCER DIAGNOSTIC TRIAGE: END-TO-END PIPELINE" -ForegroundColor Cyan
Write-Host "Team 4 | Section A | AMLBT (Kavneer Singh Mann, Divyank Singh, Shreya Singhal)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

Write-Host "`n>>> [1/5] Running Cohort Audit and Leakage-Free Stratified Splitting..." -ForegroundColor Yellow
python src/01_audit_and_split.py

Write-Host "`n>>> [2/5] Extracting Radiomics and Harmonized Feature Sets..." -ForegroundColor Yellow
python src/02_radiomics_and_features.py

Write-Host "`n>>> [3/5] Running Fold-Contained Training & Probabilistic Calibration Audit..." -ForegroundColor Yellow
python src/03_train_calibrate.py

Write-Host "`n>>> [4/5] Executing Clinical Triage Deferral, AUGRC, and Decision Curves..." -ForegroundColor Yellow
python src/04_triage_deferral.py

Write-Host "`n>>> [5/5] Computing SHAP / Log-Odds & Evaluating Untouched Test Sets..." -ForegroundColor Yellow
python src/05_interpret_eval.py

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "PIPELINE VERIFICATION: CHECKING ARTIFACTS" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$tables = @(
    "outputs/tables/dev_cv_calibration_comparison.csv",
    "outputs/tables/dev_triage_cost_workload.csv",
    "outputs/tables/dev_feature_log_odds.csv",
    "outputs/tables/final_test_results.csv"
)

$figures = @(
    "outputs/figures/calibration_BrEaST.png",
    "outputs/figures/calibration_BUS-BRA.png",
    "outputs/figures/calibration_UCI_MMD.png",
    "outputs/figures/augrc_BrEaST.png",
    "outputs/figures/augrc_BUS-BRA.png",
    "outputs/figures/augrc_UCI_MMD.png",
    "outputs/figures/dca_BrEaST.png",
    "outputs/figures/dca_BUS-BRA.png",
    "outputs/figures/dca_UCI_MMD.png",
    "outputs/figures/shap_BrEaST.png",
    "outputs/figures/shap_BUS-BRA.png",
    "outputs/figures/shap_UCI_MMD.png"
)

foreach ($t in $tables) {
    if (Test-Path $t) {
        Write-Host "  [FOUND TABLE]  $t" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING TABLE] $t" -ForegroundColor Red
        exit 1
    }
}

foreach ($f in $figures) {
    if (Test-Path $f) {
        Write-Host "  [FOUND FIGURE] $f" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING FIGURE] $f" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n================================================================================" -ForegroundColor Green
Write-Host "SUCCESS: ALL 4 TABLES AND 12 FIGURES VERIFIED IN outputs/" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green