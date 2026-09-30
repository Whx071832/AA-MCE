#!/usr/bin/env bash
# Reproduce the AA-MCE paper: data preparation, every experiment in dependency
# order, then the tables and figures.
#
# Usage: bash scripts/reproduce_paper.sh [--dry-run] [--skip-data-prep]
#        PYTHON=/path/to/python bash scripts/reproduce_paper.sh
#
# Works in the repository root, whatever the current directory is.
#  * Data preparation (src.data.prepare) is idempotent: existing outputs under
#    Data/processed/v1 are kept.
#  * An experiment stage is skipped when outputs/experiments/<run>/completion.json
#    exists. The validation-only ensemble selection is skipped when
#    outputs/experiments/optimized_tagging_validation_v2/ensemble_selection.json exists.
#  * A failed experiment stage is retried once (training runners restart with
#    --overwrite). The script stops if the retry fails as well.
#  * The reporting steps always run.
#  * Logs are written to outputs/logs/<stage>.log.
set -uo pipefail

PYTHON="${PYTHON:-python}"
DRY_RUN=0
SKIP_DATA_PREP=0
ATTEMPTS=2
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-data-prep) SKIP_DATA_PREP=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
LOG_DIR="outputs/logs"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
echo "Repository root: $ROOT"
(( DRY_RUN )) || mkdir -p "$LOG_DIR"

# stage <name> <done marker, or "" to always run> <attempts> <python arguments...>
stage() {
  local name="$1" done="$2" attempts="$3" attempt log
  shift 3
  if [[ -n "$done" && -f "$done" ]]; then
    echo "[skip] $name: $done exists"
    return 0
  fi
  if (( DRY_RUN )); then
    echo "[plan] $name: $PYTHON $*"
    return 0
  fi
  log="$LOG_DIR/$name.log"
  for (( attempt = 1; attempt <= attempts; attempt++ )); do
    echo "[run ] $name (attempt $attempt; log: $log)"
    echo ">>> $PYTHON $*   [$(date '+%Y-%m-%dT%H:%M:%S')]" >>"$log"
    if "$PYTHON" "$@" >>"$log" 2>&1 && [[ -z "$done" || -f "$done" ]]; then
      return 0
    fi
    echo "[fail] $name attempt $attempt; see $log"
  done
  return 1
}

# run <run name> <python arguments...>: experiment stage marked by completion.json
run() {
  local name="$1"
  shift
  stage "$name" "outputs/experiments/$name/completion.json" "$ATTEMPTS" "$@" \
    || { echo "Stopped: $name did not complete." >&2; exit 1; }
}

# 0. Data preparation: only the stages this paper needs (see docs/DATA.md).
if (( SKIP_DATA_PREP )); then
  echo "[skip] data preparation (--skip-data-prep)"
else
  stage prepare_data "" 1 -m src.data.prepare --project-root . --stages aa musicsem fma features tags align manifest \
    || { echo "Stopped: data preparation failed." >&2; exit 1; }
  stage validate_data "" 1 -m src.data.validate_processed --project-root . \
    || { echo "Stopped: data validation failed." >&2; exit 1; }
fi

# 1-7. Experiments in dependency order (later runs read earlier ones).
run robust_multimodal_tagging_v3 -m src.experiments.robust_multimodal_tagging --project-root . --run-name robust_multimodal_tagging_v3 --overwrite
run optimized_tagging_validation_v2 -m src.experiments.optimized_multimodal_tagging --project-root . --run-name optimized_tagging_validation_v2 --stage sweep --overwrite
stage select_tagging_ensemble outputs/experiments/optimized_tagging_validation_v2/ensemble_selection.json "$ATTEMPTS" \
  -m src.experiments.select_tagging_ensemble --project-root . --run-name optimized_tagging_validation_v2 \
  || { echo "Stopped: select_tagging_ensemble did not complete." >&2; exit 1; }
run optimized_multimodal_tagging_final_v1 -m src.experiments.optimized_multimodal_tagging --project-root . --run-name optimized_multimodal_tagging_final_v1 --stage final --overwrite
run advanced_fusion_baselines_v1 -m src.experiments.advanced_fusion_baselines --project-root . --run-name advanced_fusion_baselines_v1 --overwrite
run full_feature_ablation_v1 -m src.experiments.full_feature_ablation --project-root . --run-name full_feature_ablation_v1 --overwrite
run fma_cross_dataset_v1 -m src.experiments.fma_cross_dataset --project-root . --run-name fma_cross_dataset_v1 --overwrite
run extended_modalities_v1 -m src.experiments.extended_modalities --project-root . --run-name extended_modalities_v1 --overwrite
run availability_aware_ensemble_v1 -m src.experiments.availability_aware_ensemble --project-root . --run-name availability_aware_ensemble_v1 --overwrite
run availability_aware_ensemble_5c_v1 -m src.experiments.availability_aware_ensemble --project-root . --section availability_aware_ensemble_5c --run-name availability_aware_ensemble_5c_v1 --overwrite
run availability_aware_ensemble_fma_v1 -m src.experiments.availability_aware_campaigns --project-root . --dataset fma --run-name availability_aware_ensemble_fma_v1 --overwrite
run availability_aware_ensemble_sixview_v1 -m src.experiments.availability_aware_campaigns --project-root . --dataset sixview --run-name availability_aware_ensemble_sixview_v1 --overwrite
run posthoc_calibration_v1 -m src.experiments.posthoc_calibration --project-root . --run-name posthoc_calibration_v1
run robustness_checks_v1 -m src.experiments.robustness_checks --project-root . --run-name robustness_checks_v1
run paper_analysis_v1 -m src.experiments.paper_analysis --project-root . --run-name paper_analysis_v1
run paper_efficiency_v1 -m src.experiments.paper_efficiency --project-root .
run decision_layer_diagnostics_v1 -m src.experiments.decision_layer_diagnostics --project-root . --run-name decision_layer_diagnostics_v1

# 8. Tables and figures (always rebuilt; paths are relative to the repository root).
REPORT_FAILED=0
stage paper_data "" 1 -m src.reporting.paper_data --out outputs/paper/data || REPORT_FAILED=1
stage paper_facts "" 1 -m src.reporting.paper_facts || REPORT_FAILED=1
stage paper_figures "" 1 -m src.reporting.paper_figures --data outputs/paper/data --out outputs/paper/figures || REPORT_FAILED=1
stage final_figures "" 1 -m src.reporting.final_figures --project-root . || REPORT_FAILED=1
stage final_tables "" 1 -m src.reporting.final_tables --project-root . || REPORT_FAILED=1
if (( REPORT_FAILED )); then
  echo "Some reporting steps failed; see $LOG_DIR." >&2
  exit 1
fi
echo "Finished."
