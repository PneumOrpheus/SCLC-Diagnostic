#!/usr/bin/env bash
# Full rerun after the 2026-04-30 fixes:
#   * sclc/data/exclusions.py — drops 30 BigLunge empty-mask patients
#   * sclc/data/loaders.py    — sorts patient scans by Z-spacing
#   * configs/experiments/mil_resnet50.yaml — finetune_lr 1e-5 -> 3e-4,
#       bag_dropout 0.15 -> 0.0, freeze_epochs 5 -> 2 (fixes attention
#       collapse documented in docs/investigation_2026_04_30.md §1.3).
#   * MONAI caches were cleared — first model in each pipeline rebuilds them.
#
# Order:
#   PHASE 1: all 6 2D models, sequential (--mode full = DAPT -> DAPT-test ->
#            BL fine-tune -> BL-test). Each 2D run is ~30-60 min on H100.
#   PHASE 2: mil_resnet50 (--mode full). ~1-2 h. Watch the early-epoch
#            attention.entropy_mean log line: should drop below 0.85 by
#            epoch 5 (was stuck at 0.95-0.98 before the fix).
#   PHASE 3: thesis_results regeneration for both pipelines. Headline
#            tables include MacroAUC + per-class AUC out of the box —
#            build_thesis_results.py reads inference_probabilities_*.json
#            and computes one-vs-rest AUC there.
#
# Usage (always in tmux per CLAUDE.md):
#   tmux new -d -s rerun "PATH=/home/rhoversa/anaconda3/envs/sclc/bin:\$PATH \
#                          bash scripts/runners/run_full_rerun.sh \
#                          results/runs/$(date +%Y-%m-%d)_full_rerun"
#
# To also run mil_swin_tiny, append it to the MIL_CONFIGS array below.
# To resume after a partial run, comment out the already-finished entries
# in MODELS_2D / MIL_CONFIGS — every step is a standalone --mode full run.

set -uo pipefail

LOG_ROOT=${1:-results/runs/$(date +%Y-%m-%d_%H-%M)_full_rerun}
mkdir -p "$LOG_ROOT"
SUMMARY="$LOG_ROOT/_summary.txt"
PY=/home/rhoversa/anaconda3/bin/python

declare -a MODELS_2D=(
    "efficientnet_b0_2d  configs/experiments/2d_efficientnet_b0.yaml"
    "resnet50_2d         configs/experiments/2d_resnet50.yaml"
    "densenet121_2d      configs/experiments/2d_densenet121.yaml"
    "swin_tiny_2d        configs/experiments/2d_swin_tiny.yaml"
    "resnet50_2d_rin     configs/experiments/2d_resnet50_rin.yaml"
    "densenet121_2d_rin  configs/experiments/2d_densenet121_rin.yaml"
)

declare -a MIL_CONFIGS=(
    "mil_resnet50  configs/experiments/mil_resnet50.yaml"
    # "mil_swin_tiny  configs/experiments/mil_swin_tiny.yaml"   # uncomment to also rerun the Swin variant
)

run_step () {
    local NAME=$1; shift
    local LOG="$LOG_ROOT/${NAME}.log"
    local TS=$(date +%H:%M:%S)
    echo "[${TS}] === START: ${NAME} ===" | tee -a "$SUMMARY"
    echo "[${TS}]   cmd: $*"               | tee -a "$SUMMARY"
    if "$@" 2>&1 | tee "$LOG"; then
        echo "[$(date +%H:%M:%S)] === DONE: ${NAME} ===" | tee -a "$SUMMARY"
    else
        echo "[$(date +%H:%M:%S)] === FAIL ${NAME} rc=${PIPESTATUS[0]} (continuing) ===" | tee -a "$SUMMARY"
    fi
    echo "" | tee -a "$SUMMARY"
}

echo "=== Full rerun at $(date) ===" | tee "$SUMMARY"
echo "Logs:    $LOG_ROOT"             | tee -a "$SUMMARY"
echo "Python:  $PY"                   | tee -a "$SUMMARY"
echo ""                               | tee -a "$SUMMARY"

# ----- PHASE 1: 2D ------------------------------------------------------
echo "=== PHASE 1/3: 2D models (${#MODELS_2D[@]} configs, --mode full) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
for entry in "${MODELS_2D[@]}"; do
    set -- $entry
    run_step "$1" "$PY" -m sclc.main --config "$2"
done

# ----- PHASE 2: MIL -----------------------------------------------------
echo "=== PHASE 2/3: MIL (${#MIL_CONFIGS[@]} configs, --mode full) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
for entry in "${MIL_CONFIGS[@]}"; do
    set -- $entry
    run_step "$1" "$PY" -m sclc.main --config "$2"
done

# ----- PHASE 3: results -------------------------------------------------
# build_thesis_results.py rebuilds tables (headline.md, per_class_metrics.csv,
# per_class_f1.csv, training_summary.csv) AND figures (ROC, training curves,
# confusion matrices, accuracy / AUC / F1 bar plots). Bootstrap CIs at
# n_boot=1000. AUC is one-vs-rest, macro-averaged for the headline and
# per-class for the supplementary tables — read straight from the
# inference_probabilities_*.json the training run wrote, so no
# action is needed here to "carry over" AUC: it is computed during the
# results build.
echo "=== PHASE 3/3: thesis_results regeneration (2d + mil) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
run_step build_thesis_2d  "$PY" scripts/build_thesis_results.py --pipeline 2d
run_step build_thesis_mil "$PY" scripts/build_thesis_results.py --pipeline mil

echo "=== ALL DONE at $(date) ===" | tee -a "$SUMMARY"
echo "Summary written to: $SUMMARY"
