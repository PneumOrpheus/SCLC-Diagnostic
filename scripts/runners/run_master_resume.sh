#!/usr/bin/env bash
# Resume script after 2026-05-13 OOM diagnosis.
#
# Baseline 2D models effB0/resnet50/densenet completed folds 0-2 (DAPT+FT) and
# fold 3 DAPT. The fold-3 BL fine-tune init OOM-killed each one. Everything
# from swinv2_tiny_2d onwards never ran (separate ACL bug, fixed). This script:
#   PHASE A  baseline 2D resume: fold 3 FT (from saved DAPT ckpt) + fold 4 full
#   PHASE B  baseline never-started: swinv2_tiny_2d, MIL, 3D — full 5 folds
#   PHASE C  FPN 2D — full 5 folds
#   PHASE D  FPN MIL — FT only, baseline backbone transfer (DAPT cache reuse)
#   PHASE E  FPN 3D — full 5 folds
#
# Pairs with the main.py OOM mitigations applied 2026-05-13:
#   - end-of-fold gc.collect + cuda.empty_cache + loader nulling
#   - RSS logging at fold boundaries (search [RSS] in per-model logs)
#   - new --cv-fold-index K flag (single-fold mode, no aggregate row)
#
# Run inside tmux:
#   tmux new -d -s master \
#     "bash scripts/runners/run_master_resume.sh \
#      results/runs/$(date +%Y-%m-%d_%H-%M)_resume 2>&1 | tee /tmp/master_resume.log"

set -uo pipefail

LOG_ROOT=${1:-results/runs/$(date +%Y-%m-%d_%H-%M)_resume}
mkdir -p "$LOG_ROOT"
SUMMARY="$LOG_ROOT/_summary.txt"

PY=${PY:-/home/hansstem/anaconda3/envs/sclc/bin/python}

BASE_OUTPUT=results/output_master_base
FPN_OUTPUT=results/output_master_fpn
BASE_CKPT=/home/data/trained_models_master_base
FPN_CKPT=/home/data/trained_models_master_fpn

FPN_FLAGS="--use-advanced-fpn --use-det-seg \
  --seg-loss-weight 0.1 --bbox-loss-weight 0.1 \
  --fpn-channels 256 --tfpn-heads 4 --tfpn-layers 1 --tfpn-levels 1"

CV_FLAGS="--cv-folds 5"

run_step () {
    local NAME=$1; shift
    local LOG="$LOG_ROOT/${NAME}.log"
    local TS; TS=$(date +%H:%M:%S)
    echo "[${TS}] === START: ${NAME} ===" | tee -a "$SUMMARY"
    echo "[${TS}]   cmd: $*"              | tee -a "$SUMMARY"
    if "$@" 2>&1 | tee "$LOG"; then
        echo "[$(date +%H:%M:%S)] === DONE:  ${NAME} ===" | tee -a "$SUMMARY"
    else
        echo "[$(date +%H:%M:%S)] === FAIL  ${NAME} rc=${PIPESTATUS[0]} (continuing) ===" \
             | tee -a "$SUMMARY"
    fi
    echo "" | tee -a "$SUMMARY"
}

find_dapt_ckpt () {
    local MODEL=$1
    local FOLD=$2
    ls "${BASE_OUTPUT}/2d/${MODEL}/checkpoints/fold_${FOLD}/2d/${MODEL}/"*dapt_pbest_raw.pth \
        2>/dev/null | head -1
}

echo "=== Resume master run at $(date) ===" | tee "$SUMMARY"
echo "Logs:             $LOG_ROOT"            | tee -a "$SUMMARY"
echo "Baseline output:  $BASE_OUTPUT"         | tee -a "$SUMMARY"
echo "FPN output:       $FPN_OUTPUT"          | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# =============================================================================
# PHASE A — Baseline 2D resume: fold 3 FT + fold 4 full
# =============================================================================
echo "=== PHASE A: Baseline 2D resume (fold 3 FT + fold 4 full) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

declare -a RESUME_2D=(
    "efficientnet_b0_2d  configs/experiments/2d_efficientnet_b0.yaml"
    "resnet50_2d         configs/experiments/2d_resnet50.yaml"
    "densenet121_2d      configs/experiments/2d_densenet121.yaml"
)

for entry in "${RESUME_2D[@]}"; do
    read -r NAME CONFIG <<< "$entry"

    # Fold 3 fine-tune only — DAPT ckpt was saved before OOM.
    DAPT_CKPT=$(find_dapt_ckpt "$NAME" 3)
    if [ -z "$DAPT_CKPT" ]; then
        echo "WARNING: no fold-3 DAPT checkpoint for ${NAME}; running full mode" \
            | tee -a "$SUMMARY"
        # shellcheck disable=SC2086
        run_step "base_${NAME}_fold3_full" \
            "$PY" -m sclc.main \
                --config "$CONFIG" \
                --output-dir "$BASE_OUTPUT" \
                --checkpoint-dir "$BASE_CKPT" \
                --cv-folds 5 --cv-fold-index 3
    else
        # shellcheck disable=SC2086
        run_step "base_${NAME}_fold3_ft" \
            "$PY" -m sclc.main \
                --config "$CONFIG" \
                --output-dir "$BASE_OUTPUT" \
                --checkpoint-dir "$BASE_CKPT" \
                --mode finetune \
                --model-checkpoint "$DAPT_CKPT" \
                --cv-folds 5 --cv-fold-index 3
    fi

    # Fold 4 full (never started).
    # shellcheck disable=SC2086
    run_step "base_${NAME}_fold4_full" \
        "$PY" -m sclc.main \
            --config "$CONFIG" \
            --output-dir "$BASE_OUTPUT" \
            --checkpoint-dir "$BASE_CKPT" \
            --cv-folds 5 --cv-fold-index 4
done

# =============================================================================
# PHASE B — Baseline never-started: swinv2_tiny_2d, MIL, 3D
# =============================================================================
echo "=== PHASE B: Baseline never-started (full 5 folds) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

declare -a NEVER_STARTED=(
    "swinv2_tiny_2d   configs/experiments/2d_swinv2_tiny.yaml"
    "mil_swinv2_tiny  configs/experiments/mil_swinv2_tiny.yaml"
    "swin_unetr       configs/experiments/3d_swin_unetr.yaml"
)

for entry in "${NEVER_STARTED[@]}"; do
    read -r NAME CONFIG <<< "$entry"
    # shellcheck disable=SC2086
    run_step "base_${NAME}" \
        "$PY" -m sclc.main \
            --config "$CONFIG" \
            --output-dir "$BASE_OUTPUT" \
            --checkpoint-dir "$BASE_CKPT" \
            $CV_FLAGS
done

# =============================================================================
# PHASE C — FPN 2D (4 models)
# =============================================================================
echo "=== PHASE C: FPN 2D (full 5 folds) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

declare -a MODELS_2D=(
    "efficientnet_b0_2d  configs/experiments/2d_efficientnet_b0.yaml"
    "resnet50_2d         configs/experiments/2d_resnet50.yaml"
    "densenet121_2d      configs/experiments/2d_densenet121.yaml"
    "swinv2_tiny_2d      configs/experiments/2d_swinv2_tiny.yaml"
)

for entry in "${MODELS_2D[@]}"; do
    read -r NAME CONFIG <<< "$entry"
    # shellcheck disable=SC2086
    run_step "fpn_${NAME}" \
        "$PY" -m sclc.main \
            --config "$CONFIG" \
            --output-dir "$FPN_OUTPUT" \
            --checkpoint-dir "$FPN_CKPT" \
            $FPN_FLAGS $CV_FLAGS
done

# =============================================================================
# PHASE D — FPN MIL (FT only, baseline backbone transfer)
# =============================================================================
echo "=== PHASE D: FPN MIL (FT only, transfer from baseline DAPT) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

declare -a MODELS_MIL=(
    "mil_swinv2_tiny  configs/experiments/mil_swinv2_tiny.yaml"
)

for entry in "${MODELS_MIL[@]}"; do
    read -r NAME CONFIG <<< "$entry"
    DAPT_PATTERN="${BASE_CKPT}/fold_{fold}/mil/${NAME}/model_dapt_best.pth"
    # shellcheck disable=SC2086
    run_step "fpn_${NAME}" \
        "$PY" -m sclc.main \
            --config "$CONFIG" \
            --output-dir "$FPN_OUTPUT" \
            --checkpoint-dir "$FPN_CKPT" \
            --mode finetune \
            --dapt-backbone-pattern "$DAPT_PATTERN" \
            $FPN_FLAGS $CV_FLAGS
done

# =============================================================================
# PHASE E — FPN 3D
# =============================================================================
echo "=== PHASE E: FPN 3D (full 5 folds) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

declare -a MODELS_3D=(
    "swin_unetr  configs/experiments/3d_swin_unetr.yaml"
)

for entry in "${MODELS_3D[@]}"; do
    read -r NAME CONFIG <<< "$entry"
    # shellcheck disable=SC2086
    run_step "fpn_${NAME}" \
        "$PY" -m sclc.main \
            --config "$CONFIG" \
            --output-dir "$FPN_OUTPUT" \
            --checkpoint-dir "$FPN_CKPT" \
            $FPN_FLAGS $CV_FLAGS
done

echo "=== ALL RESUME PHASES DONE at $(date) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "After this completes, run the original run_master.sh PHASE 3-5 steps to" | tee -a "$SUMMARY"
echo "rebuild thesis tables and ablation plots from the now-complete results." | tee -a "$SUMMARY"
