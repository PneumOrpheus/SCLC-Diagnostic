#!/usr/bin/env bash
# run_master_followup.sh — second-pass cleanup after run_master_final.sh.
#
# Phase order (top of queue first, per user request):
#   1. FPN MIL × 5 folds, --mode full
#        Mirrors baseline MIL (which trained DAPT+FT in one process). The
#        baseline-DAPT-transfer shortcut tried in run_master_final.sh B.2 hit
#        a strict load_state_dict mismatch (mil.net.* vs backbone.*+fpn.*)
#        and crashed all 5 folds in <90 s each. Running --mode full makes
#        FPN-MIL symmetric with baseline MIL and with FPN 2D / FPN 3D.
#
#   2. Baseline 2D fold-3 BL-test inference-only fix (effb0 / resnet50 / densenet)
#        The fold-3 gap-fills in A.2 of run_master_final.sh ran with
#        --mode finetune and exited without running BL test inference. The
#        FT pbest_raw checkpoints exist on disk; --mode inference loads them
#        and writes the missing *__fold3_inference_probabilities.json.
#
#   3. Baseline swinv2_tiny_2d retries F0/F1/F2/F4 with --cache-workers 4
#        Original A.1 runs OOM-killed at workers=4 on the cold 256-px cache.
#        F3 succeeded (cache had warmed); the other four need workers=1.
#
#   4. FPN efficientnet_b0_2d retries F1/F3 with --cache-workers 4
#        Same cold-cache OOM pattern at workers=4. F0/F2/F4 already DONE.
#
#   5. FPN swin_unetr re-runs (idempotent; only failed folds)
#        run_master_final.sh B.3 is running these now; skip-on-BL-JSON makes
#        this phase a no-op if all 5 already completed.
#
# Crash-resilience strategy:
#   - Per-fold launches with skip-if-BL-test-JSON-exists, identical to
#     run_master_final.sh. Re-run after any failure; only missing folds re-execute.
#   - Pre-flight guard refuses to start if another sclc.main is alive.
#
# Run inside tmux (non-negotiable per user):
#   pgrep -f sclc.main          # MUST print nothing first
#   tmux new -d -s master_followup \
#     "bash scripts/runners/run_master_followup.sh \
#      results/runs/$(date +%Y-%m-%d_%H-%M)_master_followup 2>&1 | tee /tmp/master_followup.log"

set -uo pipefail

# Safety: refuse to run if another sclc.main is alive.
if pgrep -f 'sclc\.main' > /dev/null 2>&1; then
    echo "ERROR: an sclc.main process is already running. Aborting." >&2
    pgrep -af 'sclc\.main' | head -3 >&2
    exit 2
fi

LOG_ROOT=${1:-results/runs/$(date +%Y-%m-%d_%H-%M)_master_followup}
mkdir -p "$LOG_ROOT"
SUMMARY="$LOG_ROOT/_summary.txt"

PY=${PY:-/home/hansstem/anaconda3/envs/sclc/bin/python}

BASE_OUTPUT=results/output_master_base
FPN_OUTPUT=results/output_master_fpn
BASE_CKPT=/home/data/trained_models_master_base
FPN_CKPT=/home/data/trained_models_master_fpn

# Identical FPN flags to run_master_final.sh — keep in sync.
FPN_FLAGS="--use-advanced-fpn --use-det-seg \
  --seg-loss-weight 0.1 --bbox-loss-weight 0.1 \
  --fpn-channels 256 --tfpn-heads 4 --tfpn-layers 1 --tfpn-levels 1"

# ---- Helpers ---------------------------------------------------------------
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

has_bl_test () {
    local MODEL_DIR=$1 FOLD=$2
    compgen -G "${MODEL_DIR}/*__fold${FOLD}_inference_probabilities.json" > /dev/null
}

# Locate an FT pbest_raw checkpoint (NOT DAPT). Used by phase 2 to drive
# --mode inference from the existing fold-3 FT weights.
find_ft_ckpt () {
    local CKPT_ROOT=$1 OUTPUT_ROOT=$2 PIPELINE=$3 MODEL=$4 FOLD=$5
    local found
    found=$(ls -t "${CKPT_ROOT}/fold_${FOLD}/${PIPELINE}/${MODEL}/"*finetune_pbest_raw.pth \
                 2>/dev/null | head -1)
    if [ -n "$found" ]; then echo "$found"; return; fi
    ls -t "${OUTPUT_ROOT}/${PIPELINE}/${MODEL}/checkpoints/fold_${FOLD}/${PIPELINE}/${MODEL}/"*finetune_pbest_raw.pth \
         2>/dev/null | head -1
}

# ---- Banner ----------------------------------------------------------------
echo "=== run_master_followup.sh at $(date) ===" | tee "$SUMMARY"
echo "Logs:        $LOG_ROOT"          | tee -a "$SUMMARY"
echo "Baseline:    $BASE_OUTPUT"       | tee -a "$SUMMARY"
echo "FPN:         $FPN_OUTPUT"        | tee -a "$SUMMARY"
echo "FPN flags:   $FPN_FLAGS"         | tee -a "$SUMMARY"
echo "Idempotent:  steps whose BL test JSON already exists are skipped" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# =============================================================================
# PHASE 1 — FPN MIL, --mode full (mirrors baseline MIL recipe)
# =============================================================================
echo "=== PHASE 1: FPN MIL (--mode full, per fold) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

MIL_MODEL=mil_swinv2_tiny
MIL_CONFIG=configs/experiments/mil_swinv2_tiny.yaml
MIL_MD="${FPN_OUTPUT}/mil/${MIL_MODEL}"
for FOLD in 0 1 2 3 4; do
    NAME="fpn_${MIL_MODEL}_fold${FOLD}"
    if has_bl_test "$MIL_MD" "$FOLD"; then
        echo "[skip] ${NAME}: BL test JSON for fold ${FOLD} already exists" \
             | tee -a "$SUMMARY"
        continue
    fi
    # shellcheck disable=SC2086
    run_step "$NAME" \
        "$PY" -m sclc.main \
            --config "$MIL_CONFIG" \
            --output-dir "$FPN_OUTPUT" \
            --checkpoint-dir "$FPN_CKPT" \
            $FPN_FLAGS \
            --cv-folds 5 --cv-fold-index "$FOLD"
done

# =============================================================================
# PHASE 2 — Baseline 2D fold-3 BL-test inference-only fix
# =============================================================================
echo "" | tee -a "$SUMMARY"
echo "=== PHASE 2: Baseline 2D fold-3 BL-test inference-only fix ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

declare -a F3_INFERENCE=(
    "efficientnet_b0_2d  configs/experiments/2d_efficientnet_b0.yaml"
    "resnet50_2d         configs/experiments/2d_resnet50.yaml"
    "densenet121_2d      configs/experiments/2d_densenet121.yaml"
)
for ENTRY in "${F3_INFERENCE[@]}"; do
    read -r MODEL CONFIG <<< "$ENTRY"
    NAME="base_${MODEL}_fold3_inference"
    MD="${BASE_OUTPUT}/2d/${MODEL}"
    if has_bl_test "$MD" 3; then
        echo "[skip] ${NAME}: BL test JSON for fold 3 already exists" \
             | tee -a "$SUMMARY"
        continue
    fi
    FT_CKPT=$(find_ft_ckpt "$BASE_CKPT" "$BASE_OUTPUT" 2d "$MODEL" 3)
    if [ -z "$FT_CKPT" ]; then
        echo "ERROR: ${NAME}: no fold-3 FT pbest_raw checkpoint found" \
             | tee -a "$SUMMARY"
        continue
    fi
    run_step "$NAME" \
        "$PY" -m sclc.main \
            --config "$CONFIG" \
            --output-dir "$BASE_OUTPUT" \
            --checkpoint-dir "$BASE_CKPT" \
            --mode inference \
            --model-checkpoint "$FT_CKPT" \
            --cv-folds 5 --cv-fold-index 3
done

# =============================================================================
# PHASE 3 — Baseline swinv2_tiny_2d retries (workers=1 for cold cache)
# =============================================================================
echo "" | tee -a "$SUMMARY"
echo "=== PHASE 3: base_swinv2_tiny_2d retries (--cache-workers 4) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

SW2_MD="${BASE_OUTPUT}/2d/swinv2_tiny_2d"
for FOLD in 0 1 2 4; do
    NAME="base_swinv2_tiny_2d_fold${FOLD}"
    if has_bl_test "$SW2_MD" "$FOLD"; then
        echo "[skip] ${NAME}: BL test JSON for fold ${FOLD} already exists" \
             | tee -a "$SUMMARY"
        continue
    fi
    run_step "$NAME" \
        "$PY" -m sclc.main \
            --config configs/experiments/2d_swinv2_tiny.yaml \
            --output-dir "$BASE_OUTPUT" \
            --checkpoint-dir "$BASE_CKPT" \
            --cache-workers 4 \
            --cv-folds 5 --cv-fold-index "$FOLD"
done

# =============================================================================
# PHASE 4 — FPN efficientnet_b0_2d retries (workers=1 for cold cache)
# =============================================================================
echo "" | tee -a "$SUMMARY"
echo "=== PHASE 4: fpn_efficientnet_b0_2d retries (--cache-workers 4) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

EFFB0_MD="${FPN_OUTPUT}/2d/efficientnet_b0_2d"
for FOLD in 1 3; do
    NAME="fpn_efficientnet_b0_2d_fold${FOLD}"
    if has_bl_test "$EFFB0_MD" "$FOLD"; then
        echo "[skip] ${NAME}: BL test JSON for fold ${FOLD} already exists" \
             | tee -a "$SUMMARY"
        continue
    fi
    # shellcheck disable=SC2086
    run_step "$NAME" \
        "$PY" -m sclc.main \
            --config configs/experiments/2d_efficientnet_b0.yaml \
            --output-dir "$FPN_OUTPUT" \
            --checkpoint-dir "$FPN_CKPT" \
            $FPN_FLAGS \
            --cache-workers 4 \
            --cv-folds 5 --cv-fold-index "$FOLD"
done

# =============================================================================
# PHASE 5 — FPN swin_unetr re-runs (idempotent; no-op if final.sh finished)
# =============================================================================
echo "" | tee -a "$SUMMARY"
echo "=== PHASE 5: fpn_swin_unetr re-runs (idempotent) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

SW_MD="${FPN_OUTPUT}/3d/swin_unetr"
for FOLD in 0 1 2 3 4; do
    NAME="fpn_swin_unetr_fold${FOLD}"
    if has_bl_test "$SW_MD" "$FOLD"; then
        echo "[skip] ${NAME}: BL test JSON for fold ${FOLD} already exists" \
             | tee -a "$SUMMARY"
        continue
    fi
    # shellcheck disable=SC2086
    run_step "$NAME" \
        "$PY" -m sclc.main \
            --config configs/experiments/3d_swin_unetr.yaml \
            --output-dir "$FPN_OUTPUT" \
            --checkpoint-dir "$FPN_CKPT" \
            $FPN_FLAGS \
            --cv-folds 5 --cv-fold-index "$FOLD"
done

# =============================================================================
# Summary
# =============================================================================
echo "" | tee -a "$SUMMARY"
echo "=== ALL STEPS COMPLETE at $(date) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "Per-config BL-test coverage (expect 5/5 for every row):" | tee -a "$SUMMARY"
for ARM in base fpn; do
    OUT=results/output_master_${ARM}
    for PIPE in 2d mil 3d; do
        [ -d "$OUT/$PIPE" ] || continue
        for D in "$OUT/$PIPE/"*/; do
            [ -d "$D" ] || continue
            M=$(basename "$D")
            N=$(ls "$D"/*__fold*_inference_probabilities.json 2>/dev/null | wc -l)
            printf '  %-3s  %-3s  %-22s  BL folds=%d/5\n' "$ARM" "$PIPE" "$M" "$N" \
                | tee -a "$SUMMARY"
        done
    done
done
echo "" | tee -a "$SUMMARY"
echo "Re-run this script if anything is < 5/5 — the skip logic will only touch missing folds." \
     | tee -a "$SUMMARY"
