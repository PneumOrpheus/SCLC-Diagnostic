#!/usr/bin/env bash
# Gap-fill script — run AFTER run_master_resume.sh completes.
#
# Four gaps to fill:
#   1. base_efficientnet_b0_2d  fold 3 FT  (4-thread cache validation OOM in earlier attempts)
#   2. base_resnet50_2d         fold 3 FT  (same OOM mechanism)
#   3. base_densenet121_2d      fold 3 FT  (same OOM mechanism — caught by build_final_results audit)
#   4. base_swinv2_tiny_2d      full 5-fold (never ran — ACL bug, now fixed)
#
# Key precaution: --cache-workers 1 (and 2 for swinv2) replaces the YAML default
# of 4. This is the proven prophylactic against the BL test-cache validation
# spike that killed the prior fold-3 FT attempts at slice ~649/800. Slower cold
# cache builds, but no transient 4×NIfTI memory peak.
#
# Run inside tmux ONLY when no other sclc.main process is active:
#   pgrep -f sclc.main          # should print nothing
#   tmux new -d -s gap_fills \
#     "bash scripts/runners/run_gap_fills.sh \
#      results/runs/$(date +%Y-%m-%d_%H-%M)_gap_fills 2>&1 | tee /tmp/gap_fills.log"

set -uo pipefail

# Safety: refuse to run if another sclc.main is alive. Two concurrent runs
# would contend on the GPU and each likely CUDA-OOM.
if pgrep -f 'sclc\.main' > /dev/null 2>&1; then
    echo "ERROR: an sclc.main process is already running. Aborting." >&2
    pgrep -af 'sclc\.main' | head -3 >&2
    exit 2
fi

LOG_ROOT=${1:-results/runs/$(date +%Y-%m-%d_%H-%M)_gap_fills}
mkdir -p "$LOG_ROOT"
SUMMARY="$LOG_ROOT/_summary.txt"

PY=${PY:-/home/hansstem/anaconda3/envs/sclc/bin/python}

BASE_OUTPUT=results/output_master_base
BASE_CKPT=/home/data/trained_models_master_base

# --cache-workers override: 1 for the FT-only gap fills (definitive safety),
# 2 for swinv2 baseline (compromises some cold-cache parallelism for safety).
CW_FT=1
CW_SWINV2=2

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

echo "=== Gap-fill run at $(date) ===" | tee "$SUMMARY"
echo "Logs:           $LOG_ROOT"          | tee -a "$SUMMARY"
echo "Output root:    $BASE_OUTPUT"       | tee -a "$SUMMARY"
echo "cache_workers:  FT=$CW_FT  swinv2=$CW_SWINV2" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# =============================================================================
# GAP 1 — effB0 fold 3 FT (short job, ~30-60min if BL cache is warm)
# =============================================================================
echo "=== GAP 1: efficientnet_b0_2d fold 3 FT ===" | tee -a "$SUMMARY"
DAPT_CKPT=$(find_dapt_ckpt efficientnet_b0_2d 3)
if [ -z "$DAPT_CKPT" ]; then
    echo "ERROR: no fold-3 DAPT checkpoint for efficientnet_b0_2d" | tee -a "$SUMMARY"
else
    # shellcheck disable=SC2086
    run_step "base_efficientnet_b0_2d_fold3_ft" \
        "$PY" -m sclc.main \
            --config configs/experiments/2d_efficientnet_b0.yaml \
            --output-dir "$BASE_OUTPUT" \
            --checkpoint-dir "$BASE_CKPT" \
            --mode finetune \
            --model-checkpoint "$DAPT_CKPT" \
            --cv-folds 5 --cv-fold-index 3 \
            --cache-workers $CW_FT
fi

# =============================================================================
# GAP 2 — resnet50 fold 3 FT
# =============================================================================
echo "=== GAP 2: resnet50_2d fold 3 FT ===" | tee -a "$SUMMARY"
DAPT_CKPT=$(find_dapt_ckpt resnet50_2d 3)
if [ -z "$DAPT_CKPT" ]; then
    echo "ERROR: no fold-3 DAPT checkpoint for resnet50_2d" | tee -a "$SUMMARY"
else
    # shellcheck disable=SC2086
    run_step "base_resnet50_2d_fold3_ft" \
        "$PY" -m sclc.main \
            --config configs/experiments/2d_resnet50.yaml \
            --output-dir "$BASE_OUTPUT" \
            --checkpoint-dir "$BASE_CKPT" \
            --mode finetune \
            --model-checkpoint "$DAPT_CKPT" \
            --cv-folds 5 --cv-fold-index 3 \
            --cache-workers $CW_FT
fi

# =============================================================================
# GAP 3 — densenet121 fold 3 FT
# =============================================================================
echo "=== GAP 3: densenet121_2d fold 3 FT ===" | tee -a "$SUMMARY"
DAPT_CKPT=$(find_dapt_ckpt densenet121_2d 3)
if [ -z "$DAPT_CKPT" ]; then
    echo "ERROR: no fold-3 DAPT checkpoint for densenet121_2d" | tee -a "$SUMMARY"
else
    # shellcheck disable=SC2086
    run_step "base_densenet121_2d_fold3_ft" \
        "$PY" -m sclc.main \
            --config configs/experiments/2d_densenet121.yaml \
            --output-dir "$BASE_OUTPUT" \
            --checkpoint-dir "$BASE_CKPT" \
            --mode finetune \
            --model-checkpoint "$DAPT_CKPT" \
            --cv-folds 5 --cv-fold-index 3 \
            --cache-workers $CW_FT
fi

# =============================================================================
# GAP 4 — swinv2_tiny_2d baseline (full 5-fold, ~3-4h depending on cache state)
# =============================================================================
echo "=== GAP 4: swinv2_tiny_2d baseline (full 5 folds) ===" | tee -a "$SUMMARY"
# shellcheck disable=SC2086
run_step "base_swinv2_tiny_2d" \
    "$PY" -m sclc.main \
        --config configs/experiments/2d_swinv2_tiny.yaml \
        --output-dir "$BASE_OUTPUT" \
        --checkpoint-dir "$BASE_CKPT" \
        --cv-folds 5 \
        --cache-workers $CW_SWINV2

echo "=== ALL GAP-FILL STEPS DONE at $(date) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "Verify each model now has 5/5 FT folds in metrics.jsonl:"  | tee -a "$SUMMARY"
echo "  for m in efficientnet_b0_2d resnet50_2d densenet121_2d swinv2_tiny_2d; do" | tee -a "$SUMMARY"
echo "    echo -n \"\$m: \"" | tee -a "$SUMMARY"
echo "    grep -oE '\"phase\":\\s*\"test_fold_[0-9]\"' \\" | tee -a "$SUMMARY"
echo "      $BASE_OUTPUT/2d/\$m/metrics.jsonl | sort -u | wc -l" | tee -a "$SUMMARY"
echo "  done" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "Expected: each = 5" | tee -a "$SUMMARY"
