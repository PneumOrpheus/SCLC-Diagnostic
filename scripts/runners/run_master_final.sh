#!/usr/bin/env bash
# run_master_final.sh — final continuation after the FPN freeze-backbone fix.
#
# Combines the remaining work from three earlier scripts:
#   - redo_swinv2_baseline.sh  (base_swinv2_tiny_2d — never ran successfully)
#   - run_gap_fills.sh         (base 2D fold-3 FT for effB0/resnet50/densenet)
#   - run_master_resume.sh     (the entire FPN matrix — all 6 configs × 5 folds)
#
# Why this script is needed: the prior FPN runs all crashed at FT epoch 1 with
#   RuntimeError: element 0 of tensors does not require grad
# The root cause was that _HEAD_PREFIXES in sclc/main.py did not include the
# FPN-mode submodule names (`fpn.`, `head.`, `fpn_head.`, `box_head.`,
# `instance_head.`, `att_pool.`), so the LP-FT freeze step put every parameter
# into the backbone group and the optimizer had no trainable leaves. The fix
# is now in place — FPN runs are safe to retry.
#
# Crash-resilience strategy:
#   - Per-fold launches for FPN and swinv2 baseline. If one fold dies (OOM,
#     etc.), the others survive — unlike the original bundled `--cv-folds 5`
#     orchestrator where a single failure killed every remaining fold.
#   - Idempotent: each step checks for the expected BigLunge test inference
#     JSON and skips if present. Re-run the script after a crash and it picks
#     up exactly where it left off.
#   - cache_workers stays at the YAML default (4). Per the user, we intentionally
#     keep this and accept the OOM risk; per-fold launches contain the blast
#     radius if a cache-validation spike kills a process.
#
# Run inside tmux (the user has stated this is non-negotiable):
#   pgrep -f sclc.main          # MUST print nothing first
#   tmux new -d -s master_final \
#     "bash scripts/runners/run_master_final.sh \
#      results/runs/$(date +%Y-%m-%d_%H-%M)_master_final 2>&1 | tee /tmp/master_final.log"

set -uo pipefail

# Safety: refuse to run if another sclc.main is alive. Two concurrent runs
# would contend on the GPU and each likely CUDA-OOM.
if pgrep -f 'sclc\.main' > /dev/null 2>&1; then
    echo "ERROR: an sclc.main process is already running. Aborting." >&2
    pgrep -af 'sclc\.main' | head -3 >&2
    exit 2
fi

LOG_ROOT=${1:-results/runs/$(date +%Y-%m-%d_%H-%M)_master_final}
mkdir -p "$LOG_ROOT"
SUMMARY="$LOG_ROOT/_summary.txt"

PY=${PY:-/home/hansstem/anaconda3/envs/sclc/bin/python}

BASE_OUTPUT=results/output_master_base
FPN_OUTPUT=results/output_master_fpn
BASE_CKPT=/home/data/trained_models_master_base
FPN_CKPT=/home/data/trained_models_master_fpn

# Identical FPN flags to run_master.sh PHASE 2 — keep in sync.
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

# BigLunge test JSON for fold K uses a double-underscore between timestamp and
# `foldK_` (e.g. `..._2026-05-12_20-56-30__fold0_inference_probabilities.json`).
# DAPT test JSON uses `_dapt_foldK_`. The `__foldK_` glob therefore matches
# BL-test only, never DAPT.
has_bl_test () {
    local MODEL_DIR=$1 FOLD=$2
    compgen -G "${MODEL_DIR}/*__fold${FOLD}_inference_probabilities.json" > /dev/null
}

# Locate a DAPT pbest_raw checkpoint, preferring the primary --checkpoint-dir
# location, falling back to the output-dir mirror that gets populated when
# --checkpoint-dir is not writable at runtime. Sort by mtime so we pick the
# most recent one when multiple timestamped copies exist (avoids accidentally
# loading a stale pre-fix checkpoint).
find_dapt_ckpt () {
    local CKPT_ROOT=$1 OUTPUT_ROOT=$2 PIPELINE=$3 MODEL=$4 FOLD=$5
    local found
    found=$(ls -t "${CKPT_ROOT}/fold_${FOLD}/${PIPELINE}/${MODEL}/"*dapt_pbest_raw.pth \
                 2>/dev/null | head -1)
    if [ -n "$found" ]; then echo "$found"; return; fi
    ls -t "${OUTPUT_ROOT}/${PIPELINE}/${MODEL}/checkpoints/fold_${FOLD}/${PIPELINE}/${MODEL}/"*dapt_pbest_raw.pth \
         2>/dev/null | head -1
}

# ---- Banner ----------------------------------------------------------------
echo "=== run_master_final.sh at $(date) ===" | tee "$SUMMARY"
echo "Logs:        $LOG_ROOT"          | tee -a "$SUMMARY"
echo "Baseline:    $BASE_OUTPUT"       | tee -a "$SUMMARY"
echo "FPN:         $FPN_OUTPUT"        | tee -a "$SUMMARY"
echo "FPN flags:   $FPN_FLAGS"         | tee -a "$SUMMARY"
echo "Idempotent:  steps whose BL test JSON already exists are skipped" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# =============================================================================
# PHASE A — Baseline gap-fills
# =============================================================================
echo "=== PHASE A: Baseline gap-fills ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# ---- A.1 swinv2_tiny baseline (per-fold; user-requested first priority) ----
echo "--- A.1: base_swinv2_tiny_2d (full DAPT+FT, per fold) ---" | tee -a "$SUMMARY"
SW2_MD="${BASE_OUTPUT}/2d/swinv2_tiny_2d"
for FOLD in 0 1 2 3 4; do
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
            --cv-folds 5 --cv-fold-index "$FOLD"
done

# ---- A.2 baseline 2D fold-3 FT gap-fills ----------------------------------
echo "" | tee -a "$SUMMARY"
echo "--- A.2: base 2D fold-3 FT gap-fills (effB0 / resnet50 / densenet) ---" | tee -a "$SUMMARY"
declare -a GAP_FILLS=(
    "efficientnet_b0_2d  configs/experiments/2d_efficientnet_b0.yaml"
    "resnet50_2d         configs/experiments/2d_resnet50.yaml"
    "densenet121_2d      configs/experiments/2d_densenet121.yaml"
)
for ENTRY in "${GAP_FILLS[@]}"; do
    read -r MODEL CONFIG <<< "$ENTRY"
    NAME="base_${MODEL}_fold3_ft"
    MD="${BASE_OUTPUT}/2d/${MODEL}"
    if has_bl_test "$MD" 3; then
        echo "[skip] ${NAME}: BL test JSON for fold 3 already exists" \
             | tee -a "$SUMMARY"
        continue
    fi
    DAPT_CKPT=$(find_dapt_ckpt "$BASE_CKPT" "$BASE_OUTPUT" 2d "$MODEL" 3)
    if [ -z "$DAPT_CKPT" ]; then
        echo "ERROR: ${NAME}: no fold-3 DAPT checkpoint found" | tee -a "$SUMMARY"
        continue
    fi
    run_step "$NAME" \
        "$PY" -m sclc.main \
            --config "$CONFIG" \
            --output-dir "$BASE_OUTPUT" \
            --checkpoint-dir "$BASE_CKPT" \
            --mode finetune \
            --model-checkpoint "$DAPT_CKPT" \
            --cv-folds 5 --cv-fold-index 3
done

# =============================================================================
# PHASE B — FPN matrix (now safe after the freeze-backbone fix)
# =============================================================================
echo "" | tee -a "$SUMMARY"
echo "=== PHASE B: FPN matrix (per-fold) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# ---- B.1 FPN 2D × 4 models, full DAPT+FT per fold --------------------------
echo "--- B.1: FPN 2D (per fold) ---" | tee -a "$SUMMARY"
declare -a FPN_2D=(
    "efficientnet_b0_2d  configs/experiments/2d_efficientnet_b0.yaml"
    "resnet50_2d         configs/experiments/2d_resnet50.yaml"
    "densenet121_2d      configs/experiments/2d_densenet121.yaml"
    "swinv2_tiny_2d      configs/experiments/2d_swinv2_tiny.yaml"
)
for ENTRY in "${FPN_2D[@]}"; do
    read -r MODEL CONFIG <<< "$ENTRY"
    MD="${FPN_OUTPUT}/2d/${MODEL}"
    for FOLD in 0 1 2 3 4; do
        NAME="fpn_${MODEL}_fold${FOLD}"
        if has_bl_test "$MD" "$FOLD"; then
            echo "[skip] ${NAME}: BL test JSON for fold ${FOLD} already exists" \
                 | tee -a "$SUMMARY"
            continue
        fi
        # shellcheck disable=SC2086
        run_step "$NAME" \
            "$PY" -m sclc.main \
                --config "$CONFIG" \
                --output-dir "$FPN_OUTPUT" \
                --checkpoint-dir "$FPN_CKPT" \
                $FPN_FLAGS \
                --cv-folds 5 --cv-fold-index "$FOLD"
    done
done

# ---- B.2 FPN MIL — FT only, transfer from baseline MIL DAPT (per fold) ----
echo "" | tee -a "$SUMMARY"
echo "--- B.2: FPN MIL (FT only, baseline DAPT transfer, per fold) ---" | tee -a "$SUMMARY"
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
    DAPT_CKPT=$(find_dapt_ckpt "$BASE_CKPT" "$BASE_OUTPUT" mil "$MIL_MODEL" "$FOLD")
    if [ -z "$DAPT_CKPT" ]; then
        echo "ERROR: ${NAME}: no baseline MIL DAPT checkpoint for fold ${FOLD}" \
             | tee -a "$SUMMARY"
        continue
    fi
    # shellcheck disable=SC2086
    run_step "$NAME" \
        "$PY" -m sclc.main \
            --config "$MIL_CONFIG" \
            --output-dir "$FPN_OUTPUT" \
            --checkpoint-dir "$FPN_CKPT" \
            --mode finetune \
            --model-checkpoint "$DAPT_CKPT" \
            $FPN_FLAGS \
            --cv-folds 5 --cv-fold-index "$FOLD"
done

# ---- B.3 FPN 3D — full DAPT+FT per fold -----------------------------------
echo "" | tee -a "$SUMMARY"
echo "--- B.3: FPN 3D swin_unetr (per fold) ---" | tee -a "$SUMMARY"
SW_MODEL=swin_unetr
SW_CONFIG=configs/experiments/3d_swin_unetr.yaml
SW_MD="${FPN_OUTPUT}/3d/${SW_MODEL}"
for FOLD in 0 1 2 3 4; do
    NAME="fpn_${SW_MODEL}_fold${FOLD}"
    if has_bl_test "$SW_MD" "$FOLD"; then
        echo "[skip] ${NAME}: BL test JSON for fold ${FOLD} already exists" \
             | tee -a "$SUMMARY"
        continue
    fi
    # shellcheck disable=SC2086
    run_step "$NAME" \
        "$PY" -m sclc.main \
            --config "$SW_CONFIG" \
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
