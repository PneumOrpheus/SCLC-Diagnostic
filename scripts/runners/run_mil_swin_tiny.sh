#!/usr/bin/env bash
# End-to-end runner for the mil_swin_tiny pipeline:
#   DAPT (per-slice SwinTiny2DClassifier on Lung-PET-CT-Dx)
# -> FT  (attention-MIL bag-level on BigLunge, MILSwinTinyClassifier)
# -> Inference on the BigLunge held-out test split
# -> Rebuild thesis_results/mil
#
# Mirrors scripts/runners/run_swinunetr_then_mil.sh in shape and conventions
# (tee'd per-step logs, _summary.txt, fall-through behavior on FT failure).
#
# Usage (always in tmux, per CLAUDE.md):
#   tmux new -d -s mil_swin "PATH=/home/hansstem/anaconda3/envs/sclc/bin:\$PATH \\
#                              bash scripts/runners/run_mil_swin_tiny.sh \\
#                              results/runs/$(date +%Y-%m-%d)_mil_swin_tiny"
#
# Notes:
# - Uses --mode full so DAPT and FT run back-to-back from the same process
#   (the FT phase pulls the in-memory DAPT pbest_raw automatically). This
#   matches how the 2d/3d configs are run end-to-end. To resume from an
#   existing DAPT pbest instead, set MIL_SWIN_DAPT_PBEST and pass --mode
#   finetune (see commented block below).

set -uo pipefail

LOG_ROOT=${1:-results/runs/$(date +%Y-%m-%d_%H-%M)_mil_swin_tiny}
mkdir -p "$LOG_ROOT"
SUMMARY="$LOG_ROOT/_summary.txt"

PY=/home/hansstem/anaconda3/envs/sclc/bin/python
CONFIG=configs/experiments/mil_swin_tiny.yaml

echo "=== mil_swin_tiny full run at $(date) ===" | tee "$SUMMARY"
echo "Logs: $LOG_ROOT" | tee -a "$SUMMARY"
echo "Config: $CONFIG"  | tee -a "$SUMMARY"

run_step () {
    local NAME=$1; shift
    local LOG="$LOG_ROOT/${NAME}.log"
    local TS=$(date +%H:%M:%S)
    echo "[${TS}] === START: ${NAME} ===" | tee -a "$SUMMARY"
    echo "[${TS}]   cmd: $*" | tee -a "$SUMMARY"
    if "$@" 2>&1 | tee "$LOG"; then
        echo "[$(date +%H:%M:%S)] === DONE: ${NAME} ===" | tee -a "$SUMMARY"
    else
        echo "[$(date +%H:%M:%S)] === FAIL ${NAME} rc=${PIPESTATUS[0]} (continuing) ===" | tee -a "$SUMMARY"
    fi
    echo "" | tee -a "$SUMMARY"
}

# ----- Full pipeline (DAPT -> FT -> inference) ---------------------------
run_step mil_swin_tiny_full "$PY" -m sclc.main --mode full --config "$CONFIG"

# Recovery alternative — uncomment to resume from a saved DAPT pbest_raw:
#
# MIL_SWIN_DAPT_PBEST=$(ls -t /home/data/trained_models/mil/mil_swin_tiny/*dapt_pbest_raw.pth 2>/dev/null | head -1)
# if [ -n "$MIL_SWIN_DAPT_PBEST" ]; then
#     run_step mil_swin_tiny_finetune "$PY" -m sclc.main --mode finetune \
#         --config "$CONFIG" --model-checkpoint "$MIL_SWIN_DAPT_PBEST"
#     MIL_SWIN_FT_PBEST=$(ls -t /home/data/trained_models/mil/mil_swin_tiny/*finetune_pbest_raw.pth 2>/dev/null | head -1)
#     if [ -n "$MIL_SWIN_FT_PBEST" ]; then
#         run_step mil_swin_tiny_inference "$PY" -m sclc.main --mode inference \
#             --config "$CONFIG" --model-checkpoint "$MIL_SWIN_FT_PBEST"
#     fi
# fi

# ----- Rebuild thesis_results/mil ---------------------------------------
echo "=== Building thesis_results/mil (includes mil_swin_tiny) ===" | tee -a "$SUMMARY"
"$PY" scripts/build_thesis_results.py --pipeline mil 2>&1 | tee -a "$SUMMARY"

echo "=== ALL DONE at $(date) ===" | tee -a "$SUMMARY"
