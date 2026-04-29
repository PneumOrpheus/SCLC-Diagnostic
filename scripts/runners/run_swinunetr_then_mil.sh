#!/usr/bin/env bash
# Chained recovery run: SwinUNETR FT+inference, then MIL FT+inference.
# Both use --mode finetune (resumes from each model's DAPT pbest_raw on disk),
# followed by --mode inference (writes the BL-test row + inference probs).
#
# Usage (in tmux, per CLAUDE.md):
#   tmux new -d -s swin_mil "PATH=/home/hansstem/anaconda3/envs/sclc/bin:\$PATH \\
#                              bash scripts/run_swinunetr_then_mil.sh runs/2026-04-29_swin_mil"

set -uo pipefail

LOG_ROOT=${1:-runs/$(date +%Y-%m-%d_%H-%M)_swin_mil}
mkdir -p "$LOG_ROOT"
SUMMARY="$LOG_ROOT/_summary.txt"
PY=/home/hansstem/anaconda3/envs/sclc/bin/python

SWIN_CONFIG=configs/experiments/3d_swin_unetr.yaml
SWIN_DAPT_PBEST=/home/data/trained_models/3d/swin_unetr/Apr_29_04_dapt_pbest_raw.pth

MIL_CONFIG=configs/experiments/mil_resnet50.yaml
MIL_DAPT_PBEST=/home/data/trained_models/mil/mil_resnet50/Apr_28_04_dapt_pbest_raw.pth

echo "=== SwinUNETR + MIL recovery at $(date) ===" | tee "$SUMMARY"
echo "Logs: $LOG_ROOT" | tee -a "$SUMMARY"

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

# ----- SwinUNETR -----
run_step swin_unetr_finetune  "$PY" -m sclc.main --mode finetune  --config "$SWIN_CONFIG" --model-checkpoint "$SWIN_DAPT_PBEST"

SWIN_FT_PBEST=$(ls -t /home/data/trained_models/3d/swin_unetr/*finetune_pbest_raw.pth 2>/dev/null | head -1)
if [ -n "$SWIN_FT_PBEST" ]; then
    run_step swin_unetr_inference "$PY" -m sclc.main --mode inference --config "$SWIN_CONFIG" --model-checkpoint "$SWIN_FT_PBEST"
else
    echo "[$(date +%H:%M:%S)] === SKIP SwinUNETR inference: no FT pbest on disk ===" | tee -a "$SUMMARY"
fi

# ----- MIL -----
run_step mil_resnet50_finetune  "$PY" -m sclc.main --mode finetune  --config "$MIL_CONFIG" --model-checkpoint "$MIL_DAPT_PBEST"

MIL_FT_PBEST=$(ls -t /home/data/trained_models/mil/mil_resnet50/*finetune_pbest_raw.pth 2>/dev/null | head -1)
if [ -n "$MIL_FT_PBEST" ]; then
    run_step mil_resnet50_inference "$PY" -m sclc.main --mode inference --config "$MIL_CONFIG" --model-checkpoint "$MIL_FT_PBEST"
else
    echo "[$(date +%H:%M:%S)] === SKIP MIL inference: no FT pbest on disk ===" | tee -a "$SUMMARY"
fi

# ----- Rebuild thesis_results -----
echo "=== Building thesis_results/3d and thesis_results/mil ===" | tee -a "$SUMMARY"
"$PY" scripts/build_thesis_results.py --pipeline 3d  2>&1 | tee -a "$SUMMARY"
"$PY" scripts/build_thesis_results.py --pipeline mil 2>&1 | tee -a "$SUMMARY"

echo "=== ALL DONE at $(date) ===" | tee -a "$SUMMARY"
