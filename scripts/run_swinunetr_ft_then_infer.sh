#!/usr/bin/env bash
# Resume SwinUNETR after the JSON-Tensor crash. DAPT finished cleanly and
# the pbest checkpoint is on disk; we just need to do FT + BL-test.
#
# Step 1: --mode finetune from the saved DAPT pbest. Runs FT-40ep with
#         LP freeze=5 + diff-LR, writes FT pbest. (No test inference at
#         end of --mode finetune by main.py design.)
# Step 2: --mode inference using the FT pbest. Writes the phase=test row
#         + the BigLunge inference_probabilities JSON, populating the
#         missing pieces in metrics.jsonl + giving us the data the
#         build_thesis_results AUC code needs.
#
# Usage (in tmux, per CLAUDE.md):
#   PATH=/home/hansstem/anaconda3/envs/sclc/bin:$PATH \
#     bash scripts/run_swinunetr_ft_then_infer.sh runs/2026-04-28_finetune

set -uo pipefail

LOG_ROOT=${1:-runs/$(date +%Y-%m-%d_%H-%M)_swinunetr_resume}
mkdir -p "$LOG_ROOT"
SUMMARY="$LOG_ROOT/_swinunetr_resume_summary.txt"
PY=/home/hansstem/anaconda3/envs/sclc/bin/python

CONFIG=configs/experiments/3d_swin_unetr.yaml
DAPT_PBEST=/home/data/trained_models/3d/swin_unetr/Apr_29_04_dapt_pbest_raw.pth

echo "=== SwinUNETR resume (FT then inference) at $(date) ===" | tee "$SUMMARY"
echo "DAPT pbest: $DAPT_PBEST" | tee -a "$SUMMARY"

# Step 1: fine-tune from the DAPT pbest.
TS=$(date +%H:%M:%S)
echo "[${TS}] === START: FT ===" | tee -a "$SUMMARY"
if "$PY" main.py --mode finetune --config "$CONFIG" --model-checkpoint "$DAPT_PBEST" \
    2>&1 | tee "$LOG_ROOT/swin_unetr_finetune.log"; then
    echo "[$(date +%H:%M:%S)] === DONE: FT ===" | tee -a "$SUMMARY"
else
    echo "[$(date +%H:%M:%S)] === FAIL FT rc=${PIPESTATUS[0]} (continuing to inference) ===" | tee -a "$SUMMARY"
fi

# Step 2: inference using the latest FT pbest. Find it by mtime.
FT_PBEST=$(ls -t /home/data/trained_models/3d/swin_unetr/*finetune_pbest_raw.pth 2>/dev/null | head -1)
if [ -z "$FT_PBEST" ]; then
    echo "[$(date +%H:%M:%S)] === SKIP inference: no finetune_pbest_raw.pth on disk ===" | tee -a "$SUMMARY"
    exit 1
fi

TS=$(date +%H:%M:%S)
echo "[${TS}] === START: inference (ckpt=$FT_PBEST) ===" | tee -a "$SUMMARY"
if "$PY" main.py --mode inference --config "$CONFIG" --model-checkpoint "$FT_PBEST" \
    2>&1 | tee "$LOG_ROOT/swin_unetr_inference.log"; then
    echo "[$(date +%H:%M:%S)] === DONE: inference ===" | tee -a "$SUMMARY"
else
    echo "[$(date +%H:%M:%S)] === FAIL inference rc=${PIPESTATUS[0]} ===" | tee -a "$SUMMARY"
fi

echo "=== ALL DONE at $(date) ===" | tee -a "$SUMMARY"
