#!/usr/bin/env bash
# One-shot re-run of base_swinv2_tiny_2d after the 2026-05-13 ACL fix.
# The original Phase B run failed because the cache parent's poisoned default
# ACL propagated to img256_crop96_mp1/train/ at mode 666 (no exec). ACL is now
# stripped — run this AFTER run_master_resume.sh finishes (don't run in parallel
# with the main resume to avoid GPU contention).
#
# Run inside a fresh tmux pane:
#   tmux new -d -s swinv2_redo \
#     "bash scripts/runners/redo_swinv2_baseline.sh \
#      results/runs/$(date +%Y-%m-%d_%H-%M)_swinv2_redo 2>&1 | tee /tmp/swinv2_redo.log"

set -uo pipefail

LOG_ROOT=${1:-results/runs/$(date +%Y-%m-%d_%H-%M)_swinv2_redo}
mkdir -p "$LOG_ROOT"
SUMMARY="$LOG_ROOT/_summary.txt"

PY=${PY:-/home/hansstem/anaconda3/envs/sclc/bin/python}

BASE_OUTPUT=results/output_master_base
BASE_CKPT=/home/data/trained_models_master_base

echo "=== swinv2_tiny_2d baseline redo at $(date) ===" | tee "$SUMMARY"

LOG="$LOG_ROOT/base_swinv2_tiny_2d.log"
TS=$(date +%H:%M:%S)
echo "[${TS}] === START: base_swinv2_tiny_2d ===" | tee -a "$SUMMARY"
if "$PY" -m sclc.main \
        --config configs/experiments/2d_swinv2_tiny.yaml \
        --output-dir "$BASE_OUTPUT" \
        --checkpoint-dir "$BASE_CKPT" \
        --cv-folds 5 2>&1 | tee "$LOG"; then
    echo "[$(date +%H:%M:%S)] === DONE:  base_swinv2_tiny_2d ===" | tee -a "$SUMMARY"
else
    echo "[$(date +%H:%M:%S)] === FAIL  base_swinv2_tiny_2d rc=${PIPESTATUS[0]} ===" \
         | tee -a "$SUMMARY"
fi
