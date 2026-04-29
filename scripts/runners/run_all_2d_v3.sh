#!/usr/bin/env bash
# Sequential full-pipeline retrain for all 6 2D models with dapt_epochs=30.
#
# Order (sequential — only ONE 2D process at a time, designed to coexist
# with a parallel SwinUNETR job using the rest of the GPU):
#   1. efficientnet_b0_2d        (ImageNet)
#   2. resnet50_2d               (ImageNet)
#   3. densenet121_2d            (ImageNet)
#   4. swin_tiny_2d              (RadImageNet, Swin)
#   5. resnet50_2d_rin           (RadImageNet, ResNet)
#   6. densenet121_2d_rin        (RadImageNet, DenseNet)
#
# Each model: --mode full (DAPT 30ep on Lung-PET-CT-Dx → DAPT-test inference
# → BigLunge fine-tune 40ep w/ patience-10 → BL-test inference). All caches
# already built (img224_crop96_mp1, mp1, etc.); no --clear-cache.
#
# Usage (in tmux, per CLAUDE.md):
#   bash scripts/run_all_2d_v3.sh runs/2026-04-29_2d_v3
# Recommended:
#   tmux new -d -s all2d "PATH=/home/hansstem/anaconda3/envs/sclc/bin:\$PATH \\
#                          bash scripts/run_all_2d_v3.sh runs/2026-04-29_2d_v3"

set -uo pipefail

LOG_ROOT=${1:-runs/$(date +%Y-%m-%d_%H-%M)_2d_v3}
mkdir -p "$LOG_ROOT"
SUMMARY="$LOG_ROOT/_summary.txt"

declare -a MODELS=(
    "efficientnet_b0_2d  configs/experiments/2d_efficientnet_b0.yaml"
    "resnet50_2d         configs/experiments/2d_resnet50.yaml"
    "densenet121_2d      configs/experiments/2d_densenet121.yaml"
    "swin_tiny_2d        configs/experiments/2d_swin_tiny.yaml"
    "resnet50_2d_rin     configs/experiments/2d_resnet50_rin.yaml"
    "densenet121_2d_rin  configs/experiments/2d_densenet121_rin.yaml"
)

run_one () {
    local NAME=$1
    local CONFIG=$2
    local LOG="$LOG_ROOT/${NAME}.log"
    local TS=$(date +%H:%M:%S)
    echo "[${TS}] === START: ${NAME} (mode full, dapt 30ep) ===" | tee -a "$SUMMARY"
    echo "[${TS}]   cmd: python -m sclc.main --config ${CONFIG}" | tee -a "$SUMMARY"
    if python -m sclc.main --config "$CONFIG" 2>&1 | tee "$LOG"; then
        local END_TS=$(date +%H:%M:%S)
        echo "[${END_TS}] === DONE:  ${NAME} ===" | tee -a "$SUMMARY"
    else
        local RC=${PIPESTATUS[0]}
        local END_TS=$(date +%H:%M:%S)
        echo "[${END_TS}] === FAIL ${NAME} rc=${RC} (continuing) ===" | tee -a "$SUMMARY"
    fi
    echo "" | tee -a "$SUMMARY"
}

echo "=== 2D v3 (dapt_epochs=30) at $(date) ===" | tee "$SUMMARY"
echo "Logs: $LOG_ROOT" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

for entry in "${MODELS[@]}"; do
    set -- $entry
    run_one "$1" "$2"
done

echo "=== Building thesis_results/2d ===" | tee -a "$SUMMARY"
python scripts/build_thesis_results.py --pipeline 2d 2>&1 | tee -a "$SUMMARY"

echo "=== ALL DONE at $(date) ===" | tee -a "$SUMMARY"
