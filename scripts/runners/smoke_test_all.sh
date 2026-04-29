#!/usr/bin/env bash
# Smoke-test every model × {dapt, finetune}: launches `python -m sclc.main`
# with --testing, kills it after ~30 s, and parses the log for errors.
# Used to verify the post-restructure pipeline starts cleanly for all models;
# does NOT produce real training output.
#
# Usage:
#   bash scripts/runners/smoke_test_all.sh

set -uo pipefail

LOG_ROOT=results/runs/$(date +%Y-%m-%d_%H-%M)_smoke
mkdir -p "$LOG_ROOT"
SUMMARY="$LOG_ROOT/_summary.txt"
PY=/home/hansstem/anaconda3/envs/sclc/bin/python

declare -a MODELS=(
    "efficientnet_b0_2d configs/experiments/2d_efficientnet_b0.yaml /home/data/trained_models/2d/efficientnet_b0_2d/Apr_29_04_dapt_pbest_raw.pth"
    "densenet121_2d     configs/experiments/2d_densenet121.yaml     /home/data/trained_models/2d/densenet121_2d/Apr_29_04_dapt_pbest_raw.pth"
    "resnet50_2d        configs/experiments/2d_resnet50.yaml        /home/data/trained_models/2d/resnet50_2d/Apr_29_04_dapt_pbest_raw.pth"
    "swin_tiny_2d       configs/experiments/2d_swin_tiny.yaml       /home/data/trained_models/2d/swin_tiny_2d/Apr_29_04_dapt_pbest_raw.pth"
    "resnet50_2d_rin    configs/experiments/2d_resnet50_rin.yaml    /home/data/trained_models/2d/resnet50_2d_rin/Apr_29_04_dapt_pbest_raw.pth"
    "densenet121_2d_rin configs/experiments/2d_densenet121_rin.yaml /home/data/trained_models/2d/densenet121_2d_rin/Apr_29_04_dapt_pbest_raw.pth"
    "mil_resnet50       configs/experiments/mil_resnet50.yaml       /home/data/trained_models/mil/mil_resnet50/Apr_28_04_dapt_pbest_raw.pth"
    "swin_unetr         configs/experiments/3d_swin_unetr.yaml      /home/data/trained_models/3d/swin_unetr/Apr_29_04_dapt_pbest_raw.pth"
)

echo "=== Smoke test post-restructure at $(date) ===" | tee "$SUMMARY"
echo "Logs: $LOG_ROOT" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

run_smoke () {
    local NAME=$1; local CONFIG=$2; local MODE=$3; local CKPT=$4
    local LOG="$LOG_ROOT/${NAME}_${MODE}.log"
    local TS=$(date +%H:%M:%S)
    echo "[${TS}] === SMOKE: ${NAME} --mode ${MODE} ===" | tee -a "$SUMMARY"

    local cmd=( "$PY" -m sclc.main --config "$CONFIG" --mode "$MODE" --testing )
    if [ "$MODE" = "finetune" ] || [ "$MODE" = "inference" ]; then
        cmd+=( --model-checkpoint "$CKPT" )
    fi

    # Run with a hard 60-second cap. SIGTERM first, then SIGKILL if needed.
    timeout --signal=TERM --kill-after=5 60 "${cmd[@]}" >"$LOG" 2>&1
    local rc=$?

    # rc=0 means it finished within 60 s (unusual for full runs but possible
    # if --testing data is tiny). rc=124 means timeout fired (expected).
    # Anything else is a real error.
    if grep -qE "^Traceback|Error:|raise [A-Z][a-zA-Z]*Error" "$LOG"; then
        echo "  [FAIL] traceback/error in log:" | tee -a "$SUMMARY"
        grep -E "^Traceback|Error:|raise " "$LOG" | head -3 | sed 's/^/    /' | tee -a "$SUMMARY"
    elif [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
        # 124 = SIGTERM via timeout, 137 = SIGKILL via timeout --kill-after
        echo "  [OK]  reached steady state (killed by timeout, rc=$rc)" | tee -a "$SUMMARY"
    elif [ $rc -eq 0 ]; then
        echo "  [OK]  finished cleanly within 60s, rc=0" | tee -a "$SUMMARY"
    else
        echo "  [FAIL] exited rc=$rc (no traceback found)" | tee -a "$SUMMARY"
    fi
}

for entry in "${MODELS[@]}"; do
    set -- $entry
    run_smoke "$1" "$2" dapt     "$3"
    run_smoke "$1" "$2" finetune "$3"
done

echo "" | tee -a "$SUMMARY"
echo "=== Smoke summary ===" | tee -a "$SUMMARY"
grep -E "\[OK\]|\[FAIL\]" "$SUMMARY" | tee -a /tmp/smoke_summary_$$.txt >/dev/null
total=$(grep -cE "\[OK\]|\[FAIL\]" "$SUMMARY")
ok=$(grep -cE "\[OK\]" "$SUMMARY")
fail=$(grep -cE "\[FAIL\]" "$SUMMARY")
echo "passed: $ok / $total" | tee -a "$SUMMARY"
echo "failed: $fail / $total" | tee -a "$SUMMARY"
echo "=== ALL DONE at $(date) ===" | tee -a "$SUMMARY"
