#!/usr/bin/env bash
# Ablation master script: baseline (no FPN) vs FPN + det/seg
#
# Trains every non-RadImageNet model twice — once without FPN and once with
# the full HA-FPN + multi-task head — then builds thesis tables for both
# runs and generates paired-comparison ablation figures.
#
# Non-RadImageNet models (excludes *_rin variants):
#   2D:  efficientnet_b0_2d, resnet50_2d, densenet121_2d, swinv2_base_2d
#   MIL: mil_resnet50, mil_swinv2_base
#   3D:  swin_unetr
#
# Output layout (both trees are isolated; original results/ is never touched):
#   results/output_master_base/{pipeline}/{model}/   baseline metrics + probs
#   results/output_master_fpn/{pipeline}/{model}/    FPN metrics + probs
#   results/thesis_master_base/{pipeline}/           baseline tables + figures
#   results/thesis_master_fpn/{pipeline}/            FPN tables + figures
#   results/thesis_master/figures/                   ablation comparison PDFs
#   /home/data/trained_models_master_base/{pipeline}/ baseline checkpoints
#   /home/data/trained_models_master_fpn/{pipeline}/  FPN checkpoints
#
# Usage (always run inside tmux):
#   tmux new -d -s ablation \
#     "PATH=/home/rhoversa/anaconda3/bin:\$PATH \
#      bash scripts/runners/run_master.sh \
#      results/runs/$(date +%Y-%m-%d)_master"
#
# To skip a phase entirely, comment out the corresponding block below.
# To resume mid-run, comment out already-finished model entries in MODELS_*.

set -uo pipefail

LOG_ROOT=${1:-results/runs/$(date +%Y-%m-%d_%H-%M)_master}
mkdir -p "$LOG_ROOT"
SUMMARY="$LOG_ROOT/_summary.txt"

PY=/home/rhoversa/anaconda3/bin/python

# ---- Output roots -----------------------------------------------------------
BASE_OUTPUT=results/output_master_base
FPN_OUTPUT=results/output_master_fpn
BASE_THESIS=results/thesis_master_base
FPN_THESIS=results/thesis_master_fpn
ABLATION_FIGURES=results/thesis_master/figures
BASE_CKPT=/home/data/trained_models_master_base
FPN_CKPT=/home/data/trained_models_master_fpn

# ---- FPN flags (identical to run_fpn_full.sh) -------------------------------
FPN_FLAGS="--use-advanced-fpn --use-det-seg \
  --seg-loss-weight 0.1 --bbox-loss-weight 0.1 \
  --fpn-channels 256 --tfpn-heads 4 --tfpn-layers 1 --tfpn-levels 1"

# ---- Cross-validation -------------------------------------------------------
# 5-fold stratified CV on the BigLunge finetune split. Each fold writes a
# test_fold_k row in metrics.jsonl; an averaged 'test' row is written after
# all folds complete. Set to 1 to revert to the original fixed 70/15/15 split.
CV_FLAGS="--cv-folds 5"

# ---- Model lists (no *_rin variants) ----------------------------------------
declare -a MODELS_2D=(
    "efficientnet_b0_2d  configs/experiments/2d_efficientnet_b0.yaml"
    "resnet50_2d         configs/experiments/2d_resnet50.yaml"
    "densenet121_2d      configs/experiments/2d_densenet121.yaml"
    "swinv2_tiny_2d      configs/experiments/2d_swinv2_tiny.yaml"
    "swinv2_base_2d      configs/experiments/2d_swinv2_base.yaml"
)

declare -a MODELS_MIL=(
    "mil_resnet50      configs/experiments/mil_resnet50.yaml"
    "mil_swinv2_tiny   configs/experiments/mil_swinv2_tiny.yaml"
    "mil_swinv2_base   configs/experiments/mil_swinv2_base.yaml"
)

declare -a MODELS_3D=(
    "swin_unetr  configs/experiments/3d_swin_unetr.yaml"
)

# ---- Helper -----------------------------------------------------------------
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

# ---- Banner -----------------------------------------------------------------
echo "=== Master run at $(date) ===" | tee "$SUMMARY"
echo "Logs:             $LOG_ROOT"     | tee -a "$SUMMARY"
echo "Baseline output:  $BASE_OUTPUT"  | tee -a "$SUMMARY"
echo "FPN output:       $FPN_OUTPUT"   | tee -a "$SUMMARY"
echo "Ablation figures: $ABLATION_FIGURES" | tee -a "$SUMMARY"
echo "FPN flags:        $FPN_FLAGS"    | tee -a "$SUMMARY"
echo ""                                | tee -a "$SUMMARY"

# =============================================================================
# PHASE 1 — Baseline: train without FPN
# =============================================================================
echo "=== PHASE 1/5: Baseline 2D (${#MODELS_2D[@]} models) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

for entry in "${MODELS_2D[@]}"; do
    read -r NAME CONFIG <<< "$entry"
    # shellcheck disable=SC2086
    run_step "base_${NAME}" \
        "$PY" -m sclc.main \
            --config "$CONFIG" \
            --output-dir "$BASE_OUTPUT" \
            --checkpoint-dir "$BASE_CKPT" \
            $CV_FLAGS
done

echo "=== PHASE 1/5: Baseline MIL (${#MODELS_MIL[@]} models) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

for entry in "${MODELS_MIL[@]}"; do
    read -r NAME CONFIG <<< "$entry"
    # shellcheck disable=SC2086
    run_step "base_${NAME}" \
        "$PY" -m sclc.main \
            --config "$CONFIG" \
            --output-dir "$BASE_OUTPUT" \
            --checkpoint-dir "$BASE_CKPT" \
            $CV_FLAGS
done

echo "=== PHASE 1/5: Baseline 3D (${#MODELS_3D[@]} models) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

for entry in "${MODELS_3D[@]}"; do
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
# PHASE 2 — FPN: train with HA-FPN + det/seg
# =============================================================================
echo "=== PHASE 2/5: FPN 2D (${#MODELS_2D[@]} models) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

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

echo "=== PHASE 2/5: FPN MIL (${#MODELS_MIL[@]} models) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

for entry in "${MODELS_MIL[@]}"; do
    read -r NAME CONFIG <<< "$entry"
    # shellcheck disable=SC2086
    run_step "fpn_${NAME}" \
        "$PY" -m sclc.main \
            --config "$CONFIG" \
            --output-dir "$FPN_OUTPUT" \
            --checkpoint-dir "$FPN_CKPT" \
            $FPN_FLAGS $CV_FLAGS
done

echo "=== PHASE 2/5: FPN 3D (${#MODELS_3D[@]} models) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

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

# =============================================================================
# PHASE 3 — Thesis results for the baseline run
# =============================================================================
echo "=== PHASE 3/5: Thesis results — baseline ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

for pipeline in 2d mil 3d; do
    run_step "thesis_base_${pipeline}" \
        "$PY" scripts/build_thesis_results.py \
            --pipeline "$pipeline" \
            --output-root "$BASE_OUTPUT" \
            --results-root "$BASE_THESIS" \
            --checkpoint-root "$BASE_CKPT"
done

# =============================================================================
# PHASE 4 — Thesis results for the FPN run
# =============================================================================
echo "=== PHASE 4/5: Thesis results — FPN ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

for pipeline in 2d mil 3d; do
    run_step "thesis_fpn_${pipeline}" \
        "$PY" scripts/build_thesis_results.py \
            --pipeline "$pipeline" \
            --output-root "$FPN_OUTPUT" \
            --results-root "$FPN_THESIS" \
            --checkpoint-root "$FPN_CKPT"
done

# =============================================================================
# PHASE 5 — Ablation comparison plots (baseline vs FPN)
# =============================================================================
echo "=== PHASE 5/5: Ablation comparison plots ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# BigLunge test comparison
run_step ablation_plots_test \
    "$PY" scripts/ablation_plots.py \
        --baseline-root "$BASE_OUTPUT" \
        --fpn-root      "$FPN_OUTPUT" \
        --figures-dir   "$ABLATION_FIGURES" \
        --phase test

# DAPT test comparison (separate delta + per-class figures)
run_step ablation_plots_dapt \
    "$PY" scripts/ablation_plots.py \
        --baseline-root "$BASE_OUTPUT" \
        --fpn-root      "$FPN_OUTPUT" \
        --figures-dir   "$ABLATION_FIGURES/dapt" \
        --phase dapt_test \
        --only delta per_class

echo "=== ALL DONE at $(date) ===" | tee -a "$SUMMARY"
echo "Summary:          $SUMMARY"
echo "Ablation figures: $ABLATION_FIGURES"
echo "Baseline thesis:  $BASE_THESIS/README.md"
echo "FPN thesis:       $FPN_THESIS/README.md"
