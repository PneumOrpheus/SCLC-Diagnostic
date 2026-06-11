#!/usr/bin/env bash
# Reproduce the full experiment matrix, then build the thesis results.
#
#   6 models  x  {baseline, FPN}  x  5-fold cross-validation
#
# Each run is the two-phase recipe: DAPT on Lung-PET-CT-Dx, then fine-tune on
# BigLunge (see README.md). The baseline and FPN arms write to separate output
# trees; build_final_results.py consolidates both into results/thesis_final/.
#
# This is a long job — launch it inside tmux. Environment setup and data paths
# are documented in README.md / environment.yaml.

set -euo pipefail

PY=${PY:-python}

# FPN arm: hierarchical-FPN neck + auxiliary detection/segmentation head.
FPN_FLAGS="--use-advanced-fpn --use-det-seg \
  --seg-loss-weight 0.1 --bbox-loss-weight 0.1 \
  --fpn-channels 256 --tfpn-heads 4 --tfpn-layers 1 --tfpn-levels 1"

CONFIGS=(
  2d_efficientnet_b0
  2d_resnet50
  2d_densenet121
  2d_swinv2_tiny
  mil_swinv2_tiny
  3d_swin_unetr
)

for cfg in "${CONFIGS[@]}"; do
  CONFIG="configs/experiments/${cfg}.yaml"

  echo "=== ${cfg}: baseline arm (5-fold) ==="
  "$PY" -m sclc.main --config "$CONFIG" --cv-folds 5 \
      --output-dir results/output_master_base

  echo "=== ${cfg}: FPN arm (5-fold) ==="
  # shellcheck disable=SC2086
  "$PY" -m sclc.main --config "$CONFIG" --cv-folds 5 \
      --output-dir results/output_master_fpn $FPN_FLAGS
done

echo "=== Building results/thesis_final (tables + figures, bootstrap CIs) ==="
"$PY" scripts/build_final_results.py --arms base fpn --n-boot 1000
