#!/usr/bin/env bash
# Memory-safe single-stream driver for the BigLunge radiomics extractions
# followed by Phases 2-5 (stability, train/eval, SHAP, headline).
#
# BigLunge volumes are full-thorax CT; after 1mm isotropic resampling each
# patient holds ~400 MB and per-worker working memory peaks at ~3 GB. With
# 60 GB available, ≤6 workers is safe; running 3 perturbation extractions
# in parallel previously triggered SIGKILL (see results/radiomics/_bl_base.log
# from 2026-05-02 11:10).
#
# Usage (always in tmux per CLAUDE.md):
#   tmux new -d -s radio \
#     "PATH=/home/hansstem/anaconda3/envs/sclc/bin:\$PATH \
#      bash scripts/runners/run_radiomics_bl_then_post.sh"
#
# Prerequisites:
#   - results/radiomics/features_lpcd.csv (+ _dilate, _erode) already on disk.
#   - results/splits.json (run scripts/dump_splits.py if missing).
#   - PyRadiomics installed (see docs/radiomics_plan.md Phase 0).

set -uo pipefail

PY=/home/hansstem/anaconda3/envs/sclc/bin/python
ROOT=/home/hansstem/SCLC-Classification
RAD=$ROOT/results/radiomics
LOG=$RAD/_bl_pipeline.log
N_JOBS=6     # tuned for ~60 GB RAM; 8 oversubscribes on dilated full-thorax volumes

cd "$ROOT"

echo "=== radiomics BL pipeline at $(date) — n_jobs=$N_JOBS ===" | tee "$LOG"

run_step () {
    local name=$1; shift
    local out_csv="$RAD/features_${name}.csv"
    if [ -f "$out_csv" ]; then
        echo "[skip] $out_csv exists" | tee -a "$LOG"
        return 0
    fi
    echo "" | tee -a "$LOG"
    echo "=== $name extraction at $(date) ===" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
    if [ -f "$out_csv" ]; then
        echo "[ok] wrote $out_csv ($(wc -l < "$out_csv") lines)" | tee -a "$LOG"
    else
        echo "[FAIL] $out_csv missing after extraction; abort" | tee -a "$LOG"
        exit 1
    fi
}

run_step biglunge       "$PY" -u -m sclc.radiomics.extract --dataset biglunge --n-jobs $N_JOBS
run_step biglunge_dilate "$PY" -u -m sclc.radiomics.extract --dataset biglunge --perturbation dilate --n-jobs $N_JOBS
run_step biglunge_erode  "$PY" -u -m sclc.radiomics.extract --dataset biglunge --perturbation erode  --n-jobs $N_JOBS

echo "" | tee -a "$LOG"
echo "=== handing off to post-extract pipeline (Phase 2-5) ===" | tee -a "$LOG"
bash "$ROOT/scripts/runners/run_radiomics_post_extract.sh" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== ALL DONE at $(date) ===" | tee -a "$LOG"
