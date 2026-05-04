#!/usr/bin/env bash
# Driver for Phases 2-5 of the radiomics pipeline. Runs after all 6 feature
# CSVs (3 per dataset) are on disk.
#
# Usage (always in tmux per CLAUDE.md):
#   tmux new -d -s rpost \
#     "PATH=/home/hansstem/anaconda3/envs/sclc/bin:\$PATH \
#      bash scripts/runners/run_radiomics_post_extract.sh"
set -uo pipefail

PY=/home/hansstem/anaconda3/envs/sclc/bin/python
ROOT=/home/hansstem/SCLC-Classification
RAD=$ROOT/results/radiomics
LOG=$RAD/_post_extract.log

cd "$ROOT"

echo "=== radiomics post-extract pipeline at $(date) ===" | tee "$LOG"

for ds in lpcd biglunge; do
    for kind in '' _dilate _erode; do
        f="$RAD/features_${ds}${kind}.csv"
        if [ ! -f "$f" ]; then
            echo "[FAIL] missing $f — abort" | tee -a "$LOG"
            exit 1
        fi
    done
done
echo "[ok] all 6 feature CSVs present" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Phase 2: stability (LPCT-Dx) ===" | tee -a "$LOG"
$PY -m sclc.radiomics.stability --dataset lpcd 2>&1 | tee -a "$LOG"
echo "=== Phase 2: stability (BigLunge) ===" | tee -a "$LOG"
$PY -m sclc.radiomics.stability --dataset biglunge 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Phase 3: train_eval ===" | tee -a "$LOG"
$PY -m sclc.radiomics.train_eval 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Phase 4: SHAP interpret ===" | tee -a "$LOG"
$PY -m sclc.radiomics.interpret 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Phase 5: provenance + headline rebuild ===" | tee -a "$LOG"
$PY <<'PY' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
from sclc.radiomics import run as runner
RAD = Path('/home/hansstem/SCLC-Classification/results/radiomics')
with open(RAD / 'train_eval_summary.json') as f:
    summary = json.load(f)
# Try loading interpret output if present.
interp = {}
try:
    from sclc.radiomics import interpret
    # interpret.run writes per-model artifacts already, but we don't persist
    # the run() return; rebuild a stub from disk.
    PMD = Path('/home/hansstem/SCLC-Classification/results/thesis/2d/per_model')
    for m in ('radiomics_svm','radiomics_svm_bl','radiomics_rf','radiomics_rf_bl','radiomics_gb','radiomics_gb_bl'):
        d = PMD / m
        info = {}
        for k, suffix in (('shap_top_csv','shap_top10.csv'),):
            p = d / suffix
            if p.is_file():
                info[k] = str(p)
        for k, suffix in (('summary_plot','radiomics/{m}_shap_summary.png'.format(m=m)),):
            p = Path('/home/hansstem/SCLC-Classification/results/thesis/2d/figures') / suffix
            if p.is_file():
                info[k] = str(p)
        interp[m] = info
except Exception as e:
    print('SHAP-collect failed (non-fatal):', e)
written = runner._write_provenances(summary, interp)
print(f'wrote {len(written)} provenance file(s)')
runner._patch_build_thesis_results()
PY

echo "" | tee -a "$LOG"
echo "=== Phase 5b: rebuild headline tables ===" | tee -a "$LOG"
$PY scripts/build_thesis_results.py --pipeline 2d 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== ALL DONE at $(date) ===" | tee -a "$LOG"
echo "Log: $LOG"
