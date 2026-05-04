# Radiomics + ML baseline pipeline — implementation plan

**Status:** draft. Approved scope below; implementation pending.

**Author/date:** 2026-05-02.

## 1. Goal

Add a radiomics-feature classifier as a row in `results/thesis/2d/tables/headline.csv` next to the deep-learning models. Same patient splits, same test cohort, bootstrap-CI metrics from `scripts/build_thesis_results.py`. Three rows: LPCT-Dx test, BigLunge test, and LPCT-Dx → BigLunge cross-dataset transfer.

## 2. Literature alignment (from `Related_works.tex`)

The radiomic comparators in our SLR set the conventions we should match for direct comparability:

| Study | Task | Features | Classifier | Result |
|---|---|---|---|---|
| Liu 2020 | SCLC vs NSCLC | 396 → LASSO 14, 2D ROI of largest slice, CECT | LR + nomogram | AUC 0.82 → 0.94 with serum markers |
| Shah 2021 | SCLC vs other | IBSI PyRadiomics, 3D ROI, CT | RF + LOOCV + SMOTE | AUC 0.88 |
| Varyukhina 2025 | SCLC vs NSCLC | PyRadiomics, 3D, non-contrast CT | Gradient Boosting, 16 selected | AUC 0.89 |
| Li 2021 | Pairwise 3-class | PyRadiomics, 3D ROI, CECT | FNN | AUC 0.88 ADC-SCC, 0.78 SCC-SCLC |
| **Dunn 2023** | **3-class (ADC/SCLC/SCC) on Lung-PET-CT-Dx** | **3D volumetric, iMRRN + radiologist bbox** | **SVM + SMOTE** | **Acc 92.7%, AUC 0.97** |

**Direct precedent: Dunn et al. \cite{Dunn2023}.** Identical task, identical dataset, identical pipeline shape. Their headline 92.7% accuracy is the number we will be implicitly compared against. Their critical caveat: the iMRRN auto-segmentation failed in **55.3% of cases without radiologist-defined bounding boxes**. The natural framing for our work is therefore: *"what does Dunn's pipeline produce when run on the same dataset's auto-segmentation masks without radiologist correction?"* That is a defensible scientific contribution regardless of whether our numbers beat or trail Dunn's, because it isolates segmentation-quality dependence.

Convergent findings across the radiomic studies:
- **GLCM, GLRLM, GLSZM, GLDM** texture features dominate SCLC prediction (\cite{Liu2020, Shah2021, Li2021, Dunn2023}). These reflect the small-densely-packed-cell architecture of SCLC.
- `glszm_SmallAreaEmphasis`, `glszm_LargeAreaLowGrayLevelEmphasis`, `gldm_GrayLevelVariance` show up across two or three studies.
- SCC vs SCLC is the hardest pair; SCC has the lowest gray-level variance (Varyukhina). This is what we will likely struggle with too.

## 3. Architecture decisions

**Sub-package location:** `sclc/radiomics/` (mirrors `sclc/models/`, `sclc/data/`). Importable, reuses splits + class taxonomy + exclusion lists, no new repo-root directory.

```
sclc/radiomics/
├── __init__.py
├── extract.py        # PyRadiomics extraction with IBSI-compliant config
├── stability.py      # ICC under ±1-voxel mask perturbation
├── train_eval.py     # nested CV: SVM (headline) + RF + GB; SMOTE; LASSO selection
├── interpret.py      # SHAP per-feature importance, feature-name table
└── run.py            # CLI: --dataset {lpcd, biglunge, lpcd_to_biglunge}
```

**New dependency:** `pyradiomics` (verify Python-3.12 compat on the existing conda env; this is a known drift point, plan a 30-minute spike before Phase 1 commits).

**Headline integration option (a):** write under `results/thesis/2d/per_model/radiomics_svm/` (and `radiomics_rf/` etc. for ablations). Treats radiomics as a peer-2D model. No changes to `build_thesis_results.py` needed.

## 4. The auto-seg mask problem

This is the central scientific honesty point. All radiomic comparators in the literature used either (a) manual radiologist segmentation (Liu, Shah, Varyukhina, Li) or (b) auto-seg with radiologist bbox correction (Dunn). **We have neither.** What we have:

- **Lung-PET-CT-Dx** `_mask.nii.gz`: per-series tumor mask, source unverified. Multifocal-handling untested.
- **BigLunge** `_label_tc.nii.gz`: SINTEF auto-seg. **70-77% of patients are multifocal** (median 2 CCs, P95 = 8). 30 patients already excluded with empty/sub-threshold masks (`EMPTY_TUMOR_MASK` in `sclc/data/exclusions.py`).

Mitigations baked into the plan:
1. **Largest-CC selection** using the existing `largest_cc_min50` rule from `sclc/data/transforms.py:CropAroundTumord`. Extract features from the dominant connected component only. Document explicitly.
2. **ICC stability filtering** (Phase 2). Perturb each mask by ±1 voxel 3D dilation/erosion, re-extract features, drop any feature whose ICC(3,1) across the 3 versions falls below 0.75. Standard auto-seg-radiomics mitigation; cited in literature.
3. **Shape features kept but de-emphasized** in the SHAP interpretation. Texture features survive mask noise better than shape descriptors.
4. **Honesty statement in the thesis.** "Our masks are auto-segmented without radiologist correction. The expected effect is degraded shape-feature reliability and depressed peak performance relative to the manual-segmentation literature."

## 5. Pipeline phases

### Phase 1 — Feature extraction (`sclc/radiomics/extract.py`)

For each patient in `results/splits.json`:
1. Load **raw HU** CT (no normalization — radiomics is HU-anchored).
2. Load tumor mask:
   - LPCT-Dx: pick the **thinnest-Z series** via `_z_then_name` (matches the loader rule). Multi-scan averaging is rejected — mixes feature distributions across thicknesses.
   - BigLunge: `<pid>_label_tc.nii.gz`.
3. **Resample both to (1.0, 1.0, 1.0) mm isotropic.** Critical: this differs from the (1.0, 1.0, 2.0) the 2D/MIL pipeline uses. Isotropic is the radiomics convention and shape features (`Sphericity`, `Maximum3DDiameter`, `Flatness`) are voxel-anisotropy-sensitive. Document this discrepancy.
4. Apply largest-CC mask selection.
5. Run PyRadiomics extractor with IBSI-compliant defaults:
   - Feature classes: shape3D, firstorder, GLCM, GLRLM, GLSZM, GLDM, NGTDM
   - Bin width: 25 HU (Dunn, Liu, Shah convention)
   - No image filters in the headline run (LoG/wavelet ablation deferred)
   - Output: ~107 features per patient
6. Output: `results/radiomics/features_<dataset>.csv`, columns = features + `patient_id` + `class_idx` + `split`.

Single-patient extraction ≈ 30-60 s (3D volumetric extraction is the slow path). Parallelize via `joblib.Parallel(n_jobs=cpu_count // 2)`. Wall time ≈ 30-60 min per dataset.

### Phase 2 — Stability filtering (`sclc/radiomics/stability.py`)

For every patient, additionally extract on `mask_dilated_1vox` and `mask_eroded_1vox` (3D ball structuring element). Compute **ICC(3,1)** for each feature across the 3 mask versions. Drop features below 0.75 (standard cutoff in auto-seg radiomics literature).

Then, on the surviving features:
- Drop near-zero-variance (`σ < 0.01` post-z-score).
- Drop one of every `|r| > 0.9` correlated pair (Pearson on training fold).

Outputs: `results/radiomics/stable_features_<dataset>.csv` and a `feature_audit_<dataset>.json` listing dropped features with reasons.

This is the move that makes our auto-seg-only setup defensible: *"we kept only features that survive ±1-voxel mask perturbation."*

### Phase 3 — Train/eval (`sclc/radiomics/train_eval.py`)

For each of three eval modes ({lpcd, biglunge, lpcd_to_biglunge}):

1. **Use `results/splits.json` patient lists. No re-splitting.** Cross-dataset mode trains on all LPCT-Dx (train + val + test combined since LPCT is purely a source distribution for transfer) and tests on BigLunge test.
2. **Z-score normalize** on training fold; apply parameters to val/test.
3. **Feature selection on training fold only**: LASSO logistic regression (`SGDClassifier(loss='log_loss', penalty='l1')` or `LogisticRegressionCV(penalty='l1')`), α picked by inner CV. **Cap at √N_train features** — ~16 for LPCT-Dx (N≈242), ~14 for BigLunge (N≈190). N/feature ratio of ≥10 is the standard rule for radiomics.
4. **SMOTE on training fold** (after feature selection). Matches Dunn + Shah.
5. **Nested 5-fold CV** for model selection across:
   - `SVC(kernel='rbf', class_weight='balanced', probability=True)` — the **headline** classifier. Probabilities via Platt scaling.
   - `RandomForestClassifier(class_weight='balanced')` — RF comparator (matches Shah).
   - `GradientBoostingClassifier()` — GB comparator (matches Varyukhina).
   - Inner CV: hyperparam grid (C and γ for SVM; n_estimators, max_depth for RF/GB).
   - Outer CV: model-family selection by mean macro-F1.
6. **Pre-registered winner-selection rule** (committed before looking at test scores): the model with the highest mean inner-CV macro-F1 is the headline; the other two go into `results/thesis/2d/per_model/radiomics_<rf|gb>/` as ablation rows.
7. Refit winner on full train, evaluate **once** on test.
8. **Output JSON probability format** that matches `sclc/main.py:_save_inference_probabilities`:
   ```
   results/output/radiomics/<radiomics_svm|rf|gb>/<...>_inference_probabilities.json
   ```
   Same key shape (`samples: [{patient_id, volume_id, true_label, probabilities: {...}}]`) so `build_thesis_results.py` consumes it without modification.

### Phase 4 — Interpretation (`sclc/radiomics/interpret.py`)

For the headline (winner) model:
- **SHAP analysis** via `shap.KernelExplainer` (SVM) or `shap.TreeExplainer` (RF/GB).
- Output `results/thesis/2d/per_model/radiomics_<winner>/shap_top10.csv`: top 10 features by mean(|SHAP|), per class.
- Output a per-feature SHAP summary plot (PNG) under `results/thesis/2d/figures/`.

This matches the lit convention (Shah, Varyukhina both used SHAP) and lets the thesis discussion ground in feature semantics.

### Phase 5 — Headline integration

1. Drop `_provenance.json` per radiomics-model dir mirroring the deep-model layout, but with simpler keys (no train/finetune split). Fields:
   ```json
   {
     "model_type": "radiomics_svm",
     "pipeline": "2d",
     "extraction_config": "results/radiomics/extraction_config.json",
     "n_features_pre_filter": 107,
     "n_features_post_icc": 73,
     "n_features_selected": 16,
     "selected_features": [...],
     "winner_inner_cv_macro_f1": 0.62,
     "inference_probs_sources": { "lpcd_test": "...", "biglunge_test": "...", "lpcd_to_biglunge": "..." }
   }
   ```
2. Run `python scripts/build_thesis_results.py --pipeline 2d`. The script reads the new probs JSONs and adds three rows to `results/thesis/2d/tables/headline.csv` with bootstrap CIs.

## 6. Three eval rows and their narrative weight

| Row | Train | Test | Compare against | Narrative |
|---|---|---|---|---|
| 1 | LPCT-Dx train | LPCT-Dx test | Dunn 92.7% | Direct precedent. We expect lower numbers due to auto-seg-only masks; the gap quantifies segmentation-quality dependence. |
| 2 | BigLunge train | BigLunge test | None (novel) | First radiomic 3-class result on BigLunge. Establishes the auto-seg-only baseline on the harder distribution. |
| 3 | LPCT-Dx train | BigLunge test | None (novel) | Cross-institution transfer. Mirrors our deep models' "DAPT-only" column. The most interesting result for the thesis discussion. |

Row 3 is what the thesis discussion section actually wants — radiomic generalization across institutions and scanners is a genuine open question and the literature has only anecdotal external-validation results (Karimpour 2025 saw 100% → 66.26% on transfer).

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| PyRadiomics broken on Python 3.12 | 30-minute spike before Phase 1 commits. If broken: pin older PyRadiomics or fall back to MIRP / radiomics_features. |
| Auto-seg shape features unreliable | ICC stability filter (Phase 2) drops the worst offenders. Honesty statement in thesis. |
| N/feature ratio overfit | Hard cap at √N_train. No exceptions. |
| Multiple-comparison inflation across {SVM, RF, GB} | Pre-register inner-CV winner-selection rule before looking at test scores. Treat the other two as ablations, not headline. |
| LPCT-Dx multi-scan ambiguity | Pick thinnest-Z series per patient (matches loader fix). No averaging. |
| BigLunge ADC class imbalance (≈ 60/60/60 train, but test varies) | SMOTE on training fold + class_weight='balanced'. |
| Resampling-spacing inconsistency vs CNN pipeline (1.0,1.0,1.0 vs 1.0,1.0,2.0) | Document clearly. Radiomics convention requires isotropic; CNN convention does not. Different by design. |
| `build_thesis_results.py` JSON-format coupling | Mimic `_save_inference_probabilities` shape exactly. Add a unit test loading our output and rejecting mismatches. |

## 8. Time-to-ship

| Phase | Code time | Run time |
|---|---|---|
| 0 | PyRadiomics compat spike | 30 min |
| 1 | Extraction | ½ day | 1 h per dataset |
| 2 | Stability filter | ¼ day | 2 h per dataset (3× extraction) |
| 3 | Train/eval | ½ day | minutes |
| 4 | SHAP interpret | ¼ day | minutes |
| 5 | Headline integration | ¼ day | minutes |

**Total: ~1.5 days of code, ~6 h of compute. All on CPU; no GPU required.**

## 9. Decisions committed

- Headline integration: **option (a)** — under `results/thesis/2d/per_model/radiomics_<svm|rf|gb>/`.
- **All three eval rows** produced: LPCD, BigLunge, LPCD→BigLunge.
- **SVM headline + RF/GB ablations** picked by pre-registered inner-CV macro-F1.
- **SMOTE** on training folds.
- **(1.0, 1.0, 1.0) mm isotropic** resampling (radiomics convention; deviates from CNN pipeline by design).
- **Largest-CC mask selection** matching `CropAroundTumord`.
- **ICC stability filter** at 0.75.
- **SHAP interpretation** for the headline model.

## 10. Open questions to resolve during implementation

1. Shall the SHAP plot live under `results/thesis/2d/figures/` (mixed with deep-model figures) or `results/thesis/2d/figures/radiomics/` (sub-folder)? Lean: sub-folder for clarity.
2. Should the thesis discussion section reference `gldm_GrayLevelVariance` (Varyukhina's headline finding) explicitly even if it's not in our top-SHAP list? Lean: yes, with our value as a check.
3. SMOTE before or after LASSO feature selection? Lean: after — feature selection on the natural-distribution training fold is more honest; SMOTE oversamples the minority class for fitting the classifier only.
4. Skip GB if PyRadiomics-side gradient boosting is too slow on this N? Lean: keep all three; XGBoost/LightGBM are fast on N≈200.
