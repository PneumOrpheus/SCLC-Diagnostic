# Radiomics pipeline — implementation audit and improvement backlog

**Date:** 2026-05-04. **Source code:** `sclc/radiomics/` (extract, stability, train_eval, interpret, plots, run).

This audits what the radiomics pipeline currently does and ranks improvements by expected impact and cost. The framing is *what would I change if I had two more weeks* — not *what's broken*.

## What the pipeline does today

1. **Phase 1 — extraction**: PyRadiomics 3.1.1 (master, Python-3.12 build), IBSI defaults, bin width 25 HU, 1 mm isotropic resampling, largest-CC mask selection, 7 feature classes (shape3D, firstorder, GLCM, GLRLM, GLSZM, GLDM, NGTDM). 107 features per patient. No image filters.
2. **Phase 2 — stability**: ICC(3,1) under ±1-voxel mask perturbation (cutoff 0.75) + near-zero-variance + |r|>0.9 correlation reduction. Result: 107 raw → 40 stable on LPCT-Dx, 34 stable on BL, 18 in the dataset intersection.
3. **Phase 3 — train/eval**: z-score → LASSO L1-multinomial selection capped at √N_train (~16 LPCT, ~14 BL) → SMOTE on training fold → nested 5-fold CV across SVM-RBF, RandomForest, GradientBoosting → refit on train+val → predict on test.
4. **Phase 4 — SHAP**: KernelExplainer for SVM, TreeExplainer for RF/GB. Top-10 by mean(|SHAP|) per class.
5. **Phase 5 — integration**: writes `inference_probabilities_*.json` in the deep-pipeline format; `build_thesis_results.py` consumes them, computes bootstrap-CI metrics, and rebuilds the headline tables.

## Current results (LPCT-Dx test, 53 patients)

| Model | Acc | Macro-F1 | Macro-AUC |
|---|---|---|---|
| **radiomics_rf** | **0.717** | **0.626** | **0.816** |
| radiomics_gb | 0.679 | 0.547 | 0.746 |
| radiomics_svm | 0.642 | 0.564 | 0.781 |
| swin_tiny_2d (best DL) | 0.660 | 0.513 | 0.787 |

Reference: Dunn 2023, same task and dataset, **manual radiologist-corrected bbox + iMRRN auto-seg**, SVM + SMOTE → **0.927 acc**. Our pipeline on auto-seg-only masks lands ~21pp behind. That gap *is* the contribution.

## Audit findings — ranked by expected impact

### ⭐ High impact, low cost

**1. Enable LoG + wavelet image filters (likely +3-8pp acc, ~2-3× extraction time).**
Standard radiomics studies decompose the image with Laplacian-of-Gaussian (σ ∈ {1, 2, 3, 4, 5}) and wavelet filters (8 sub-bands) before computing texture features. This produces ~6× more features (LoG ×5 + wavelet ×8 = ×13 in principle, with stability-filter survival usually ~4-6×). PyRadiomics supports this with one config flag:

```python
EXTRACTOR_SETTINGS = {
    ...,
    "additionalInfo": True,
    # add filters:
}
ex.enableImageTypeByName("LoG", customArgs={"sigma": [1.0, 2.0, 3.0, 4.0, 5.0]})
ex.enableImageTypeByName("Wavelet")
```

Cost: extraction wall time goes from ~30 min to ~90-120 min per dataset.
Risk: more features means LASSO can pick from a richer pool, but the √N cap still limits final feature count. Stability filter still drops most of the wavelet+LoG features that don't survive ±1-voxel mask perturbation, so it self-corrects. Net: almost certainly a net positive.

**2. Document the GB-CV-winner vs RF-test-winner discrepancy.**
Inner-CV macro-F1: GB 0.526, RF 0.495, SVM 0.470 (LPCT-Dx). **Pre-registered winner = GB.** But on held-out test: RF 0.717, GB 0.679, SVM 0.642 (acc). Test-set winner is RF.

This is the standard "the model with best inner-CV doesn't always win on test" small-N artifact. The honest thesis writeup is:
- Headline cell of the table: **GB** (the pre-registered winner).
- RF result reported as a secondary observation, not promoted to "the radiomics result."

If we promote RF as the headline because it scored best on test, we're cherry-picking — exactly what nested CV is supposed to prevent. Resist the urge.

**3. Add ElasticNet (L1+L2) feature selection alongside LASSO.**
LASSO collapses BL-trained models to a single feature (`glcm_Correlation` for all three algos — see `selected_features.png`). That's L1's known behavior on highly-correlated features: keep one, drop the rest. ElasticNet keeps correlated groups, often producing more interpretable feature signatures. Plug-in via `LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga')`. ~30 min code, runs in seconds.

### Medium impact, medium cost

**4. Multi-resolution mask features (whole-tumor + tumor margin).**
Several radiomics studies report 2-4pp gains from concatenating features computed on the tumor proper plus a 5-10mm tumor margin. This captures peri-tumoral parenchymal effects (consolidation, ground-glass) that have been linked to histology. Cost: ~3× extraction time, plus manual margin definition.
Risk: with auto-seg masks, the margin will pick up noise. Validate on held-out val first.

**5. Tune bin width via inner CV.**
Bin width is hardcoded at 25 HU. Common tuning range: 10, 16, 25, 32. Different bin widths probe different texture scales. Doable as an inner-CV hyperparameter: ~5-10 min code.
Risk: bin-width tuning effectively gives ~4× the implicit hyperparameter complexity, increasing overfitting risk on small N.

**6. Compare RFE / mRMR feature selection to LASSO.**
LASSO is greedy; RFE (recursive feature elimination) and mRMR (minimum-redundancy maximum-relevance) often select more diverse feature sets. Implementation: `sklearn.feature_selection.RFE` with the SVM/RF as estimator, target √N features. ~2 hours code, runs in seconds.

**7. Use class-weighted-only baseline (no SMOTE) as an ablation.**
SMOTE can introduce synthetic-sample artefacts that hurt held-out performance. Currently we use SMOTE *and* `class_weight='balanced'`. Adding a `_no_smote` ablation row would isolate the contribution.

### Low impact, low cost

**8. Run on `firstorder` features only as an ablation.** First-order features are the most stable to mask noise. If they get most of the way, that quantifies how much of the signal is "tumor density distribution" vs "tumor texture."

**9. Save fitted classifiers (joblib pickles) so SHAP doesn't refit.**
`interpret.py` currently re-runs `fit_final()` because the classifier isn't pickled. Saves a few minutes per run.

**10. Tighten `outputs/2d/<radiomics_*>/_provenance.json` so `build_thesis_results.py` doesn't overwrite the radiomics-side metadata.**
Build_thesis_results currently rewrites our provenance with the deep-model schema, dropping `selected_features` / `algo` / `trained_on` / `hyperparams`. Workaround in `plots.py` reads from `train_eval_summary.json`. A proper fix: after `build_thesis_results` rebuild, re-write the radiomics-specific keys (or extend `write_provenance` in `build_thesis_results.py` to preserve a known set of "model-family" keys).

### Do not pursue

- **Wavelet-only feature pool** — LoG + wavelet is fine, wavelet alone is no better in the literature.
- **Deep-feature concatenation** (ResNet penultimate + radiomics) — different paper, different scope.
- **Manual feature engineering** (e.g. ratios of percentiles) — domain-specific, hard to defend without ablation.

## Auto-seg-mask ceiling (the real bottleneck)

Every prior 3-class study on Lung-PET-CT-Dx (Dunn 2023, Liu 2020, Shah 2021, Varyukhina 2025) used either manual radiologist segmentation or auto-seg with radiologist bbox correction. **Our masks have neither.** This is a hard ceiling that no ML-side improvement will lift past:

- Shape features are intrinsically mask-quality-bounded (sphericity, flatness, axis lengths derive directly from mask voxels).
- GLCM/GLRLM textures bleed into peri-tumoral parenchyma when the mask is too generous.
- Multifocal masks (~70-77% of BL) require the largest-CC selection rule, which throws away potentially-discriminative metastatic-deposit information.

Realistic upper bound for our pipeline on auto-seg-only: ~0.78-0.82 LPCT-Dx accuracy with full LoG+wavelet + ElasticNet. Anything claiming to match Dunn's 0.927 would be lying.

## Where this work lands in the thesis

Headline table cell: **GB row** (pre-registered winner). Secondary table or appendix: SVM/RF ablations. Discussion section: the auto-seg-mask gap to Dunn 2023 quantifies segmentation-quality dependence — a finding the radiomics literature rarely measures because nearly everyone hand-corrects masks. That's the contribution.
