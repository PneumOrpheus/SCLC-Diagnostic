# Thesis results

Generated: `2026-05-04T20:44:45`  ·  Git: `de7e895`

All numbers reported in the thesis are derived from the files in this tree. Run `python scripts/build_thesis_results.py` to regenerate from `output/<pipeline>/<model>/metrics.jsonl` plus the inference-probability JSONs in the same directories.

## Layout

```
thesis_results/<pipeline>/
├── per_model/<model>/         per-run training curves + test rows + CMs
├── tables/                    cross-model headline + per-class + summary
└── figures/                   PDF figures, sourced from tables/
```

## Canonical-run identification

`metrics.jsonl` is append-only across attempts. For each (model, phase) we identify the **canonical run** as the LAST monotonic-epoch block in chronological order — i.e., the most recent training attempt. Earlier rows from failed attempts are kept in the source files for audit but are not reflected here.

## Per-model canonical runs (this generation)

### 3D pipeline

| Model | DAPT epochs | DAPT range | FT epochs | FT range |
|---|---|---|---|---|
| SwinUNETR (3D) | 20 | 2026-05-04T15:56:33 → 2026-05-04T19:09:48 | 0 | — |

## Headline tables

# 3D pipeline — headline test results

Patient-level metrics on each held-out test split. CIs are stratified bootstrap (n_boot=1000) on the patient-level predictions. AUC is one-vs-rest, macro-averaged.


## Overall metrics

| Model | Dataset | n | Accuracy | Balanced Acc | MacroF1 | MacroAUC |
|---|---|---|---|---|---|---|
| SwinUNETR (3D) | Lung-PET-CT-Dx (test) | 53 | 0.585 [0.519, 0.679] | 0.328 [0.237, 0.438] | 0.337 [0.233, 0.443] | 0.616 [0.538, 0.695] |

## Per-class F1

| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |
|---|---|---|---|---|
| SwinUNETR (3D) | Lung-PET-CT-Dx (test) | 0.744 [0.639, 0.828] | 0.000 [0.000, 0.000] | 0.267 [0.000, 0.556] |

## Per-class AUC (one-vs-rest)

| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |
|---|---|---|---|---|
| SwinUNETR (3D) | Lung-PET-CT-Dx (test) | 0.597 [0.473, 0.712] | 0.581 [0.449, 0.703] | 0.672 [0.542, 0.793] |
