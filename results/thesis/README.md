# Thesis results

Generated: `2026-04-29T17:39:38`  ·  Git: `9215fbc`

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

### MIL pipeline

| Model | DAPT epochs | DAPT range | FT epochs | FT range |
|---|---|---|---|---|
| MIL ResNet-50 | 12 | 2026-04-28T22:56:36 → 2026-04-28T23:14:40 | 16 | 2026-04-29T10:49:35 → 2026-04-29T11:58:33 |
| MIL Swin-Tiny (RadImageNet) | 13 | 2026-04-29T16:24:54 → 2026-04-29T17:02:34 | 13 | 2026-04-29T17:31:32 → 2026-04-29T17:37:40 |

## Headline tables

# MIL pipeline — headline test results

Patient-level metrics on each held-out test split. CIs are stratified bootstrap (n_boot=1000) on the patient-level predictions. AUC is one-vs-rest, macro-averaged.


## Overall metrics

| Model | Dataset | n | Accuracy | Balanced Acc | MacroF1 | MacroAUC |
|---|---|---|---|---|---|---|
| MIL ResNet-50 | Lung-PET-CT-Dx (test) | 53 | 0.566 [0.452, 0.679] | 0.508 [0.371, 0.653] | 0.450 [0.305, 0.607] | 0.704 [0.570, 0.826] |
| MIL ResNet-50 | BigLunge (test) | 46 | 0.261 [0.152, 0.391] | 0.265 [0.152, 0.390] | 0.262 [0.144, 0.382] | 0.426 [0.310, 0.553] |
| MIL Swin-Tiny (RadImageNet) | Lung-PET-CT-Dx (test) | 53 | 0.491 [0.358, 0.623] | 0.444 [0.288, 0.601] | 0.384 [0.263, 0.519] | 0.708 [0.614, 0.809] |
| MIL Swin-Tiny (RadImageNet) | BigLunge (test) | 46 | 0.304 [0.196, 0.413] | 0.301 [0.190, 0.408] | 0.231 [0.133, 0.328] | 0.450 [0.322, 0.573] |

## Per-class F1

| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |
|---|---|---|---|---|
| MIL ResNet-50 | Lung-PET-CT-Dx (test) | 0.710 [0.582, 0.818] | 0.250 [0.000, 0.667] | 0.389 [0.250, 0.514] |
| MIL ResNet-50 | BigLunge (test) | 0.308 [0.083, 0.519] | 0.256 [0.102, 0.410] | 0.222 [0.000, 0.444] |
| MIL Swin-Tiny (RadImageNet) | Lung-PET-CT-Dx (test) | 0.644 [0.508, 0.776] | 0.154 [0.000, 0.444] | 0.353 [0.188, 0.500] |
| MIL Swin-Tiny (RadImageNet) | BigLunge (test) | 0.431 [0.302, 0.539] | 0.261 [0.000, 0.480] | 0.000 [0.000, 0.000] |

## Per-class AUC (one-vs-rest)

| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |
|---|---|---|---|---|
| MIL ResNet-50 | Lung-PET-CT-Dx (test) | 0.781 [0.637, 0.896] | 0.670 [0.411, 0.911] | 0.662 [0.465, 0.833] |
| MIL ResNet-50 | BigLunge (test) | 0.483 [0.308, 0.669] | 0.440 [0.250, 0.634] | 0.356 [0.198, 0.531] |
| MIL Swin-Tiny (RadImageNet) | Lung-PET-CT-Dx (test) | 0.786 [0.651, 0.904] | 0.702 [0.546, 0.858] | 0.636 [0.465, 0.808] |
| MIL Swin-Tiny (RadImageNet) | BigLunge (test) | 0.485 [0.300, 0.681] | 0.491 [0.308, 0.656] | 0.373 [0.215, 0.542] |
