# Thesis results

Generated: `2026-05-04T08:06:20`  ·  Git: `1e14254`

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

### 2D pipeline

| Model | DAPT epochs | DAPT range | FT epochs | FT range |
|---|---|---|---|---|
| EffNet-B0 (2D, ImageNet) | 28 | 2026-04-29T07:48:09 → 2026-04-29T07:53:25 | 18 | 2026-04-29T07:53:39 → 2026-04-29T07:55:52 |
| ResNet-50 (2D, ImageNet) | 30 | 2026-04-29T07:56:17 → 2026-04-29T08:03:40 | 14 | 2026-04-29T08:04:20 → 2026-04-29T08:06:15 |
| ResNet-50 (2D, RadImageNet) | 30 | 2026-04-29T08:28:40 → 2026-04-29T08:35:39 | 18 | 2026-04-29T08:35:52 → 2026-04-29T08:38:23 |
| DenseNet121 (2D, ImageNet) | 25 | 2026-04-29T08:07:02 → 2026-04-29T08:12:38 | 22 | 2026-04-29T08:12:53 → 2026-04-29T08:15:49 |
| DenseNet121 (2D, RadImageNet) | 30 | 2026-04-29T08:39:06 → 2026-04-29T08:47:10 | 19 | 2026-04-29T08:47:25 → 2026-04-29T08:50:23 |
| Swin-Tiny (2D, RadImageNet) | 17 | 2026-04-29T08:16:35 → 2026-04-29T08:21:38 | 35 | 2026-04-29T08:21:52 → 2026-04-29T08:27:56 |
| Radiomics SVM (LPCT-train → LPCT-test / BL-test transfer) | 0 | — | 0 | — |
| Radiomics SVM (BL-train → BL-test in-sample) | 0 | — | 0 | — |
| Radiomics RF (LPCT-train → LPCT-test / BL-test transfer) | 0 | — | 0 | — |
| Radiomics RF (BL-train → BL-test in-sample) | 0 | — | 0 | — |
| Radiomics GB (LPCT-train → LPCT-test / BL-test transfer) | 0 | — | 0 | — |
| Radiomics GB (BL-train → BL-test in-sample) | 0 | — | 0 | — |

## Headline tables

# 2D pipeline — headline test results

Patient-level metrics on each held-out test split. CIs are stratified bootstrap (n_boot=1000) on the patient-level predictions. AUC is one-vs-rest, macro-averaged.


## Overall metrics

| Model | Dataset | n | Accuracy | Balanced Acc | MacroF1 | MacroAUC |
|---|---|---|---|---|---|---|
| EffNet-B0 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 53 | 0.698 [0.585, 0.811] | 0.550 [0.386, 0.724] | 0.526 [0.378, 0.680] | 0.795 [0.685, 0.897] |
| EffNet-B0 (2D, ImageNet) | BigLunge (test) | 43 | 0.465 [0.326, 0.605] | 0.465 [0.321, 0.609] | 0.469 [0.318, 0.610] | 0.641 [0.513, 0.770] |
| ResNet-50 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 53 | 0.736 [0.604, 0.849] | 0.709 [0.554, 0.857] | 0.651 [0.488, 0.799] | 0.815 [0.697, 0.919] |
| ResNet-50 (2D, ImageNet) | BigLunge (test) | 43 | 0.465 [0.326, 0.605] | 0.455 [0.315, 0.595] | 0.444 [0.296, 0.586] | 0.639 [0.510, 0.759] |
| ResNet-50 (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 53 | 0.679 [0.547, 0.792] | 0.635 [0.498, 0.789] | 0.557 [0.420, 0.710] | 0.779 [0.666, 0.888] |
| ResNet-50 (2D, RadImageNet) | BigLunge (test) | 43 | 0.326 [0.186, 0.465] | 0.318 [0.185, 0.455] | 0.309 [0.179, 0.444] | 0.540 [0.415, 0.666] |
| DenseNet121 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 53 | 0.774 [0.679, 0.868] | 0.595 [0.437, 0.770] | 0.614 [0.422, 0.785] | 0.831 [0.721, 0.931] |
| DenseNet121 (2D, ImageNet) | BigLunge (test) | 43 | 0.442 [0.302, 0.605] | 0.443 [0.296, 0.609] | 0.442 [0.286, 0.602] | 0.605 [0.477, 0.726] |
| DenseNet121 (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 53 | 0.604 [0.491, 0.736] | 0.544 [0.396, 0.710] | 0.487 [0.367, 0.628] | 0.738 [0.621, 0.852] |
| DenseNet121 (2D, RadImageNet) | BigLunge (test) | 43 | 0.349 [0.209, 0.488] | 0.340 [0.207, 0.477] | 0.338 [0.210, 0.467] | 0.555 [0.430, 0.680] |
| Swin-Tiny (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 53 | 0.660 [0.547, 0.792] | 0.542 [0.395, 0.724] | 0.513 [0.378, 0.672] | 0.787 [0.660, 0.893] |
| Swin-Tiny (2D, RadImageNet) | BigLunge (test) | 43 | 0.488 [0.349, 0.628] | 0.487 [0.340, 0.627] | 0.479 [0.327, 0.622] | 0.603 [0.468, 0.731] |
| Radiomics SVM (LPCT-train → LPCT-test / BL-test transfer) | Lung-PET-CT-Dx (test) | 53 | 0.698 [0.585, 0.811] | 0.578 | 0.557 [0.396, 0.716] | 0.752 [0.612, 0.874] |
| Radiomics SVM (LPCT-train → LPCT-test / BL-test transfer) | BigLunge (test) | 41 | 0.317 [0.268, 0.341] | 0.310 | 0.160 [0.141, 0.170] | 0.500 [0.366, 0.637] |
| Radiomics SVM (BL-train → BL-test in-sample) | Lung-PET-CT-Dx (test) | 53 | 0.170 [0.094, 0.264] | 0.266 | 0.126 [0.066, 0.195] | 0.378 [0.257, 0.496] |
| Radiomics SVM (BL-train → BL-test in-sample) | BigLunge (test) | 41 | 0.244 [0.098, 0.390] | 0.247 | 0.244 [0.103, 0.374] | 0.464 [0.345, 0.590] |
| Radiomics RF (LPCT-train → LPCT-test / BL-test transfer) | Lung-PET-CT-Dx (test) | 53 | 0.604 [0.491, 0.736] | 0.506 | 0.488 [0.338, 0.647] | 0.744 [0.619, 0.865] |
| Radiomics RF (LPCT-train → LPCT-test / BL-test transfer) | BigLunge (test) | 41 | 0.317 [0.268, 0.341] | 0.310 | 0.160 [0.141, 0.170] | 0.454 [0.323, 0.586] |
| Radiomics RF (BL-train → BL-test in-sample) | Lung-PET-CT-Dx (test) | 53 | 0.660 [0.566, 0.736] | 0.364 | 0.343 [0.252, 0.442] | 0.505 [0.367, 0.645] |
| Radiomics RF (BL-train → BL-test in-sample) | BigLunge (test) | 41 | 0.390 [0.244, 0.537] | 0.386 | 0.381 [0.239, 0.530] | 0.573 [0.433, 0.703] |
| Radiomics GB (LPCT-train → LPCT-test / BL-test transfer) | Lung-PET-CT-Dx (test) | 53 | 0.660 [0.528, 0.792] | 0.635 | 0.573 [0.430, 0.726] | 0.743 [0.602, 0.872] |
| Radiomics GB (LPCT-train → LPCT-test / BL-test transfer) | BigLunge (test) | 41 | 0.341 [0.341, 0.341] | 0.333 | 0.170 [0.170, 0.170] | 0.476 [0.367, 0.601] |
| Radiomics GB (BL-train → BL-test in-sample) | Lung-PET-CT-Dx (test) | 53 | 0.491 [0.377, 0.604] | 0.454 | 0.337 [0.264, 0.411] | 0.572 [0.408, 0.721] |
| Radiomics GB (BL-train → BL-test in-sample) | BigLunge (test) | 41 | 0.415 [0.268, 0.561] | 0.414 | 0.414 [0.245, 0.560] | 0.531 [0.399, 0.666] |

## Per-class F1

| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |
|---|---|---|---|---|
| EffNet-B0 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 0.849 [0.761, 0.932] | 0.353 [0.118, 0.588] | 0.375 [0.000, 0.667] |
| EffNet-B0 (2D, ImageNet) | BigLunge (test) | 0.444 [0.250, 0.606] | 0.500 [0.240, 0.714] | 0.462 [0.214, 0.667] |
| ResNet-50 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 0.836 [0.730, 0.917] | 0.545 [0.182, 0.833] | 0.571 [0.424, 0.727] |
| ResNet-50 (2D, ImageNet) | BigLunge (test) | 0.529 [0.333, 0.688] | 0.286 [0.000, 0.522] | 0.516 [0.307, 0.703] |
| ResNet-50 (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 0.813 [0.690, 0.899] | 0.308 [0.000, 0.615] | 0.552 [0.417, 0.692] |
| ResNet-50 (2D, RadImageNet) | BigLunge (test) | 0.389 [0.194, 0.556] | 0.182 [0.000, 0.417] | 0.357 [0.148, 0.572] |
| DenseNet121 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 0.872 [0.805, 0.937] | 0.444 [0.000, 0.800] | 0.526 [0.235, 0.778] |
| DenseNet121 (2D, ImageNet) | BigLunge (test) | 0.452 [0.240, 0.625] | 0.462 [0.210, 0.692] | 0.414 [0.167, 0.625] |
| DenseNet121 (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 0.750 [0.625, 0.853] | 0.250 [0.000, 0.526] | 0.462 [0.261, 0.667] |
| DenseNet121 (2D, RadImageNet) | BigLunge (test) | 0.364 [0.166, 0.550] | 0.167 [0.000, 0.385] | 0.483 [0.240, 0.688] |
| Swin-Tiny (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 0.800 [0.697, 0.892] | 0.286 [0.000, 0.571] | 0.455 [0.235, 0.696] |
| Swin-Tiny (2D, RadImageNet) | BigLunge (test) | 0.571 [0.378, 0.722] | 0.480 [0.214, 0.692] | 0.385 [0.143, 0.593] |
| Radiomics SVM (LPCT-train → LPCT-test / BL-test transfer) | Lung-PET-CT-Dx (test) | 0.822 [0.727, 0.904] | 0.429 [0.133, 0.714] | 0.421 [0.125, 0.667] |
| Radiomics SVM (LPCT-train → LPCT-test / BL-test transfer) | BigLunge (test) | 0.481 [0.423, 0.509] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Radiomics SVM (BL-train → BL-test in-sample) | Lung-PET-CT-Dx (test) | 0.217 [0.050, 0.385] | 0.160 [0.078, 0.235] | 0.000 [0.000, 0.000] |
| Radiomics SVM (BL-train → BL-test in-sample) | BigLunge (test) | 0.214 [0.000, 0.400] | 0.370 [0.148, 0.593] | 0.148 [0.000, 0.333] |
| Radiomics RF (LPCT-train → LPCT-test / BL-test transfer) | Lung-PET-CT-Dx (test) | 0.754 [0.636, 0.853] | 0.462 [0.167, 0.769] | 0.250 [0.000, 0.476] |
| Radiomics RF (LPCT-train → LPCT-test / BL-test transfer) | BigLunge (test) | 0.481 [0.423, 0.509] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Radiomics RF (BL-train → BL-test in-sample) | Lung-PET-CT-Dx (test) | 0.795 [0.723, 0.857] | 0.000 [0.000, 0.000] | 0.235 [0.000, 0.500] |
| Radiomics RF (BL-train → BL-test in-sample) | BigLunge (test) | 0.438 [0.231, 0.621] | 0.261 [0.000, 0.476] | 0.444 [0.190, 0.643] |
| Radiomics GB (LPCT-train → LPCT-test / BL-test transfer) | Lung-PET-CT-Dx (test) | 0.765 [0.647, 0.861] | 0.500 [0.235, 0.769] | 0.455 [0.210, 0.667] |
| Radiomics GB (LPCT-train → LPCT-test / BL-test transfer) | BigLunge (test) | 0.509 [0.509, 0.509] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Radiomics GB (BL-train → BL-test in-sample) | Lung-PET-CT-Dx (test) | 0.600 [0.456, 0.730] | 0.000 [0.000, 0.000] | 0.410 [0.300, 0.514] |
| Radiomics GB (BL-train → BL-test in-sample) | BigLunge (test) | 0.424 [0.229, 0.606] | 0.400 [0.129, 0.621] | 0.417 [0.174, 0.636] |

## Per-class AUC (one-vs-rest)

| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |
|---|---|---|---|---|
| EffNet-B0 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 0.879 [0.768, 0.965] | 0.791 [0.631, 0.929] | 0.715 [0.525, 0.891] |
| EffNet-B0 (2D, ImageNet) | BigLunge (test) | 0.640 [0.481, 0.795] | 0.695 [0.500, 0.867] | 0.588 [0.402, 0.767] |
| ResNet-50 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 0.914 [0.828, 0.977] | 0.716 [0.443, 0.940] | 0.816 [0.694, 0.924] |
| ResNet-50 (2D, ImageNet) | BigLunge (test) | 0.600 [0.426, 0.757] | 0.692 [0.510, 0.836] | 0.624 [0.440, 0.790] |
| ResNet-50 (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 0.898 [0.805, 0.972] | 0.613 [0.358, 0.844] | 0.826 [0.697, 0.937] |
| ResNet-50 (2D, RadImageNet) | BigLunge (test) | 0.545 [0.364, 0.721] | 0.531 [0.341, 0.708] | 0.545 [0.364, 0.722] |
| DenseNet121 (2D, ImageNet) | Lung-PET-CT-Dx (test) | 0.933 [0.860, 0.984] | 0.734 [0.514, 0.933] | 0.826 [0.662, 0.952] |
| DenseNet121 (2D, ImageNet) | BigLunge (test) | 0.571 [0.386, 0.736] | 0.700 [0.541, 0.862] | 0.543 [0.352, 0.733] |
| DenseNet121 (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 0.819 [0.679, 0.932] | 0.603 [0.387, 0.809] | 0.790 [0.629, 0.934] |
| DenseNet121 (2D, RadImageNet) | BigLunge (test) | 0.545 [0.357, 0.738] | 0.508 [0.331, 0.687] | 0.612 [0.431, 0.776] |
| Swin-Tiny (2D, RadImageNet) | Lung-PET-CT-Dx (test) | 0.839 [0.716, 0.940] | 0.706 [0.457, 0.901] | 0.816 [0.689, 0.934] |
| Swin-Tiny (2D, RadImageNet) | BigLunge (test) | 0.664 [0.490, 0.817] | 0.685 [0.503, 0.844] | 0.460 [0.257, 0.645] |
| Radiomics SVM (LPCT-train → LPCT-test / BL-test transfer) | Lung-PET-CT-Dx (test) | 0.746 [0.588, 0.881] | 0.812 [0.620, 0.965] | 0.699 [0.490, 0.871] |
| Radiomics SVM (LPCT-train → LPCT-test / BL-test transfer) | BigLunge (test) | 0.500 [0.299, 0.709] | 0.266 [0.132, 0.423] | 0.733 [0.569, 0.876] |
| Radiomics SVM (BL-train → BL-test in-sample) | Lung-PET-CT-Dx (test) | 0.419 [0.263, 0.588] | 0.277 [0.085, 0.507] | 0.437 [0.217, 0.626] |
| Radiomics SVM (BL-train → BL-test in-sample) | BigLunge (test) | 0.336 [0.177, 0.532] | 0.508 [0.310, 0.717] | 0.548 [0.365, 0.738] |
| Radiomics RF (LPCT-train → LPCT-test / BL-test transfer) | Lung-PET-CT-Dx (test) | 0.747 [0.600, 0.879] | 0.773 [0.567, 0.943] | 0.712 [0.533, 0.861] |
| Radiomics RF (LPCT-train → LPCT-test / BL-test transfer) | BigLunge (test) | 0.452 [0.267, 0.644] | 0.299 [0.143, 0.470] | 0.611 [0.428, 0.796] |
| Radiomics RF (BL-train → BL-test in-sample) | Lung-PET-CT-Dx (test) | 0.549 [0.368, 0.738] | 0.372 [0.152, 0.621] | 0.592 [0.369, 0.802] |
| Radiomics RF (BL-train → BL-test in-sample) | BigLunge (test) | 0.480 [0.287, 0.680] | 0.558 [0.353, 0.746] | 0.683 [0.508, 0.844] |
| Radiomics GB (LPCT-train → LPCT-test / BL-test transfer) | Lung-PET-CT-Dx (test) | 0.716 [0.563, 0.856] | 0.791 [0.581, 0.957] | 0.722 [0.523, 0.886] |
| Radiomics GB (LPCT-train → LPCT-test / BL-test transfer) | BigLunge (test) | 0.444 [0.276, 0.634] | 0.250 [0.111, 0.405] | 0.733 [0.579, 0.872] |
| Radiomics GB (BL-train → BL-test in-sample) | Lung-PET-CT-Dx (test) | 0.605 [0.426, 0.770] | 0.475 [0.181, 0.759] | 0.636 [0.419, 0.826] |
| Radiomics GB (BL-train → BL-test in-sample) | BigLunge (test) | 0.447 [0.275, 0.630] | 0.544 [0.324, 0.753] | 0.603 [0.413, 0.775] |
