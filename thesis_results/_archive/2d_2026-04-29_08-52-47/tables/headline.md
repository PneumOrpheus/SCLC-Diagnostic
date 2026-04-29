# 2D pipeline — headline test results

Patient-level metrics on each held-out test split. CIs are stratified bootstrap (n_boot=1000) on the patient-level predictions. AUC is one-vs-rest, macro-averaged.


## Overall metrics

| Model | Dataset | n | Accuracy | Balanced Acc | MacroF1 | MacroAUC |
|---|---|---|---|---|---|---|
| EffNet-B0 (2D) | Lung-PET-CT-Dx (test) | 53 | 0.698 [0.585, 0.811] | 0.550 [0.386, 0.724] | 0.526 [0.378, 0.680] | 0.795 [0.685, 0.897] |
| EffNet-B0 (2D) | BigLunge (test) | 43 | 0.465 [0.326, 0.605] | 0.465 [0.321, 0.609] | 0.469 [0.318, 0.610] | 0.641 [0.513, 0.770] |
| ResNet-50 (2D) | Lung-PET-CT-Dx (test) | 53 | 0.736 [0.604, 0.849] | 0.709 [0.554, 0.857] | 0.651 [0.488, 0.799] | 0.815 [0.697, 0.919] |
| ResNet-50 (2D) | BigLunge (test) | 43 | 0.465 [0.326, 0.605] | 0.455 [0.315, 0.595] | 0.444 [0.296, 0.586] | 0.639 [0.510, 0.759] |
| DenseNet121 (2D) | Lung-PET-CT-Dx (test) | 53 | 0.774 [0.679, 0.868] | 0.595 [0.437, 0.770] | 0.614 [0.422, 0.785] | 0.831 [0.721, 0.931] |
| DenseNet121 (2D) | BigLunge (test) | 43 | 0.442 [0.302, 0.605] | 0.443 [0.296, 0.609] | 0.442 [0.286, 0.602] | 0.605 [0.477, 0.726] |
| Swin-Tiny (2D) | Lung-PET-CT-Dx (test) | 53 | 0.660 [0.547, 0.792] | 0.542 [0.395, 0.724] | 0.513 [0.378, 0.672] | 0.787 [0.660, 0.893] |
| Swin-Tiny (2D) | BigLunge (test) | 43 | 0.488 [0.349, 0.628] | 0.487 [0.340, 0.627] | 0.479 [0.327, 0.622] | 0.603 [0.468, 0.731] |

## Per-class F1

| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |
|---|---|---|---|---|
| EffNet-B0 (2D) | Lung-PET-CT-Dx (test) | 0.849 [0.761, 0.932] | 0.353 [0.118, 0.588] | 0.375 [0.000, 0.667] |
| EffNet-B0 (2D) | BigLunge (test) | 0.444 [0.250, 0.606] | 0.500 [0.240, 0.714] | 0.462 [0.214, 0.667] |
| ResNet-50 (2D) | Lung-PET-CT-Dx (test) | 0.836 [0.730, 0.917] | 0.545 [0.182, 0.833] | 0.571 [0.424, 0.727] |
| ResNet-50 (2D) | BigLunge (test) | 0.529 [0.333, 0.688] | 0.286 [0.000, 0.522] | 0.516 [0.307, 0.703] |
| DenseNet121 (2D) | Lung-PET-CT-Dx (test) | 0.872 [0.805, 0.937] | 0.444 [0.000, 0.800] | 0.526 [0.235, 0.778] |
| DenseNet121 (2D) | BigLunge (test) | 0.452 [0.240, 0.625] | 0.462 [0.210, 0.692] | 0.414 [0.167, 0.625] |
| Swin-Tiny (2D) | Lung-PET-CT-Dx (test) | 0.800 [0.697, 0.892] | 0.286 [0.000, 0.571] | 0.455 [0.235, 0.696] |
| Swin-Tiny (2D) | BigLunge (test) | 0.571 [0.378, 0.722] | 0.480 [0.214, 0.692] | 0.385 [0.143, 0.593] |

## Per-class AUC (one-vs-rest)

| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |
|---|---|---|---|---|
| EffNet-B0 (2D) | Lung-PET-CT-Dx (test) | 0.879 [0.768, 0.965] | 0.791 [0.631, 0.929] | 0.715 [0.525, 0.891] |
| EffNet-B0 (2D) | BigLunge (test) | 0.640 [0.481, 0.795] | 0.695 [0.500, 0.867] | 0.588 [0.402, 0.767] |
| ResNet-50 (2D) | Lung-PET-CT-Dx (test) | 0.914 [0.828, 0.977] | 0.716 [0.443, 0.940] | 0.816 [0.694, 0.924] |
| ResNet-50 (2D) | BigLunge (test) | 0.600 [0.426, 0.757] | 0.692 [0.510, 0.836] | 0.624 [0.440, 0.790] |
| DenseNet121 (2D) | Lung-PET-CT-Dx (test) | 0.933 [0.860, 0.984] | 0.734 [0.514, 0.933] | 0.826 [0.662, 0.952] |
| DenseNet121 (2D) | BigLunge (test) | 0.571 [0.386, 0.736] | 0.700 [0.541, 0.862] | 0.543 [0.352, 0.733] |
| Swin-Tiny (2D) | Lung-PET-CT-Dx (test) | 0.839 [0.716, 0.940] | 0.706 [0.457, 0.901] | 0.816 [0.689, 0.934] |
| Swin-Tiny (2D) | BigLunge (test) | 0.664 [0.490, 0.817] | 0.685 [0.503, 0.844] | 0.460 [0.257, 0.645] |
