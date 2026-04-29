# 2D pipeline — headline test results

Patient-level metrics on each held-out test split. CIs are stratified bootstrap (n_boot=1000) on the patient-level predictions. AUC is one-vs-rest, macro-averaged.


## Overall metrics

| Model | Dataset | n | Accuracy | Balanced Acc | MacroF1 | MacroAUC |
|---|---|---|---|---|---|---|
| EffNet-B0 (2D) | Lung-PET-CT-Dx (test) | 53 | 0.717 [0.604, 0.830] | 0.568 [0.403, 0.751] | 0.554 [0.399, 0.718] | 0.823 [0.717, 0.914] |
| EffNet-B0 (2D) | BigLunge (test) | 43 | 0.465 [0.326, 0.605] | 0.462 [0.321, 0.605] | 0.452 [0.305, 0.599] | 0.688 [0.574, 0.797] |
| ResNet-50 (2D) | Lung-PET-CT-Dx (test) | 53 | 0.736 [0.623, 0.830] | 0.540 [0.429, 0.632] | 0.471 [0.386, 0.557] | 0.768 [0.642, 0.881] |
| ResNet-50 (2D) | BigLunge (test) | 43 | 0.419 [0.279, 0.558] | 0.410 [0.274, 0.550] | 0.404 [0.267, 0.547] | 0.651 [0.539, 0.759] |
| DenseNet121 (2D) | Lung-PET-CT-Dx (test) | 53 | 0.774 [0.679, 0.868] | 0.623 [0.456, 0.790] | 0.615 [0.452, 0.768] | 0.787 [0.661, 0.905] |
| DenseNet121 (2D) | BigLunge (test) | 43 | 0.395 [0.233, 0.535] | 0.395 [0.236, 0.545] | 0.393 [0.233, 0.536] | 0.576 [0.462, 0.692] |
| Swin-Tiny (2D) | Lung-PET-CT-Dx (test) | 53 | 0.755 [0.642, 0.868] | 0.689 [0.522, 0.854] | 0.654 [0.481, 0.807] | 0.835 [0.714, 0.930] |
| Swin-Tiny (2D) | BigLunge (test) | 43 | 0.442 [0.302, 0.581] | 0.432 [0.296, 0.573] | 0.426 [0.272, 0.564] | 0.632 [0.498, 0.754] |

## Per-class F1

| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |
|---|---|---|---|---|
| EffNet-B0 (2D) | Lung-PET-CT-Dx (test) | 0.849 [0.758, 0.932] | 0.286 [0.000, 0.588] | 0.526 [0.250, 0.762] |
| EffNet-B0 (2D) | BigLunge (test) | 0.541 [0.375, 0.686] | 0.417 [0.167, 0.640] | 0.400 [0.160, 0.615] |
| ResNet-50 (2D) | Lung-PET-CT-Dx (test) | 0.853 [0.767, 0.923] | 0.000 [0.000, 0.000] | 0.560 [0.364, 0.762] |
| ResNet-50 (2D) | BigLunge (test) | 0.412 [0.207, 0.595] | 0.300 [0.000, 0.522] | 0.500 [0.303, 0.688] |
| DenseNet121 (2D) | Lung-PET-CT-Dx (test) | 0.880 [0.806, 0.947] | 0.333 [0.000, 0.667] | 0.632 [0.374, 0.842] |
| DenseNet121 (2D) | BigLunge (test) | 0.437 [0.222, 0.621] | 0.370 [0.148, 0.571] | 0.370 [0.143, 0.581] |
| Swin-Tiny (2D) | Lung-PET-CT-Dx (test) | 0.857 [0.765, 0.933] | 0.545 [0.200, 0.833] | 0.560 [0.370, 0.737] |
| Swin-Tiny (2D) | BigLunge (test) | 0.424 [0.214, 0.606] | 0.353 [0.000, 0.600] | 0.500 [0.312, 0.667] |

## Per-class AUC (one-vs-rest)

| Model | Dataset | Adenocarcinoma | Small Cell | Squamous |
|---|---|---|---|---|
| EffNet-B0 (2D) | Lung-PET-CT-Dx (test) | 0.868 [0.768, 0.953] | 0.762 [0.578, 0.936] | 0.838 [0.697, 0.952] |
| EffNet-B0 (2D) | BigLunge (test) | 0.695 [0.538, 0.833] | 0.744 [0.572, 0.885] | 0.624 [0.433, 0.790] |
| ResNet-50 (2D) | Lung-PET-CT-Dx (test) | 0.846 [0.726, 0.944] | 0.716 [0.479, 0.926] | 0.742 [0.553, 0.904] |
| ResNet-50 (2D) | BigLunge (test) | 0.602 [0.431, 0.771] | 0.708 [0.556, 0.854] | 0.643 [0.469, 0.812] |
| DenseNet121 (2D) | Lung-PET-CT-Dx (test) | 0.904 [0.816, 0.972] | 0.681 [0.426, 0.919] | 0.775 [0.581, 0.937] |
| DenseNet121 (2D) | BigLunge (test) | 0.579 [0.398, 0.752] | 0.577 [0.392, 0.767] | 0.574 [0.390, 0.750] |
| Swin-Tiny (2D) | Lung-PET-CT-Dx (test) | 0.881 [0.782, 0.961] | 0.787 [0.543, 0.979] | 0.838 [0.699, 0.942] |
| Swin-Tiny (2D) | BigLunge (test) | 0.600 [0.417, 0.769] | 0.695 [0.490, 0.864] | 0.602 [0.402, 0.783] |
